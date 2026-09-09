"""Real graph routing with deterministic model calls (no GPU or network)."""
import pytest

from sw import backends, config, pipeline

FR = "Le montant de la facture est faux et je voudrais comprendre la différence"
JA = "これは日本語のテストです。"


class Backend:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def reformat(self, text, prompt):
        self.calls.append((text, prompt))
        result = next(self.responses)
        if isinstance(result, Exception):
            raise result
        return result


def run(responses, language="ja", mode="message", engine="langgraph"):
    backend = Backend(responses)
    cfg = config.apply_defaults({"pipeline": engine})
    report = pipeline.run_pipeline(cfg, "raw", mode, language,
                                   backend_factory=lambda *_: backend)
    return report, backend


def test_disabled_does_not_build_model():
    report, backend = run([], "none", "disabled")
    assert report.output == "raw"
    assert not backend.calls
    assert [step.node for step in report.steps] == ["prepare", "finish"]


def test_graph_retry_is_bounded_and_uses_original_text():
    report, backend = run([FR, JA])
    assert report.output == JA and report.warning is None
    assert [step.node for step in report.steps] == [
        "prepare", "format", "validate", "retry", "format", "validate", "finish"]
    assert len(backend.calls) == 2
    assert all(text == "raw" for text, _ in backend.calls)
    assert backends.RETRY_DIRECTIVE in backend.calls[1][1]
    assert all(step.duration_ms >= 0 for step in report.steps)
    assert report.steps[1].before == "raw" and report.steps[1].after == FR


@pytest.mark.parametrize("engine", ["legacy", "langgraph"])
@pytest.mark.parametrize("responses,expected,warning", [
    ([JA], JA, False), ([FR, FR], FR, True),
    ([FR, backends.ReformatError("offline")], FR, True),
    ([backends.ReformatError("offline")], "raw", True),
    ([RuntimeError("unexpected")], "raw", True),
])
def test_legacy_and_graph_fallback_semantics(engine, responses, expected, warning):
    report, backend = run(responses, engine=engine)
    assert report.output == expected
    assert bool(report.warning) == warning
    assert len(backend.calls) <= 2


def test_indeterminate_language_does_not_retry():
    report, backend = run(["Hi"], "en")
    assert report.output == "Hi" and report.warning is None
    assert len(backend.calls) == 1
    assert "indéterminée" in next(s.detail for s in report.steps if s.node == "validate")


def test_translation_with_disabled_format_still_calls_model():
    report, backend = run([JA], mode="disabled")
    assert report.output == JA and len(backend.calls) == 1


def test_backend_construction_failure_keeps_raw():
    def fail(*_):
        raise ImportError("dependency missing")
    report = pipeline.run_pipeline({"pipeline": "langgraph"}, "raw", "message", "none",
                                   backend_factory=fail)
    assert report.output == "raw" and report.warning
    assert "fallback" in [step.node for step in report.steps]


@pytest.mark.parametrize("name", ["unknown", None])
def test_unknown_backend_cannot_bypass_graph_local_guard(name):
    from sw.langchain_backend import LangChainOllamaBackend
    backend = pipeline._factory({"ollama_host": "https://example.com", "ollama_model": "remote-cloud"},
                                name, "langgraph")
    assert isinstance(backend, LangChainOllamaBackend)
    with pytest.raises(backends.ReformatError):
        backend.reformat("private", "prompt")


def test_missing_framework_dependency_returns_raw(monkeypatch):
    import builtins
    original = builtins.__import__
    def missing(name, *args, **kwargs):
        if name == "langgraph.graph":
            raise ImportError("langgraph unavailable")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", missing)
    report = pipeline.run_pipeline({"pipeline": "langgraph"}, "raw", "message", "none")
    assert report.output == "raw"
    assert "Dépendances" in report.warning


def test_claude_remains_explicit_option_in_graph(monkeypatch):
    calls = []
    def build(config, name):
        calls.append(name)
        return Backend([JA])
    monkeypatch.setattr(backends, "build_backend", build)
    report = pipeline.run_pipeline({"pipeline": "langgraph", "reformat_backend": "claude"},
                                   "raw", "message", "ja")
    assert calls == ["claude"]
    assert report.output == JA and report.backend == "claude"
