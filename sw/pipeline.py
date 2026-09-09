"""Selectable post-transcription pipelines, with an in-memory execution report.

Whisper + artifact filtering + vocabulary corrections run upstream, once. These
pipelines have no clipboard, UI or persistence effects; the application delivers
exactly one result. Optional framework imports keep the legacy path usable.
"""
from dataclasses import dataclass, field
from time import perf_counter
from typing import TypedDict

from . import backends, langcheck, presets

UNCERTAIN = "Traduction incertaine — vérifie la langue"


@dataclass
class Step:
    node: str
    duration_ms: float
    before: str
    after: str
    detail: str
    attempt: int = 0
    prompt: str = ""


@dataclass
class Execution:
    pipeline: str
    source: str
    mode: str
    language: str
    backend: str
    model: str
    output: str = ""
    warning: str | None = None
    steps: list[Step] = field(default_factory=list)


class DictationState(TypedDict):
    source: str
    candidate: str
    prompt: str
    attempt: int
    route: str
    warning: str | None
    error: str | None


def _factory(config, name, engine):
    if engine == "langgraph" and name != "claude":
        from .langchain_backend import LangChainOllamaBackend
        return LangChainOllamaBackend.from_config(config)
    return backends.build_backend(config, name)


def run_pipeline(config, text, mode, target_language, *, backend_factory=None):
    """Return a report and safe output, including dependency/model failures."""
    engine = "langgraph" if config.get("pipeline") == "langgraph" else "legacy"
    effective_mode = presets.resolve_effective_mode(mode, target_language)
    language = presets.effective_language(effective_mode, target_language)
    name = presets.mode_backend(config, effective_mode)
    if name not in backends.BACKENDS:
        name = "ollama"
    report = Execution(engine, text, effective_mode, language, name,
                       config.get("ollama_model", "") if name == "ollama" else "Claude Code")
    prompt = presets.resolve(config, mode, target_language)
    factory = backend_factory or _factory
    try:
        if engine == "langgraph":
            # Disable tracing even if the parent process exports LangSmith credentials.
            from langsmith import tracing_context
            with tracing_context(enabled=False):
                _run_graph(report, config, prompt, factory)
        else:
            _run_legacy(report, config, prompt, factory)
    except Exception as exc:
        report.output = text
        report.warning = _error_message(exc)
        report.steps.append(Step("fallback", 0, text, text, report.warning))
    return report


def _error_message(exc):
    if isinstance(exc, backends.ReformatError):
        return str(exc)
    if isinstance(exc, ImportError):
        return "Dépendances LangChain/LangGraph absentes — texte brut conservé"
    # Don't put exception payloads (possibly model prompts/headers) in the report.
    return "Reformulation impossible — texte brut conservé"


def _run_graph(report, config, prompt, factory):
    from langgraph.graph import END, START, StateGraph

    backend = None

    def prepare(state):
        return {"route": "format" if state["prompt"] else "finish"}

    def format_text(state):
        nonlocal backend
        attempt = state["attempt"] + 1
        try:
            if backend is None:
                backend = factory(config, report.backend, "langgraph")
            result = backend.reformat(state["source"], state["prompt"])
            if not result.strip():
                raise backends.ReformatError("Réponse vide — texte brut conservé")
            return {"candidate": result, "attempt": attempt, "route": "validate"}
        except Exception as exc:
            return {"error": _error_message(exc), "attempt": attempt, "route": "fallback"}

    def validate(state):
        verdict = (langcheck.looks_like(state["candidate"], report.language)
                   if presets.is_translating(report.language) else None)
        if verdict is False:
            return {"route": "retry" if state["attempt"] < 2 else "finish",
                    "warning": UNCERTAIN if state["attempt"] >= 2 else None}
        return {"route": "finish", "warning": None}

    def retry(state):
        return {"prompt": state["prompt"] + backends.RETRY_DIRECTIVE, "route": "format"}

    def fallback(state):
        # Like the existing backend: a failed retry keeps the first candidate.
        return {"candidate": state["candidate"] or state["source"],
                "warning": UNCERTAIN if state["candidate"] else state["error"],
                "route": "finish"}

    def finish(state):
        return {"candidate": state["candidate"] or state["source"], "route": "END"}

    def measured(node, function):
        def invoke(state):
            start = perf_counter()
            update = function(state)
            after = {**state, **update}
            detail = f"→ {after['route']}"
            if node == "validate":
                verdict = (langcheck.looks_like(after["candidate"], report.language)
                           if presets.is_translating(report.language) else None)
                detail = ("Langue non demandée" if not presets.is_translating(report.language)
                          else {True: "Langue plausible", False: "Langue non respectée",
                                None: "Langue indéterminée"}[verdict]) + " " + detail
            if node == "retry":
                detail = "Consigne renforcée ; une seule nouvelle tentative → format"
            if after.get("error") and node in ("format", "fallback"):
                detail = after["error"] + " " + detail
            report.steps.append(Step(node, (perf_counter() - start) * 1000,
                                     state["source"] if node == "format" else
                                     state["candidate"] or state["source"],
                                     after["candidate"] or state["source"], detail,
                                     after["attempt"],
                                     state["prompt"] if node == "format" else ""))
            return update
        return invoke

    builder = StateGraph(DictationState)
    for node, function in (("prepare", prepare), ("format", format_text),
                           ("validate", validate), ("retry", retry),
                           ("fallback", fallback), ("finish", finish)):
        builder.add_node(node, measured(node, function))
    builder.add_edge(START, "prepare")
    for node, routes in (("prepare", ["format", "finish"]),
                         ("format", ["validate", "fallback"]),
                         ("validate", ["retry", "finish"])):
        builder.add_conditional_edges(node, lambda state: state["route"], routes)
    builder.add_edge("retry", "format")
    builder.add_edge("fallback", "finish")
    builder.add_edge("finish", END)
    result = builder.compile().invoke(
        {"source": report.source, "candidate": "", "prompt": prompt or "",
         "attempt": 0, "route": "", "warning": None, "error": None},
        config={"recursion_limit": 16, "callbacks": []})
    report.output = result["candidate"]
    report.warning = result["warning"]


def _run_legacy(report, config, prompt, factory):
    report.steps.append(Step("prepare", 0, report.source, report.source,
                             "Pipeline historique ; transcription et nettoyage déjà effectués"))
    report.output = report.source
    if prompt:
        backend = factory(config, report.backend, "legacy")

        class ObservedBackend:
            attempt = 0

            def reformat(self, text, system_prompt):
                self.attempt += 1
                start = perf_counter()
                result = text
                detail = "Appel du backend historique"
                try:
                    result = backend.reformat(text, system_prompt)
                    return result
                except Exception:
                    detail = "Échec du backend historique"
                    raise
                finally:
                    report.steps.append(Step("format", (perf_counter() - start) * 1000,
                                             text, result, detail, self.attempt, system_prompt))

        report.output, report.warning = backends.reformat(
            ObservedBackend(), report.source, prompt, report.language)
    report.steps.append(Step("finish", 0, report.source, report.output,
                             report.warning or "Résultat prêt"))
