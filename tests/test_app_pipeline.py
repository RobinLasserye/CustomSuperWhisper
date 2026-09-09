"""Application boundary: exactly one copy/paste, including raw fallback."""
import importlib
import os
from types import SimpleNamespace

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from sw import backends, runtime


@pytest.fixture
def module(monkeypatch):
    monkeypatch.setattr(runtime, "ensure_cuda_libs", lambda *_: False)
    return importlib.import_module("superwhisper")


class Signal:
    def __init__(self, callback=lambda *_: None):
        self.emissions = []
        self.callback = callback

    def emit(self, *args):
        self.emissions.append(args)
        self.callback(*args)


@pytest.mark.parametrize("engine", ["legacy", "langgraph"])
@pytest.mark.parametrize("mode,fail", [("disabled", False), ("message", False), ("message", True)])
@pytest.mark.parametrize("picker,auto_paste,picker_paste,expected_pastes", [
    (False, True, True, 1), (True, True, False, 0), (False, False, True, 0),
])
def test_delivery_once(module, monkeypatch, engine, mode, fail,
                       picker, auto_paste, picker_paste, expected_pastes):
    copies, pastes = [], []
    monkeypatch.setattr(module, "clipboard_copy", copies.append)
    monkeypatch.setattr(module, "auto_paste", lambda delay: pastes.append(delay))

    class ImmediateThread:
        def __init__(self, target, args=(), **kwargs):
            self.target, self.args = target, args
        def start(self):
            self.target(*self.args)

    monkeypatch.setattr(module.threading, "Thread", ImmediateThread)

    class Backend:
        def reformat(self, text, prompt):
            if fail:
                raise backends.ReformatError("offline")
            return "formatted"

    monkeypatch.setattr(module.pipeline, "_factory", lambda *_: Backend())
    app = SimpleNamespace(
        config={"pipeline": engine, "auto_paste": auto_paste,
                "auto_paste_after_picker": picker_paste},
        is_processing=True, icon_idle=None,
        tray=SimpleNamespace(setIcon=lambda *_: None),
        overlay=SimpleNamespace(show_done=lambda: None),
    )
    app.signals = SimpleNamespace(
        transcription_done=Signal(lambda text, from_picker:
                                  module.SuperWhisper._on_transcription_done(app, text, from_picker)),
        execution_done=Signal(), warning=Signal(), reformulation_started=Signal(),
    )
    module.SuperWhisper._reformat_and_finish(app, "raw", mode, "none", picker)
    expected = "raw" if mode == "disabled" or fail else "formatted"
    assert copies == [expected]
    assert len(pastes) == expected_pastes
    assert len(app.signals.transcription_done.emissions) == 1
    assert len(app.signals.execution_done.emissions) == 1
    assert not app.is_processing


def test_graph_preload_never_uses_legacy_network_path(module, monkeypatch):
    calls = []
    monkeypatch.setattr(backends.OllamaBackend, "warm_up", lambda _: calls.append(True))
    app = SimpleNamespace(config={"pipeline": "langgraph", "reformat_mode": "message"})
    module.SuperWhisper._warm_up_backend(app)
    assert not calls


def test_history_failure_does_not_prevent_delivery(module, monkeypatch):
    class BrokenStore:
        def save(self, *args, **kwargs):
            raise OSError('disk full')
    monkeypatch.setattr(module.pipeline, 'run_pipeline', lambda *_: module.pipeline.Execution(
        'legacy', 'synthetic', 'disabled', 'none', 'ollama', 'test', output='delivered'))
    app = SimpleNamespace(config={'history_enabled': True}, history_store=BrokenStore(),
                          signals=SimpleNamespace(execution_done=Signal(), transcription_done=Signal(),
                                                  warning=Signal(), reformulation_started=Signal()))
    module.SuperWhisper._reformat_and_finish(app, 'synthetic', 'disabled', 'none', False)
    assert app.signals.transcription_done.emissions == [('delivered', False)]
    assert app.signals.execution_done.emissions[0][0].metrics["history_saved"] is False
