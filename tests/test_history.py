import os
import pytest
from sw import pipeline


def test_history_persists_feedback_and_clears(tmp_path):
    from sw.history import HistoryStore
    path = tmp_path / 'private' / 'history.sqlite3'
    store = HistoryStore(path)
    report = pipeline.Execution('langgraph', 'synthetic input', 'message', 'en', 'ollama', 'test')
    report.output = 'synthetic output'
    store.save(report)
    reopened = HistoryStore(path)
    restored = reopened.recent()[0]
    assert restored.source == report.source
    assert restored.run_id == report.run_id
    reopened.review(report.run_id, 'incorrect', 'synthetic correction')
    assert reopened.recent()[0].review == {'rating': 'incorrect', 'correction': 'synthetic correction'}
    reopened.clear()
    assert reopened.recent() == []
    if os.name != 'nt':
        assert path.stat().st_mode & 0o777 == 0o600
        assert path.parent.stat().st_mode & 0o777 == 0o700


def test_history_refuses_repository_directory(tmp_path):
    from sw.history import HistoryStore
    (tmp_path / '.git').mkdir()
    (tmp_path / '.git' / 'HEAD').write_text('ref: refs/heads/main')
    with pytest.raises(ValueError):
        HistoryStore(tmp_path / 'data' / 'history.sqlite3')


def test_metrics_whitelist_and_rates():
    from sw.metrics import model_metrics
    got = model_metrics({'eval_count': 20, 'eval_duration': 2_000_000_000,
                         'prompt_eval_count': 10, 'total_duration': 3_000_000_000,
                         'message': {'content': 'not retained'}, 'secret': 'not allowed'})
    assert got['output_tokens_per_second'] == 10
    assert got['total_duration_ms'] == 3000
    assert 'secret' not in got and 'message' not in got
    assert 'output_tokens_per_second' not in model_metrics({'eval_count': 20, 'eval_duration': 0})


@pytest.mark.parametrize('engine', ['legacy', 'langgraph'])
def test_reports_capture_each_attempt_without_config_secrets(engine):
    class Backend:
        last_metrics = {}
        def reformat(self, text, prompt):
            self.last_metrics = {'output_tokens': 9}
            return 'hello'
    report = pipeline.run_pipeline({'pipeline': engine, 'ollama_model': 'test', 'secret': 'hidden'},
                                   'input', 'message', 'none', backend_factory=lambda *_: Backend())
    assert report.metrics['pipeline_duration_ms'] >= 0
    assert report.steps[1].metrics['output_tokens'] == 9
    assert 'secret' not in report.settings
    assert report.created_at and report.run_id


def test_audio_is_lossless_and_removed_with_history(tmp_path):
    import numpy as np
    from sw.history import HistoryStore
    store = HistoryStore(tmp_path / 'private' / 'history.sqlite3')
    report = pipeline.Execution('legacy', 'synthetic', 'disabled', 'none', 'ollama', 'test')
    audio = np.array([0., -.23456789, .3456789], dtype=np.float32)
    store.save(report, audio=audio, sample_rate=16000)
    restored, sample_rate = store.audio(report.run_id)
    assert np.array_equal(audio, restored)
    assert sample_rate == 16000
    store.clear()
    assert store.audio(report.run_id) is None


def test_execution_dialog_reopens_persistent_reviews(tmp_path):
    from PySide6.QtWidgets import QApplication
    from sw.history import HistoryStore
    from sw.ui.execution import ExecutionDialog
    app = QApplication.instance() or QApplication([])
    store = HistoryStore(tmp_path / 'private' / 'history.sqlite3')
    report = pipeline.Execution('legacy', 'synthetic', 'disabled', 'none', 'ollama', 'test')
    report.output = 'result'
    store.save(report)
    dialog = ExecutionDialog(store=store)
    assert dialog.runs.count() == 1
    dialog.rating.setCurrentIndex(dialog.rating.findData('good'))
    dialog.correction.setPlainText('corrected example')
    dialog.save_review()
    assert store.recent()[0].review['rating'] == 'good'
    assert store.recent()[0].review['correction'] == 'corrected example'
    dialog.close()


def test_clear_overwrites_private_payload_and_delete_removes_audio(tmp_path):
    import numpy as np
    from sw.history import HistoryStore
    path = tmp_path / 'private' / 'history.sqlite3'
    store = HistoryStore(path)
    report = pipeline.Execution('legacy', 'PRIVATE_SYNTHETIC_SENTINEL_7654321', 'disabled', 'none', 'ollama', 'test')
    store.save(report, audio=np.zeros(50, dtype=np.float32))
    store.delete(report.run_id)
    assert store.audio(report.run_id) is None
    assert store.recent() == []
    assert b'PRIVATE_SYNTHETIC_SENTINEL_7654321' not in path.read_bytes()


def test_save_is_idempotent_and_preserves_review(tmp_path):
    from sw.history import HistoryStore
    store = HistoryStore(tmp_path / 'private' / 'history.sqlite3')
    report = pipeline.Execution('legacy', 'synthetic', 'disabled', 'none', 'ollama', 'test')
    store.save(report)
    store.review(report.run_id, 'good', 'correction')
    store.save(report)
    assert store.summary() == 1
    assert store.recent()[0].review['rating'] == 'good'


def test_statistics_excludes_text_and_counts_only_measured_runs(tmp_path):
    from sw.history import HistoryStore
    store = HistoryStore(tmp_path / 'private' / 'history.sqlite3')
    report = pipeline.Execution('legacy', 'PRIVATE_SYNTHETIC_TEXT', 'disabled', 'none', 'ollama', 'test')
    report.metrics['pipeline_duration_ms'] = 120
    store.save(report)
    stats = store.statistics()
    assert stats['runs'] == 1
    assert stats['pipeline_duration_ms']['median'] == 120
    assert 'PRIVATE_SYNTHETIC_TEXT' not in str(stats)


def test_failed_optional_audio_still_saves_text(tmp_path, monkeypatch):
    import numpy as np
    from sw.history import HistoryStore
    from sw import audio_archive
    def fail(*args):
        raise RuntimeError('encoder unavailable')
    monkeypatch.setattr(audio_archive, 'encode', fail)
    store = HistoryStore(tmp_path / 'private' / 'history.sqlite3')
    report = pipeline.Execution('legacy', 'synthetic retained', 'disabled', 'none', 'ollama', 'test')
    store.save(report, audio=np.zeros(10, dtype=np.float32), codec='opus')
    saved = store.recent()[0]
    assert saved.source == 'synthetic retained'
    assert saved.metrics['audio_storage']['saved'] is False
    assert store.audio(report.run_id) is None


def test_reference_transcript_is_distinct_from_formatted_target(tmp_path):
    from sw.history import HistoryStore
    store = HistoryStore(tmp_path / 'private' / 'history.sqlite3')
    report = pipeline.Execution('legacy', 'synthetic', 'message', 'none', 'ollama', 'test')
    store.save(report)
    store.review(report.run_id, 'incorrect', 'Formatted target.', 'spoken words')
    review = store.recent()[0].review
    assert review['correction'] == 'Formatted target.'
    assert review['reference_transcript'] == 'spoken words'
