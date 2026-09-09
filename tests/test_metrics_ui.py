import json

from sw.ui.settings import format_metrics_summary


def test_metrics_summary_is_aggregate_only():
    text = format_metrics_summary(
        {"runs": 3, "pipeline_duration_ms": {"median": 12.0, "p95": 18.0}},
        {"history_enabled": True, "device_metrics_enabled": True,
         "history_audio_enabled": False, "history_audio_codec": "lossless"},
    )
    payload = json.loads(text)
    assert payload["aggregates"]["runs"] == 3
    assert "prompts" in payload["collection"]["excluded_from_telemetry"]
    assert "transcripts" in payload["collection"]["excluded_from_telemetry"]
