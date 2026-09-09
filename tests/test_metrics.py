from sw.metrics import SystemMetricsSampler, sanitize_attributes


def test_sanitize_attributes_excludes_content_and_non_scalars():
    result = sanitize_attributes({
        "pipeline": "langgraph", "attempt": 1, "ok": True,
        "prompt": "secret", "transcript": "private", "nested": {"x": 1},
    })
    assert result == {"pipeline": "langgraph", "attempt": 1, "ok": True}


def test_sampler_collects_process_metrics_without_gpu_requirement():
    sampler = SystemMetricsSampler(interval=0.01).start()
    metrics = sampler.stop()
    assert metrics["samples"] >= 2
    assert metrics["sampling_interval_ms"] == 10.0
    # NVIDIA metrics are optional, but process metrics should be available when
    # psutil is installed (the supported installation path installs it).
    assert metrics["process_rss_bytes"] > 0
    assert metrics["process_rss_peak_bytes"] >= metrics["process_rss_bytes"]
