"""Allowlisted model counters. Missing measurements stay missing, never zero."""
import math
import threading


def model_metrics(metadata):
    result = {}
    if not isinstance(metadata, dict):
        return result
    for source, target in (('prompt_eval_count', 'input_tokens'), ('eval_count', 'output_tokens')):
        value = metadata.get(source)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0:
            result[target] = value
    for name in ('total_duration', 'load_duration', 'prompt_eval_duration', 'eval_duration'):
        value = metadata.get(name)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0:
            result[name + '_ms'] = value / 1_000_000
    for count, duration, target in (('output_tokens', 'eval_duration_ms', 'output_tokens_per_second'),
                                    ('input_tokens', 'prompt_eval_duration_ms', 'input_tokens_per_second')):
        if count in result and result.get(duration, 0) > 0:
            result[target] = result[count] * 1000 / result[duration]
    return result


def _number(value):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else None


class SystemMetricsSampler:
    """Best-effort local process/system sampler; never raises or sends data."""

    def __init__(self, interval=0.1, gpu_index=0):
        self.interval = interval
        self.gpu_index = gpu_index
        self._stop = threading.Event()
        self._thread = None
        self._samples = []

    @staticmethod
    def _sample_gpu(index):
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(int(index))
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            sample = {
                "gpu_utilization_percent": _number(util.gpu),
                "gpu_memory_utilization_percent": _number(util.memory),
                "gpu_memory_used_bytes": _number(memory.used),
                "gpu_memory_total_bytes": _number(memory.total),
            }
            for key, fn in (("gpu_temperature_c", pynvml.nvmlDeviceGetTemperature),
                            ("gpu_power_watts", lambda h, sensor=pynvml.NVML_TEMPERATURE_GPU: pynvml.nvmlDeviceGetPowerUsage(h) / 1000)):
                try:
                    sample[key] = _number(fn(handle, sensor=0) if key.endswith("temperature_c") else fn(handle))
                except Exception:
                    pass
            return sample
        except Exception:
            return {}

    def sample(self):
        result = {}
        try:
            import psutil
            process = psutil.Process()
            result.update({"process_rss_bytes": _number(process.memory_info().rss),
                           "process_cpu_user_seconds": _number(process.cpu_times().user),
                           "process_cpu_system_seconds": _number(process.cpu_times().system),
                           "system_cpu_percent": _number(psutil.cpu_percent(None)),
                           "system_memory_percent": _number(psutil.virtual_memory().percent)})
        except Exception:
            pass
        result.update(self._sample_gpu(self.gpu_index))
        return result

    def _run(self):
        while not self._stop.is_set():
            self._samples.append(self.sample())
            self._stop.wait(self.interval)

    def start(self):
        self._samples = []
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="superwhisper-metrics", daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=max(1.0, self.interval * 3))
        self._samples.append(self.sample())
        keys = sorted({key for sample in self._samples for key in sample})
        result = {"samples": len(self._samples), "sampling_interval_ms": self.interval * 1000}
        for key in keys:
            values = [sample[key] for sample in self._samples if _number(sample.get(key)) is not None]
            if values:
                result[key] = values[-1]
                result[key.replace("percent", "peak_percent").replace("bytes", "peak_bytes")] = max(values)
        return result


def local_trace(name, attributes=None):
    """Optional OpenTelemetry span, configured with no exporter and no content."""
    try:
        from opentelemetry import trace
        tracer = trace.get_tracer("superwhisper.local")
        return tracer.start_as_current_span(name, attributes=attributes or {})
    except Exception:
        from contextlib import nullcontext
        return nullcontext()


def sanitize_attributes(attributes):
    """Allow only scalar, non-content telemetry attributes."""
    result = {}
    for key, value in (attributes or {}).items():
        if not isinstance(key, str) or key in {"prompt", "transcript", "audio", "content", "output"}:
            continue
        if isinstance(value, (str, int, float, bool)) and (not isinstance(value, float) or math.isfinite(value)):
            result[key] = value
    return result


from functools import lru_cache


@lru_cache(maxsize=1)
def runtime_versions():
    from importlib.metadata import PackageNotFoundError, version
    import platform
    result = {'python': platform.python_version(), 'os': platform.system(), 'architecture': platform.machine()}
    for package in ('faster-whisper', 'ctranslate2', 'langchain-ollama', 'langgraph', 'av'):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            pass
    return result
