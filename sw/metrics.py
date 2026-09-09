"""Allowlisted model counters. Missing measurements stay missing, never zero."""
import math


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
