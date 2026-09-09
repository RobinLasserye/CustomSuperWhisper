from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from types import SimpleNamespace
from sw import transcriber


def test_concurrent_transcriptions_keep_their_own_metrics(monkeypatch):
    barrier = Barrier(2)
    worker = transcriber.Transcriber()
    class Model:
        def transcribe(self, audio, **kwargs):
            return iter([SimpleNamespace(text=audio)]), SimpleNamespace(language=audio)
    worker.model = Model()
    def postprocess(segments, config):
        barrier.wait(timeout=5)
        return segments[0].text, []
    monkeypatch.setattr(transcriber, 'postprocess', postprocess)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda value: worker.transcribe(value, {}, with_metrics=True), ['A', 'B']))
    for text, removed, metrics in results:
        assert metrics['raw_transcript'] == text
        assert metrics['segments'][0]['text'] == text
        assert metrics['detected_language'] == text
