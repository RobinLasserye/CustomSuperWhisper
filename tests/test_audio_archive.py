import numpy as np
from sw import audio_archive


def test_original_roundtrip_is_bit_exact():
    rng = np.random.default_rng(7)
    samples = rng.uniform(-.3, .3, 16000).astype(np.float32)
    data = audio_archive.encode(samples, 16000, 'lossless')
    restored = audio_archive.decode(data, 16000, 'lossless')
    assert np.array_equal(samples, restored)


def test_opus_is_compact_decodable_and_preserves_duration():
    samples = (.2 * np.sin(2*np.pi*440*np.arange(16000)/16000)).astype(np.float32)
    data = audio_archive.encode(samples, 16000, 'opus')
    restored = audio_archive.decode(data, 16000, 'opus')
    assert len(data) < samples.nbytes / 4
    assert abs(len(restored) - len(samples)) <= 1
    assert np.sqrt(np.mean((samples - restored[:len(samples)])**2)) < .03
