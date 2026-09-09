"""Local audio codecs. Lossless preserves input float32 bits; Opus is lossy."""
from io import BytesIO
import zlib


def encode(audio, sample_rate, codec):
    import numpy as np
    samples = np.asarray(audio, dtype='<f4')
    if samples.ndim != 1:
        raise ValueError('Mono audio expected')
    if codec == 'lossless':
        return zlib.compress(samples.tobytes(), level=6)
    if codec != 'opus':
        raise ValueError('Unknown audio codec')
    import av
    buffer = BytesIO()
    with av.open(buffer, mode='w', format='ogg') as container:
        stream = container.add_stream('libopus', rate=sample_rate)
        stream.layout = 'mono'
        stream.bit_rate = 32000
        stream.options = {'application': 'voip'}
        frame = av.AudioFrame.from_ndarray(samples.reshape(1, -1), format='flt', layout='mono')
        frame.sample_rate = sample_rate
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)
    return buffer.getvalue()


def decode(data, sample_rate, codec):
    import numpy as np
    if codec == 'lossless':
        return np.frombuffer(zlib.decompress(data), dtype='<f4').copy()
    if codec != 'opus':
        raise ValueError('Unknown audio codec')
    import av
    chunks = []
    resampler = av.AudioResampler(format='flt', layout='mono', rate=sample_rate)
    with av.open(BytesIO(data), mode='r') as container:
        for frame in container.decode(audio=0):
            for converted in resampler.resample(frame):
                chunks.append(converted.to_ndarray().reshape(-1))
        for converted in resampler.resample(None):
            chunks.append(converted.to_ndarray().reshape(-1))
    return np.concatenate(chunks) if chunks else np.empty(0, dtype=np.float32)
