"""Enregistrement audio via sounddevice (multi-plateforme)."""

import numpy as np

SAMPLE_RATE = 16000
NUM_BARS = 40


class AudioRecorder:
    def __init__(self, signals, device=None):
        self.signals = signals
        self.device = device
        self.recording = False
        self.frames = []
        self.stream = None

    def start(self):
        import sounddevice as sd
        self.frames = []
        self.recording = True

        # `default` (ou toute valeur non numérique) → device=None, donc le micro par défaut du
        # système. Un index numérique n'est PAS stable entre deux redémarrages : les devices
        # matériels bruts refusent le 16 kHz et font échouer l'enregistrement.
        device = None
        if self.device and self.device != "default":
            try:
                device = int(self.device)
            except (TypeError, ValueError):
                device = None

        def callback(indata, _frame_count, _time_info, _status):
            if not self.recording:
                return
            samples = indata[:, 0].copy()
            self.frames.append(samples)
            spectrum = np.abs(np.fft.rfft(samples))[:NUM_BARS]
            peak = spectrum.max()
            if peak > 0:
                spectrum = spectrum / peak
            self.signals.audio_level.emit(spectrum)

        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="float32",
            device=device, blocksize=1024, callback=callback)
        self.stream.start()

    def stop(self):
        self.recording = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        if self.frames:
            return np.concatenate(self.frames)
        return np.array([], dtype="float32")
