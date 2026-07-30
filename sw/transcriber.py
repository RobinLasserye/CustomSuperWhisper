"""Transcription via faster-whisper, avec biais vocabulaire et post-traitement."""

import gc
import os

from . import artifacts, vocabulary
from .runtime import log

AVAILABLE_MODELS = [
    "large-v3", "large-v3-turbo", "distil-large-v3",
    "medium", "small", "base", "tiny",
]

# Libellé affiché → code passé à Whisper
WHISPER_LANGUAGES = [
    ("Français", "fr"), ("Anglais", "en"), ("Espagnol", "es"),
    ("Allemand", "de"), ("Italien", "it"), ("Portugais", "pt"),
    ("Néerlandais", "nl"), ("Japonais", "ja"), ("Chinois", "zh"),
    ("Russe", "ru"), ("Arabe", "ar"), ("Auto-détection", None),
]

COMPUTE_TYPES = [("float16 — rapide", "float16"), ("int8 — léger en VRAM", "int8")]


class Transcriber:
    def __init__(self):
        self.model = None
        self.current_model_name = None
        self.current_gpu = None
        self.current_compute_type = None

    # — Cycle de vie —

    def load_model(self, config):
        gpu = config.get("gpu_index", "0")
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
        model_name = config["model"]
        compute_type = config.get("compute_type", "float16")
        if (self.model is not None and self.current_model_name == model_name
                and self.current_gpu == gpu and self.current_compute_type == compute_type):
            return
        from faster_whisper import WhisperModel
        log(f"chargement du modèle {model_name} ({compute_type}) sur le GPU {gpu}")
        self.model = WhisperModel(model_name, device="cuda", compute_type=compute_type)
        self.current_model_name = model_name
        self.current_gpu = gpu
        self.current_compute_type = compute_type

    def unload(self):
        """Libère la VRAM. Utile sur une petite carte, entre deux dictées."""
        if self.model is None:
            return False
        log("déchargement du modèle Whisper (inactivité)")
        self.model = None
        self.current_model_name = None
        self.current_compute_type = None
        gc.collect()
        return True

    @property
    def is_loaded(self):
        return self.model is not None

    # — Transcription —

    def transcribe_segments(self, audio, config):
        if self.model is None:
            self.load_model(config)

        kwargs = {}
        if config.get("vocab_biasing", True):
            terms = config.get("vocabulary") or []
            hotwords = vocabulary.build_hotwords(terms)
            initial_prompt = vocabulary.build_initial_prompt(terms)
            if hotwords:
                kwargs["hotwords"] = hotwords
            if initial_prompt:
                kwargs["initial_prompt"] = initial_prompt

        segments, _info = self.model.transcribe(
            audio, language=config.get("language"), beam_size=5, vad_filter=True, **kwargs)
        return list(segments)

    def transcribe(self, audio, config):
        """Retourne (texte final, artefacts retirés)."""
        segments = self.transcribe_segments(audio, config)
        return postprocess(segments, config)


def postprocess(segments, config):
    """Filtre les hallucinations puis applique les corrections de vocabulaire."""
    text, removed = artifacts.filter_transcription(segments, config)
    text = vocabulary.correct(text, config)
    return text.strip(), removed
