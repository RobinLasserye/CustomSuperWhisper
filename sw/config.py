"""Configuration : valeurs par défaut, chargement, sauvegarde et migration."""

import copy
import json
import os
import time

from . import artifacts, models_catalog, vocabulary
from .runtime import CONFIG_PATH, log

DEFAULT_CONFIG = {
    # Transcription
    "model": "large-v3",
    "language": "fr",
    "compute_type": "float16",
    "gpu_index": "0",
    "audio_device": "default",
    "whisper_idle_unload_min": 0,          # 0 = ne jamais décharger

    # Vocabulaire et corrections
    "vocab_biasing": True,
    "vocabulary": list(vocabulary.DEFAULT_VOCABULARY),
    "corrections_enabled": True,
    "corrections": copy.deepcopy(vocabulary.DEFAULT_CORRECTIONS),
    "cloud_rule_enabled": True,
    "cloud_exceptions": list(vocabulary.DEFAULT_CLOUD_EXCEPTIONS),

    # Filtre d'hallucinations
    "artifact_filter": True,
    "artifact_patterns": list(artifacts.DEFAULT_ARTIFACT_PATTERNS),
    "artifact_ambiguous_enabled": True,
    "artifact_ambiguous": list(artifacts.DEFAULT_AMBIGUOUS_PATTERNS),
    "artifact_no_speech_threshold": artifacts.DEFAULT_NO_SPEECH_THRESHOLD,
    "artifact_logprob_threshold": artifacts.DEFAULT_LOGPROB_THRESHOLD,
    "collapse_repetitions": True,

    # Reformulation
    "reformat_mode": "disabled",
    "reformat_backend": "ollama",
    "reformat_mode_backends": {},
    "reformat_prompt_overrides": {},
    "reformat_custom_modes": [],
    "target_language": "none",

    # Backend local
    "ollama_host": "http://127.0.0.1:11434",
    "ollama_model": models_catalog.DEFAULT_LLM_MODEL,
    "ollama_keep_alive": "30m",
    "ollama_timeout_s": 60,
    "ollama_num_ctx": 8192,
    "ollama_temperature": 0.2,

    # Comportement
    "auto_paste": True,
    "auto_paste_after_picker": True,
    "picker_shows_language": True,
}

# Anciennes clés → nouvelles clés
LEGACY_KEYS = {
    "claude_mode": "reformat_mode",
    "claude_prompt_overrides": "reformat_prompt_overrides",
    "claude_custom_modes": "reformat_custom_modes",
}


def needs_migration(raw):
    return any(key in raw for key in LEGACY_KEYS)


def migrate(raw):
    """Convertit une config d'une version précédente. Ne perd aucun prompt personnalisé."""
    config = dict(raw)
    for old_key, new_key in LEGACY_KEYS.items():
        if old_key in config:
            value = config.pop(old_key)
            # Une valeur déjà présente sous le nouveau nom est prioritaire
            config.setdefault(new_key, value)

    # Les modes personnalisés passaient tous par Claude ; ils basculent sur le backend par défaut
    # mais gardent leur prompt.
    for custom in config.get("reformat_custom_modes", []) or []:
        custom.setdefault("backend", "ollama")

    return config


def apply_defaults(config):
    for key, value in DEFAULT_CONFIG.items():
        if key not in config:
            config[key] = copy.deepcopy(value)
    return config


def load_config(path=CONFIG_PATH):
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as handle:
                raw = json.load(handle)
        except (OSError, ValueError) as exc:
            log(f"config illisible ({exc}) — valeurs par défaut utilisées")
            raw = {}
    else:
        raw = {}
    return apply_defaults(migrate(raw))


def save_config(config, path=CONFIG_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, ensure_ascii=False)


def migrate_file(path=CONFIG_PATH):
    """Réécrit la config sur le disque si elle utilise l'ancien schéma, après sauvegarde.

    Retourne le chemin de la sauvegarde, ou None s'il n'y avait rien à migrer.
    """
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as handle:
            raw = json.load(handle)
    except (OSError, ValueError):
        return None
    if not needs_migration(raw):
        return None

    backup = f"{path}.bak-{time.strftime('%Y%m%d-%H%M%S')}"
    try:
        with open(backup, "w", encoding="utf-8") as handle:
            json.dump(raw, handle, indent=2, ensure_ascii=False)
    except OSError as exc:
        log(f"sauvegarde de la config impossible ({exc}) — migration annulée")
        return None

    save_config(apply_defaults(migrate(raw)), path)
    log(f"config migrée vers le nouveau schéma (sauvegarde : {backup})")
    return backup
