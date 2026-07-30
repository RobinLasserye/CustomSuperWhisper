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


def _merge_legacy(current, legacy):
    """Fusionne une valeur héritée avec la valeur déjà présente sous le nouveau nom.

    La valeur nouvelle gagne, mais **clé par clé** : une nouvelle clé vide ne doit pas jeter les
    consignes écrites à la main sous l'ancien nom. C'est le cas qui se produit quand une version
    précédente de l'application réécrit le fichier alors que le nouveau schéma existe déjà.
    """
    if isinstance(current, dict) and isinstance(legacy, dict):
        merged = dict(legacy)
        merged.update(current)
        return merged
    if isinstance(current, list) and isinstance(legacy, list):
        if not current:
            return list(legacy)
        known = {item.get("name") for item in current if isinstance(item, dict)}
        extra = [item for item in legacy
                 if not (isinstance(item, dict) and item.get("name") in known)]
        return current + extra
    if current in (None, "", {}, []):
        return legacy
    return current


def migrate(raw):
    """Convertit une config d'une version précédente. Ne perd aucun prompt personnalisé."""
    config = copy.deepcopy(raw)
    for old_key, new_key in LEGACY_KEYS.items():
        if old_key not in config:
            continue
        legacy = config.pop(old_key)
        if new_key in config:
            config[new_key] = _merge_legacy(config[new_key], legacy)
        else:
            config[new_key] = legacy

    # Les modes personnalisés passaient tous par Claude ; ils basculent sur le backend par défaut
    # mais gardent leur prompt.
    for custom in config.get("reformat_custom_modes", []) or []:
        if isinstance(custom, dict):
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
    """Écriture atomique : le fichier existant n'est remplacé qu'une fois le nouveau complet.

    Un `open(path, "w")` tronque immédiatement le fichier ; une erreur de sérialisation ou une
    coupure en cours d'écriture laisserait alors une config amputée — donc les consignes écrites à
    la main perdues, silencieusement. On sérialise d'abord en mémoire, on écrit à côté, puis on
    remplace.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = json.dumps(config, indent=2, ensure_ascii=False)
    temporary = f"{path}.tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


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

    payload = json.dumps(raw, indent=2, ensure_ascii=False)
    backup = None
    for candidate in (f"{path}.bak-{time.strftime('%Y%m%d-%H%M%S')}", f"{path}.bak"):
        try:
            with open(candidate, "w", encoding="utf-8") as handle:
                handle.write(payload)
            backup = candidate
            break
        except OSError as exc:
            log(f"sauvegarde impossible dans {candidate} ({exc})")

    if backup is None:
        # Sans copie de secours on ne réécrit pas le fichier. La config est tout de même migrée en
        # mémoire par load_config, donc on prévient bruyamment : le prochain enregistrement
        # convertira le fichier sans filet.
        log("ATTENTION : aucune sauvegarde n'a pu être créée, le fichier n'est pas converti — "
            "vérifie les droits sur le dossier de configuration")
        return None

    save_config(apply_defaults(migrate(raw)), path)
    log(f"config migrée vers le nouveau schéma (sauvegarde : {backup})")
    return backup
