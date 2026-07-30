"""Chargement, valeurs par défaut et migration de la configuration.

La migration est le point sensible : la config réelle contient des consignes personnalisées
écrites à la main, elles ne doivent pas disparaître au passage à la nouvelle version.
"""

import json

from sw import config as config_module

# Extrait fidèle de l'ancienne configuration (schéma « claude_* »)
LEGACY = {
    "model": "large-v3",
    "language": "fr",
    "compute_type": "float16",
    "gpu_index": "0",
    "audio_device": "default",
    "claude_mode": "message",
    "claude_custom_modes": [{"name": "Tweet", "prompt": "Fais court"}],
    "claude_prompt_overrides": {
        "message": "Ma consigne personnalisée pour les messages",
        "dev": "Ma consigne personnalisée pour le dev",
    },
}


def write(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return str(path)


# ─── Détection ────────────────────────────────────────────────────────────────

def test_detecte_l_ancien_schema():
    assert config_module.needs_migration(LEGACY)
    assert not config_module.needs_migration({"reformat_mode": "message"})


# ─── Migration en mémoire ─────────────────────────────────────────────────────

def test_migration_conserve_le_mode_et_les_consignes():
    migrated = config_module.migrate(LEGACY)
    assert migrated["reformat_mode"] == "message"
    assert migrated["reformat_prompt_overrides"]["message"] == \
        "Ma consigne personnalisée pour les messages"
    assert migrated["reformat_prompt_overrides"]["dev"] == \
        "Ma consigne personnalisée pour le dev"
    assert migrated["reformat_custom_modes"][0]["name"] == "Tweet"


def test_migration_retire_les_anciennes_cles():
    migrated = config_module.migrate(LEGACY)
    for key in ("claude_mode", "claude_prompt_overrides", "claude_custom_modes"):
        assert key not in migrated


def test_migration_bascule_les_modes_personnalises_sur_le_local():
    migrated = config_module.migrate(LEGACY)
    assert migrated["reformat_custom_modes"][0]["backend"] == "ollama"


def test_migration_ne_touche_pas_a_l_original():
    original = json.loads(json.dumps(LEGACY))
    config_module.migrate(LEGACY)
    assert LEGACY == original


def test_une_valeur_deja_migree_prime():
    source = dict(LEGACY, reformat_mode="github")
    assert config_module.migrate(source)["reformat_mode"] == "github"


# ─── Valeurs par défaut ───────────────────────────────────────────────────────

def test_les_defauts_completent_sans_ecraser():
    config = config_module.apply_defaults({"model": "small"})
    assert config["model"] == "small"
    assert config["reformat_backend"] == "ollama"
    assert config["cloud_rule_enabled"] is True
    assert config["vocabulary"]


def test_les_listes_par_defaut_ne_sont_pas_partagees():
    first = config_module.apply_defaults({})
    second = config_module.apply_defaults({})
    first["vocabulary"].append("PIÈGE")
    assert "PIÈGE" not in second["vocabulary"]
    assert "PIÈGE" not in config_module.DEFAULT_CONFIG["vocabulary"]


def test_le_modele_local_par_defaut_est_celui_du_catalogue():
    from sw import models_catalog
    assert config_module.DEFAULT_CONFIG["ollama_model"] == models_catalog.DEFAULT_LLM_MODEL


# ─── Chargement depuis le disque ──────────────────────────────────────────────

def test_load_config_migre_a_la_lecture(tmp_path):
    path = write(tmp_path / "config.json", LEGACY)
    config = config_module.load_config(path)
    assert config["reformat_mode"] == "message"
    assert "claude_mode" not in config
    assert config["ollama_host"] == "http://127.0.0.1:11434"


def test_load_config_absent_donne_les_defauts(tmp_path):
    config = config_module.load_config(str(tmp_path / "absent.json"))
    assert config["model"] == config_module.DEFAULT_CONFIG["model"]


def test_load_config_illisible_ne_leve_pas(tmp_path):
    path = tmp_path / "casse.json"
    path.write_text("{ ceci n'est pas du json", encoding="utf-8")
    config = config_module.load_config(str(path))
    assert config["model"] == config_module.DEFAULT_CONFIG["model"]


def test_aller_retour_sauvegarde(tmp_path):
    path = str(tmp_path / "config.json")
    config = config_module.load_config(path)
    config["reformat_mode"] = "github"
    config_module.save_config(config, path)
    assert config_module.load_config(path)["reformat_mode"] == "github"


def test_sauvegarde_lisible_par_un_humain(tmp_path):
    path = tmp_path / "config.json"
    config_module.save_config({"vocabulary": ["Créé", "Ollama"]}, str(path))
    contenu = path.read_text(encoding="utf-8")
    assert "Créé" in contenu            # pas d'échappement \uXXXX
    assert "\n" in contenu              # indenté


# ─── Migration du fichier ─────────────────────────────────────────────────────

def test_migrate_file_sauvegarde_avant_de_reecrire(tmp_path):
    path = write(tmp_path / "config.json", LEGACY)
    backup = config_module.migrate_file(path)

    assert backup and "bak-" in backup
    with open(backup, encoding="utf-8") as handle:
        assert json.load(handle) == LEGACY

    with open(path, encoding="utf-8") as handle:
        migrated = json.load(handle)
    assert migrated["reformat_mode"] == "message"
    assert "claude_mode" not in migrated
    assert migrated["reformat_prompt_overrides"]["dev"].startswith("Ma consigne")


def test_migrate_file_ne_fait_rien_sur_un_fichier_a_jour(tmp_path):
    path = write(tmp_path / "config.json", {"reformat_mode": "github"})
    assert config_module.migrate_file(path) is None


def test_migrate_file_ne_fait_rien_si_absent(tmp_path):
    assert config_module.migrate_file(str(tmp_path / "absent.json")) is None
