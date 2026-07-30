"""Fenêtre de réglages : les chemins où une consigne écrite à la main peut disparaître.

Tourne hors écran. Chaque test correspond à un défaut réel trouvé en revue.
"""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QInputDialog        # noqa: E402

from sw import config as config_module, presets                 # noqa: E402
from sw.ui.settings import SettingsDialog                        # noqa: E402

CONSIGNE = "Ma consigne écrite à la main pour les messages"


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def config():
    base = config_module.apply_defaults({})
    base["reformat_prompt_overrides"] = {"message": CONSIGNE}
    base["reformat_mode"] = "message"
    return base


@pytest.fixture
def saved(monkeypatch, tmp_path):
    """Intercepte l'enregistrement pour inspecter ce qui serait écrit sur le disque."""
    captured = {}
    import sw.ui.settings as module
    monkeypatch.setattr(module, "save_config", lambda cfg, path=None: captured.update(cfg))
    return captured


def test_la_consigne_est_affichee(app, config):
    dialog = SettingsDialog(config)
    assert dialog.prompt_edit.toPlainText() == CONSIGNE


def test_vider_la_zone_ne_detruit_pas_la_consigne(app, config, saved):
    dialog = SettingsDialog(config)
    dialog.prompt_edit.setPlainText("")          # Ctrl+A puis Suppr
    dialog._save()
    assert saved["reformat_prompt_overrides"]["message"] == CONSIGNE


def test_remettre_la_consigne_integree_retire_la_surcharge(app, config, saved):
    dialog = SettingsDialog(config)
    dialog.prompt_edit.setPlainText(presets.BUILTIN_PRESETS["message"]["prompt"])
    dialog._save()
    assert "message" not in saved["reformat_prompt_overrides"]


def test_une_consigne_modifiee_est_enregistree(app, config, saved):
    dialog = SettingsDialog(config)
    dialog.prompt_edit.setPlainText("Version révisée")
    dialog._save()
    assert saved["reformat_prompt_overrides"]["message"] == "Version révisée"


def test_creer_un_format_conserve_la_consigne_en_cours(app, config, saved, monkeypatch):
    monkeypatch.setattr(QInputDialog, "getText",
                        staticmethod(lambda *a, **k: ("Nouveau", True)))
    dialog = SettingsDialog(config)
    dialog.prompt_edit.setPlainText("Édition en cours à ne pas perdre")
    dialog._add_custom_mode()
    dialog._save()
    assert saved["reformat_prompt_overrides"]["message"] == "Édition en cours à ne pas perdre"
    assert any(c["name"] == "Nouveau" for c in saved["reformat_custom_modes"])


def test_changer_de_format_conserve_la_consigne_en_cours(app, config, saved):
    dialog = SettingsDialog(config)
    dialog.prompt_edit.setPlainText("Édition à conserver")
    modes = [m for _, m in presets.list_modes(config)]
    dialog._select(dialog.mode_combo, "github")
    dialog._save()
    assert saved["reformat_prompt_overrides"]["message"] == "Édition à conserver"
    assert saved["reformat_mode"] == "github"
    assert "github" in modes


def test_dupliquer_propose_un_nom_libre(app, config, monkeypatch):
    demandes = []
    monkeypatch.setattr(QInputDialog, "getText",
                        staticmethod(lambda *a, **k: (demandes.append(k.get("text")), ("", False))[1]))
    dialog = SettingsDialog(config)
    dialog._duplicate_mode()
    assert demandes and demandes[0].endswith("(copie)")


def test_un_nom_deja_pris_est_signale(app, config, monkeypatch):
    config["reformat_custom_modes"] = [{"name": "Tweet", "prompt": "x", "backend": "ollama"}]
    monkeypatch.setattr(QInputDialog, "getText", staticmethod(lambda *a, **k: ("Tweet", True)))
    dialog = SettingsDialog(config)
    dialog._add_custom_mode()
    assert "existe déjà" in dialog.models_tab.status.text()
    assert len(dialog.config["reformat_custom_modes"]) == 1


def test_backend_vide_signifie_suivre_le_defaut(app, config, saved, monkeypatch):
    config["reformat_custom_modes"] = [{"name": "Tweet", "prompt": "x", "backend": "claude"}]
    dialog = SettingsDialog(config)
    dialog._select(dialog.mode_combo, "custom:Tweet")
    dialog._select(dialog.mode_backend_combo, "")      # « Backend par défaut »
    dialog._save()
    custom = saved["reformat_custom_modes"][0]
    assert custom["backend"] == ""
    assert presets.mode_backend(saved, "custom:Tweet") == saved["reformat_backend"]


def test_la_recommandation_est_poussee_dans_les_widgets(app, config, saved):
    dialog = SettingsDialog(config)
    dialog._apply_recommendation({"model": "large-v3-turbo", "compute_type": "int8",
                                  "ollama_model": "qwen3:1.7b", "ollama_num_ctx": 2048,
                                  "whisper_idle_unload_min": 5})
    assert dialog.model_combo.currentData() == "large-v3-turbo"
    assert dialog.compute_combo.currentData() == "int8"
    assert dialog.unload_spin.value() == 5
    dialog._save()
    assert saved["model"] == "large-v3-turbo"
    assert saved["compute_type"] == "int8"
    assert saved["ollama_num_ctx"] == 2048


def test_un_choix_explicite_apres_la_recommandation_gagne(app, config, saved):
    dialog = SettingsDialog(config)
    dialog._apply_recommendation({"model": "large-v3-turbo", "compute_type": "int8"})
    dialog._select(dialog.compute_combo, "float16")     # l'utilisateur change d'avis
    dialog._save()
    assert saved["compute_type"] == "float16"


def test_le_modele_local_n_est_jamais_enregistre_vide(app, config, saved):
    dialog = SettingsDialog(config)
    dialog.ollama_model_combo.clear()                   # Ollama éteint
    dialog._save()
    assert saved["ollama_model"] == config["ollama_model"]


def test_les_corrections_font_l_aller_retour(app, config, saved):
    dialog = SettingsDialog(config)
    dialog.corrections_edit.setPlainText("machin => Machin\n# desactive => rien")
    dialog._save()
    regles = saved["corrections"]
    assert regles[0] == {"from": "machin", "to": "Machin", "regex": False, "enabled": True}
    assert regles[1]["enabled"] is False
