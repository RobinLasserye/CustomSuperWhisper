"""Comportement clavier du sélecteur de format.

Tourne hors écran : aucun affichage réel, mais les événements clavier sont ceux de Qt.
"""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt                                    # noqa: E402
from PySide6.QtTest import QTest                                 # noqa: E402
from PySide6.QtWidgets import QApplication, QDialog              # noqa: E402

from sw import config as config_module, presets                  # noqa: E402
from sw.ui.picker import PresetPicker                             # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def config():
    return config_module.apply_defaults({})


def test_liste_tous_les_formats_plus_brut(app, config):
    picker = PresetPicker(config, "texte")
    attendu = len(presets.list_modes(config))
    assert picker.list.count() == attendu
    assert picker.list.item(0).data(Qt.UserRole) == presets.DISABLED


def test_le_format_par_defaut_est_preselectionne(app, config):
    config["reformat_mode"] = "github"
    picker = PresetPicker(config, "texte")
    assert picker.list.currentItem().data(Qt.UserRole) == "github"


def test_chiffre_choisit_et_valide(app, config):
    picker = PresetPicker(config, "texte")
    modes = presets.list_modes(config)
    QTest.keyClick(picker, Qt.Key_3)
    assert picker.result() == QDialog.Accepted
    assert picker.chosen_mode == modes[2][1]


def test_zero_ne_choisit_rien(app, config):
    picker = PresetPicker(config, "texte")
    QTest.keyClick(picker, Qt.Key_0)
    assert picker.result() != QDialog.Accepted


def test_echap_donne_le_texte_brut(app, config):
    config["reformat_mode"] = "mail"
    config["target_language"] = "en"
    picker = PresetPicker(config, "texte")
    QTest.keyClick(picker, Qt.Key_Escape)
    assert picker.result() == QDialog.Rejected
    assert picker.chosen_mode == presets.DISABLED
    assert picker.chosen_language == "none"


def test_entree_valide_la_selection_courante(app, config):
    config["reformat_mode"] = "notes"
    picker = PresetPicker(config, "texte")
    QTest.keyClick(picker, Qt.Key_Return)
    assert picker.result() == QDialog.Accepted
    assert picker.chosen_mode == "notes"


def test_fleches_changent_de_langue(app, config):
    picker = PresetPicker(config, "texte")
    depart = picker.language_combo.currentIndex()
    QTest.keyClick(picker, Qt.Key_Right)
    assert picker.language_combo.currentIndex() == depart + 1
    QTest.keyClick(picker, Qt.Key_Left)
    assert picker.language_combo.currentIndex() == depart


def test_la_langue_choisie_est_renvoyee(app, config):
    picker = PresetPicker(config, "texte")
    QTest.keyClick(picker, Qt.Key_Right)
    attendu = picker.language_combo.currentData()
    QTest.keyClick(picker, Qt.Key_Return)
    assert picker.chosen_language == attendu
    assert attendu != "none"


def test_langue_masquee_si_desactivee(app, config):
    config["picker_shows_language"] = False
    picker = PresetPicker(config, "texte")
    assert picker.language_combo is None
    QTest.keyClick(picker, Qt.Key_Right)        # ne doit pas planter
    QTest.keyClick(picker, Qt.Key_Return)
    assert picker.chosen_language == config["target_language"]


def test_apercu_tronque_les_longs_textes(app, config):
    long_texte = "mot " * 300
    picker = PresetPicker(config, long_texte)
    apercu = picker._preview(long_texte)
    assert apercu.endswith("…")
    assert len(apercu) <= 221


def test_apercu_gere_le_texte_vide(app, config):
    assert PresetPicker._preview("") == "(vide)"
    assert PresetPicker._preview(None) == "(vide)"


def test_formats_personnalises_selectionnables(app, config):
    config["reformat_custom_modes"] = [{"name": "Tweet", "prompt": "court", "backend": "ollama"}]
    picker = PresetPicker(config, "texte")
    dernier = picker.list.count() - 1
    assert picker.list.item(dernier).data(Qt.UserRole) == "custom:Tweet"
    picker.list.setCurrentRow(dernier)
    QTest.keyClick(picker, Qt.Key_Return)
    assert picker.chosen_mode == "custom:Tweet"
