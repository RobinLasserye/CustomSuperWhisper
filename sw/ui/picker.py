"""Sélecteur de format ouvert après la transcription (Ctrl+Alt+Maj+Espace).

Il prend le focus : lire des touches sans le prendre injecterait les chiffres dans l'application
cible. Le collage automatique est donc déclenché après sa fermeture, avec un délai un peu plus
long, le temps que le gestionnaire de fenêtres rende le focus.
"""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication, QComboBox, QDialog, QHBoxLayout, QLabel, QListWidget,
    QListWidgetItem, QVBoxLayout, QWidget,
)

from .. import presets
from . import style

PREVIEW_CHARS = 220


class PresetPicker(QDialog):
    """Retourne le mode et la langue choisis via `chosen_mode` / `chosen_language`."""

    def __init__(self, config, text, parent=None):
        super().__init__(parent)
        self.config = config
        self.chosen_mode = presets.DISABLED
        self.chosen_language = config.get("target_language", "none")

        self.setWindowFlags(Qt.Dialog | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setWindowTitle("SuperWhisper — Reformuler en…")
        self.setStyleSheet(style.PICKER)
        self.setMinimumWidth(430)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        card = QWidget()
        card.setObjectName("card")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(20, 16, 20, 14)
        layout.setSpacing(10)

        title = QLabel("Reformuler en…")
        title.setObjectName("title")
        layout.addWidget(title)

        preview = QLabel(self._preview(text))
        preview.setObjectName("preview")
        preview.setWordWrap(True)
        layout.addWidget(preview)

        self.list = QListWidget()
        modes = presets.list_modes(config)
        default_mode = config.get("reformat_mode", presets.DISABLED)
        for index, (label, mode_id) in enumerate(modes):
            shortcut = f"{index + 1}  " if index < 9 else "    "
            item = QListWidgetItem(f"{shortcut}{label}")
            item.setData(Qt.UserRole, mode_id)
            self.list.addItem(item)
            if mode_id == default_mode:
                self.list.setCurrentRow(index)
        if self.list.currentRow() < 0:
            self.list.setCurrentRow(0)
        self.list.setFixedHeight(min(len(modes), 10) * 34 + 8)
        self.list.itemActivated.connect(lambda _item: self._accept_current())
        self.list.itemClicked.connect(lambda _item: self._accept_current())
        layout.addWidget(self.list)

        self.language_combo = None
        if config.get("picker_shows_language", True):
            row = QHBoxLayout()
            row.setSpacing(8)
            language_label = QLabel("Langue de sortie")
            row.addWidget(language_label)
            self.language_combo = QComboBox()
            for label, code in presets.LANGUAGES:
                self.language_combo.addItem(label, code)
            current = config.get("target_language", "none")
            for position in range(self.language_combo.count()):
                if self.language_combo.itemData(position) == current:
                    self.language_combo.setCurrentIndex(position)
                    break
            self.language_combo.setFocusPolicy(Qt.NoFocus)   # les chiffres restent à la liste
            row.addWidget(self.language_combo, 1)
            layout.addLayout(row)

        hint = QLabel("1-9 choisir · ↑↓ naviguer · ←→ changer de langue · "
                      "Entrée appliquer · Échap texte brut")
        hint.setObjectName("hint")
        layout.addWidget(hint)

        outer.addWidget(card)
        self.list.setFocus()

    # — Aperçu —

    @staticmethod
    def _preview(text):
        flat = " ".join((text or "").split())
        if len(flat) > PREVIEW_CHARS:
            flat = flat[:PREVIEW_CHARS].rstrip() + "…"
        return flat or "(vide)"

    # — Interaction —

    def _accept_current(self):
        item = self.list.currentItem()
        self.chosen_mode = item.data(Qt.UserRole) if item else presets.DISABLED
        if self.language_combo is not None:
            self.chosen_language = self.language_combo.currentData()
        self.accept()

    def _shift_language(self, delta):
        if self.language_combo is None:
            return
        count = self.language_combo.count()
        self.language_combo.setCurrentIndex((self.language_combo.currentIndex() + delta) % count)

    def keyPressEvent(self, event):
        key = event.key()
        text = event.text()

        if key in (Qt.Key_Escape,):
            self.chosen_mode = presets.DISABLED
            self.chosen_language = "none"
            self.reject()
            return
        if key in (Qt.Key_Return, Qt.Key_Enter):
            self._accept_current()
            return
        if key == Qt.Key_Left:
            self._shift_language(-1)
            return
        if key == Qt.Key_Right:
            self._shift_language(1)
            return
        if text and text.isdigit() and text != "0":
            row = int(text) - 1
            if row < self.list.count():
                self.list.setCurrentRow(row)
                self._accept_current()
            return
        super().keyPressEvent(event)

    def showEvent(self, event):
        super().showEvent(event)
        screen = QApplication.primaryScreen().geometry()
        self.move((screen.width() - self.width()) // 2,
                  max(40, int(screen.height() * 0.22)))
        self.raise_()
        self.activateWindow()
        self.list.setFocus()
