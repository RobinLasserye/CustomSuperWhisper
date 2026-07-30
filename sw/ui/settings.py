"""Fenêtre de réglages, organisée en onglets."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout,
    QInputDialog, QLabel, QLineEdit, QPlainTextEdit, QPushButton, QScrollArea, QSpinBox,
    QTabWidget, QVBoxLayout, QWidget,
)

from .. import artifacts, backends, presets, transcriber, vocabulary
from ..config import save_config
from ..hardware import get_audio_inputs, get_gpu_list
from . import style
from .models_tab import ModelsTab


def _hint(text):
    label = QLabel(text)
    label.setProperty("hint", "true")
    label.setWordWrap(True)
    return label


def _muted(button):
    button.setProperty("muted", "true")
    return button


def _scrollable(widget):
    area = QScrollArea()
    area.setWidgetResizable(True)
    area.setWidget(widget)
    return area


class SettingsDialog(QDialog):
    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.config = config
        self.setWindowTitle("SuperWhisper Custom")
        self.setMinimumSize(720, 640)
        self.setStyleSheet(style.DIALOG)

        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(20, 16, 20, 16)

        header = QLabel("SuperWhisper Custom")
        header.setStyleSheet(f"color: {style.TEXT}; font-size: 20px; font-weight: bold;")
        header.setAlignment(Qt.AlignCenter)
        layout.addWidget(header)

        subtitle = QLabel("Ctrl + Alt + Espace : dicter · "
                          "Ctrl + Alt + Maj + Espace : dicter puis choisir le format")
        subtitle.setProperty("hint", "true")
        subtitle.setAlignment(Qt.AlignCenter)
        layout.addWidget(subtitle)

        self.tabs = QTabWidget()
        self.tabs.addTab(_scrollable(self._build_transcription_tab()), "Transcription")
        self.tabs.addTab(_scrollable(self._build_vocabulary_tab()), "Vocabulaire")
        self.tabs.addTab(_scrollable(self._build_cleanup_tab()), "Nettoyage")
        self.tabs.addTab(_scrollable(self._build_reformat_tab()), "Reformulation")
        self.models_tab = ModelsTab(config)
        self.tabs.addTab(_scrollable(self.models_tab), "Modèles")
        self.tabs.addTab(_scrollable(self._build_general_tab()), "Général")
        layout.addWidget(self.tabs, 1)

        save = QPushButton("Sauvegarder")
        save.clicked.connect(self._save)
        layout.addWidget(save)

        self._on_mode_changed()

    # ─── Onglet Transcription ────────────────────────────────────────────────

    def _build_transcription_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(12)

        box = QGroupBox("Modèle")
        form = QFormLayout(box)

        self.model_combo = QComboBox()
        for name in transcriber.AVAILABLE_MODELS:
            self.model_combo.addItem(name, name)
        self._select(self.model_combo, self.config.get("model"))
        form.addRow("Modèle :", self.model_combo)

        self.compute_combo = QComboBox()
        for label, value in transcriber.COMPUTE_TYPES:
            self.compute_combo.addItem(label, value)
        self._select(self.compute_combo, self.config.get("compute_type"))
        form.addRow("Précision :", self.compute_combo)

        self.lang_combo = QComboBox()
        for label, code in transcriber.WHISPER_LANGUAGES:
            self.lang_combo.addItem(label, code)
        self._select(self.lang_combo, self.config.get("language"))
        form.addRow("Langue dictée :", self.lang_combo)
        layout.addWidget(box)
        layout.addWidget(_hint("L'onglet « Modèles » compare vitesse, VRAM et qualité mesurées, "
                               "et permet de télécharger un modèle plus léger."))

        hardware = QGroupBox("Matériel")
        hardware_form = QFormLayout(hardware)

        self.gpu_combo = QComboBox()
        for index, name, vram in get_gpu_list():
            suffix = f" — {vram / 1024:.0f} Go" if vram else ""
            self.gpu_combo.addItem(f"GPU {index} : {name}{suffix}", index)
        self._select(self.gpu_combo, self.config.get("gpu_index", "0"))
        hardware_form.addRow("GPU :", self.gpu_combo)

        self.audio_combo = QComboBox()
        for identifier, name in get_audio_inputs():
            self.audio_combo.addItem(name, identifier)
        self._select(self.audio_combo, self.config.get("audio_device", "default"))
        hardware_form.addRow("Microphone :", self.audio_combo)

        self.unload_spin = QSpinBox()
        self.unload_spin.setRange(0, 240)
        self.unload_spin.setSuffix(" min")
        self.unload_spin.setSpecialValueText("jamais")
        self.unload_spin.setValue(int(self.config.get("whisper_idle_unload_min", 0)))
        hardware_form.addRow("Décharger Whisper après :", self.unload_spin)
        layout.addWidget(hardware)
        layout.addWidget(_hint("Garder « default » comme microphone : les index numériques "
                               "changent au redémarrage et tombent sur un périphérique qui refuse "
                               "le 16 kHz. Le déchargement libère la VRAM entre deux dictées, au "
                               "prix d'un rechargement de 1 à 2 s."))
        layout.addStretch()
        return page

    # ─── Onglet Vocabulaire ──────────────────────────────────────────────────

    def _build_vocabulary_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(12)

        bias_box = QGroupBox("Biais de reconnaissance")
        bias_layout = QVBoxLayout(bias_box)
        self.bias_check = QCheckBox("Injecter le vocabulaire dans le décodage de Whisper")
        self.bias_check.setChecked(self.config.get("vocab_biasing", True))
        bias_layout.addWidget(self.bias_check)
        bias_layout.addWidget(_hint("Mesuré : « Claude » passe de 6 clips sur 9 à 9 sur 9, et le "
                                    "taux d'erreur global baisse de 0,37 à 0,25. Un terme par "
                                    "ligne."))
        self.vocab_edit = QPlainTextEdit("\n".join(self.config.get("vocabulary", [])))
        self.vocab_edit.setMinimumHeight(120)
        bias_layout.addWidget(self.vocab_edit)
        layout.addWidget(bias_box)

        corrections_box = QGroupBox("Corrections après transcription")
        corrections_layout = QVBoxLayout(corrections_box)
        self.corrections_check = QCheckBox("Appliquer le dictionnaire de corrections")
        self.corrections_check.setChecked(self.config.get("corrections_enabled", True))
        corrections_layout.addWidget(self.corrections_check)
        corrections_layout.addWidget(_hint(
            "Une règle par ligne : <code>motif =&gt; remplacement</code>. Préfixe "
            "<code>re:</code> pour une expression régulière, <code>#</code> pour désactiver une "
            "règle. L'ordre compte : les règles à plusieurs mots doivent précéder les règles "
            "isolées."))
        self.corrections_edit = QPlainTextEdit(
            vocabulary.format_corrections_text(self.config.get("corrections", [])))
        self.corrections_edit.setMinimumHeight(150)
        self.corrections_edit.textChanged.connect(self._update_test)
        corrections_layout.addWidget(self.corrections_edit)
        layout.addWidget(corrections_box)

        cloud_box = QGroupBox("Règle « cloud » → « Claude »")
        cloud_layout = QVBoxLayout(cloud_box)
        self.cloud_check = QCheckBox("Remplacer « cloud » par « Claude »")
        self.cloud_check.setChecked(self.config.get("cloud_rule_enabled", True))
        self.cloud_check.stateChanged.connect(self._update_test)
        cloud_layout.addWidget(self.cloud_check)
        cloud_layout.addWidget(_hint("Sauf dans les expressions ci-dessous, où « cloud » garde son "
                                     "sens d'hébergement. Une expression par ligne."))
        self.cloud_edit = QPlainTextEdit("\n".join(self.config.get("cloud_exceptions", [])))
        self.cloud_edit.setMaximumHeight(110)
        self.cloud_edit.textChanged.connect(self._update_test)
        cloud_layout.addWidget(self.cloud_edit)
        layout.addWidget(cloud_box)

        test_box = QGroupBox("Essayer")
        test_layout = QVBoxLayout(test_box)
        self.test_input = QLineEdit("Je lance cloud code puis je déploie dans le cloud AWS")
        self.test_input.textChanged.connect(self._update_test)
        test_layout.addWidget(self.test_input)
        self.test_output = QLabel()
        self.test_output.setWordWrap(True)
        self.test_output.setStyleSheet(f"color: {style.GREEN};")
        test_layout.addWidget(self.test_output)
        layout.addWidget(test_box)
        self._update_test()

        layout.addStretch()
        return page

    def _update_test(self):
        probe = {
            "corrections_enabled": self.corrections_check.isChecked(),
            "corrections": vocabulary.parse_corrections_text(
                self.corrections_edit.toPlainText()),
            "cloud_rule_enabled": self.cloud_check.isChecked(),
            "cloud_exceptions": vocabulary.parse_list_text(self.cloud_edit.toPlainText()),
        }
        try:
            result = vocabulary.correct(self.test_input.text(), probe)
        except Exception as exc:
            self.test_output.setStyleSheet(f"color: {style.RED};")
            self.test_output.setText(f"Règle invalide : {exc}")
            return
        self.test_output.setStyleSheet(f"color: {style.GREEN};")
        self.test_output.setText("→ " + result)

    # ─── Onglet Nettoyage ────────────────────────────────────────────────────

    def _build_cleanup_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(12)

        box = QGroupBox("Hallucinations de Whisper")
        box_layout = QVBoxLayout(box)
        self.artifact_check = QCheckBox("Retirer les artefacts de sous-titrage")
        self.artifact_check.setChecked(self.config.get("artifact_filter", True))
        box_layout.addWidget(self.artifact_check)
        box_layout.addWidget(_hint(
            "Whisper a été entraîné sur des sous-titres : sur du silence ou une fin de phrase "
            "coupée, il produit « Sous-titrage ST' 501 », « Merci d'avoir regardé cette vidéo » ou "
            "un crédit Amara.org qui n'ont jamais été prononcés. Ces motifs sont retirés du "
            "segment sans toucher au reste. Comparaison insensible à la casse, aux accents et à "
            "la ponctuation."))
        self.artifact_edit = QPlainTextEdit("\n".join(self.config.get("artifact_patterns", [])))
        self.artifact_edit.setMinimumHeight(130)
        box_layout.addWidget(self.artifact_edit)
        layout.addWidget(box)

        ambiguous_box = QGroupBox("Formules ambiguës")
        ambiguous_layout = QVBoxLayout(ambiguous_box)
        self.ambiguous_check = QCheckBox(
            "Retirer « merci », « au revoir »… seulement si Whisper doute du segment")
        self.ambiguous_check.setChecked(self.config.get("artifact_ambiguous_enabled", True))
        ambiguous_layout.addWidget(self.ambiguous_check)
        ambiguous_layout.addWidget(_hint(
            "Ces formules peuvent être réellement dictées : elles ne sont supprimées que si le "
            "segment entier s'y réduit ET que les métriques de Whisper le trahissent."))
        self.ambiguous_edit = QPlainTextEdit("\n".join(self.config.get("artifact_ambiguous", [])))
        self.ambiguous_edit.setMaximumHeight(90)
        ambiguous_layout.addWidget(self.ambiguous_edit)

        thresholds = QFormLayout()
        self.no_speech_spin = QDoubleSpinBox()
        self.no_speech_spin.setRange(0.0, 1.0)
        self.no_speech_spin.setSingleStep(0.05)
        self.no_speech_spin.setValue(float(self.config.get(
            "artifact_no_speech_threshold", artifacts.DEFAULT_NO_SPEECH_THRESHOLD)))
        thresholds.addRow("Seuil « pas de parole » :", self.no_speech_spin)

        self.logprob_spin = QDoubleSpinBox()
        self.logprob_spin.setRange(-5.0, 0.0)
        self.logprob_spin.setSingleStep(0.1)
        self.logprob_spin.setValue(float(self.config.get(
            "artifact_logprob_threshold", artifacts.DEFAULT_LOGPROB_THRESHOLD)))
        thresholds.addRow("Seuil de vraisemblance :", self.logprob_spin)
        ambiguous_layout.addLayout(thresholds)
        layout.addWidget(ambiguous_box)

        self.collapse_check = QCheckBox(
            "Effondrer les phrases répétées en boucle (bug classique de Whisper)")
        self.collapse_check.setChecked(self.config.get("collapse_repetitions", True))
        layout.addWidget(self.collapse_check)

        layout.addStretch()
        return page

    # ─── Onglet Reformulation ────────────────────────────────────────────────

    def _build_reformat_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(12)

        engine = QGroupBox("Moteur")
        engine_form = QFormLayout(engine)

        self.backend_combo = QComboBox()
        self.backend_combo.addItem(backends.OllamaBackend.label, "ollama")
        self.backend_combo.addItem(backends.ClaudeCliBackend.label, "claude")
        self._select(self.backend_combo, self.config.get("reformat_backend", "ollama"))
        engine_form.addRow("Backend :", self.backend_combo)

        self.host_edit = QLineEdit(self.config.get("ollama_host", "http://127.0.0.1:11434"))
        engine_form.addRow("Hôte Ollama :", self.host_edit)

        model_row = QHBoxLayout()
        self.ollama_model_combo = QComboBox()
        self.ollama_model_combo.setEditable(True)
        model_row.addWidget(self.ollama_model_combo, 1)
        refresh = _muted(QPushButton("Rafraîchir"))
        refresh.clicked.connect(self._refresh_ollama_models)
        model_row.addWidget(refresh)
        engine_form.addRow("Modèle local :", model_row)

        self.ollama_status = _hint("")
        engine_form.addRow("", self.ollama_status)
        self._refresh_ollama_models()

        self.keep_alive_edit = QLineEdit(self.config.get("ollama_keep_alive", "30m"))
        engine_form.addRow("Garder chargé :", self.keep_alive_edit)

        self.timeout_spin = QSpinBox()
        self.timeout_spin.setRange(5, 600)
        self.timeout_spin.setSuffix(" s")
        self.timeout_spin.setValue(int(self.config.get("ollama_timeout_s", 60)))
        engine_form.addRow("Délai maximal :", self.timeout_spin)

        self.num_ctx_combo = QComboBox()
        for value in (2048, 4096, 8192, 16384, 32768):
            self.num_ctx_combo.addItem(str(value), value)
        self._select(self.num_ctx_combo, int(self.config.get("ollama_num_ctx", 8192)))
        engine_form.addRow("Contexte :", self.num_ctx_combo)
        layout.addWidget(engine)
        layout.addWidget(_hint("« Garder chargé » accepte 0 (décharge tout de suite), 30m, 1h ou "
                               "-1 (jamais décharger). Un contexte plus court économise de la "
                               "VRAM mais tronque les longues dictées : 8192 couvre environ "
                               "3 000 mots."))

        defaults = QGroupBox("Par défaut")
        defaults_form = QFormLayout(defaults)

        self.mode_combo = QComboBox()
        self._rebuild_mode_combo()
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        defaults_form.addRow("Format :", self.mode_combo)

        self.target_lang_combo = QComboBox()
        for label, code in presets.LANGUAGES:
            self.target_lang_combo.addItem(label, code)
        self._select(self.target_lang_combo, self.config.get("target_language", "none"))
        defaults_form.addRow("Langue de sortie :", self.target_lang_combo)
        layout.addWidget(defaults)
        layout.addWidget(_hint("La langue de sortie s'applique à n'importe quel format : mail "
                               "formel en anglais, ticket GitHub en japonais, etc. Elle est "
                               "modifiable à la volée dans le sélecteur."))

        prompt_box = QGroupBox("Consigne du format sélectionné")
        prompt_layout = QVBoxLayout(prompt_box)
        self.mode_backend_combo = QComboBox()
        self.mode_backend_combo.addItem("Backend par défaut", "")
        self.mode_backend_combo.addItem(backends.OllamaBackend.label, "ollama")
        self.mode_backend_combo.addItem(backends.ClaudeCliBackend.label, "claude")
        backend_row = QHBoxLayout()
        backend_row.addWidget(QLabel("Backend pour ce format :"))
        backend_row.addWidget(self.mode_backend_combo, 1)
        prompt_layout.addLayout(backend_row)

        self.prompt_edit = QPlainTextEdit()
        self.prompt_edit.setMinimumHeight(180)
        prompt_layout.addWidget(self.prompt_edit)

        buttons = QHBoxLayout()
        add = _muted(QPushButton("+ Nouveau format"))
        add.clicked.connect(self._add_custom_mode)
        buttons.addWidget(add)
        duplicate = _muted(QPushButton("Dupliquer"))
        duplicate.clicked.connect(self._duplicate_mode)
        buttons.addWidget(duplicate)
        self.reset_button = _muted(QPushButton("Réinitialiser"))
        self.reset_button.clicked.connect(self._reset_prompt)
        buttons.addWidget(self.reset_button)
        self.delete_button = _muted(QPushButton("Supprimer"))
        self.delete_button.clicked.connect(self._delete_custom_mode)
        buttons.addWidget(self.delete_button)
        buttons.addStretch()
        prompt_layout.addLayout(buttons)
        layout.addWidget(prompt_box)

        layout.addStretch()
        return page

    def _refresh_ollama_models(self):
        host = self.host_edit.text().strip() or "http://127.0.0.1:11434"
        backend = backends.OllamaBackend(host=host)
        current = self.ollama_model_combo.currentText() or self.config.get("ollama_model")
        models = backend.list_models()
        self.ollama_model_combo.clear()
        self.ollama_model_combo.addItems(models)
        if current:
            index = self.ollama_model_combo.findText(current)
            if index >= 0:
                self.ollama_model_combo.setCurrentIndex(index)
            else:
                self.ollama_model_combo.setEditText(current)
        if models:
            self.ollama_status.setText(f"{len(models)} modèles disponibles sur {host}")
        else:
            self.ollama_status.setText(
                f"Ollama ne répond pas sur {host} — la reformulation collera le texte brut")

    def _rebuild_mode_combo(self):
        self.mode_combo.blockSignals(True)
        self.mode_combo.clear()
        for label, mode_id in presets.list_modes(self.config):
            self.mode_combo.addItem(label, mode_id)
        self._select(self.mode_combo, self.config.get("reformat_mode", presets.DISABLED))
        self.mode_combo.blockSignals(False)

    def _on_mode_changed(self):
        self._store_current_prompt()
        mode = self.mode_combo.currentData()
        is_builtin = mode in presets.BUILTIN_PRESETS
        is_custom = bool(mode) and mode.startswith(presets.CUSTOM_PREFIX)

        self.prompt_edit.setEnabled(bool(mode) and mode != presets.DISABLED)
        self.reset_button.setVisible(is_builtin)
        self.delete_button.setVisible(is_custom)
        self.mode_backend_combo.setEnabled(bool(mode) and mode != presets.DISABLED)

        if not mode or mode == presets.DISABLED:
            self.prompt_edit.setPlainText(
                "Aucune reformulation : le texte transcrit est collé tel quel.")
            self._current_mode = mode
            return

        self.prompt_edit.setPlainText(presets.preset_prompt(self.config, mode) or "")
        self._select(self.mode_backend_combo,
                     self.config.get("reformat_mode_backends", {}).get(mode, "")
                     if is_builtin else self._custom_backend(mode))
        self._current_mode = mode

    def _custom_backend(self, mode):
        name = mode[len(presets.CUSTOM_PREFIX):]
        for custom in self.config.get("reformat_custom_modes", []):
            if custom.get("name") == name:
                return custom.get("backend", "") or ""
        return ""

    def _store_current_prompt(self):
        """Mémorise la consigne éditée avant de changer de format."""
        mode = getattr(self, "_current_mode", None)
        if not mode or mode == presets.DISABLED:
            return
        text = self.prompt_edit.toPlainText()
        backend = self.mode_backend_combo.currentData() or ""

        if mode in presets.BUILTIN_PRESETS:
            overrides = dict(self.config.get("reformat_prompt_overrides", {}))
            if text.strip() and text != presets.BUILTIN_PRESETS[mode]["prompt"]:
                overrides[mode] = text
            else:
                overrides.pop(mode, None)
            self.config["reformat_prompt_overrides"] = overrides
            mode_backends = dict(self.config.get("reformat_mode_backends", {}))
            if backend:
                mode_backends[mode] = backend
            else:
                mode_backends.pop(mode, None)
            self.config["reformat_mode_backends"] = mode_backends
        elif mode.startswith(presets.CUSTOM_PREFIX):
            name = mode[len(presets.CUSTOM_PREFIX):]
            for custom in self.config.get("reformat_custom_modes", []):
                if custom.get("name") == name:
                    custom["prompt"] = text
                    custom["backend"] = backend or "ollama"
                    break

    def _add_custom_mode(self, prompt=""):
        name, accepted = QInputDialog.getText(self, "Nouveau format", "Nom du format :",
                                              text="Mon format")
        if not accepted or not name.strip():
            return
        name = name.strip()
        customs = self.config.setdefault("reformat_custom_modes", [])
        if any(custom.get("name") == name for custom in customs):
            return
        customs.append({"name": name, "prompt": prompt or "", "backend": "ollama"})
        self._current_mode = None
        self._rebuild_mode_combo()
        self._select(self.mode_combo, presets.CUSTOM_PREFIX + name)
        self._on_mode_changed()

    def _duplicate_mode(self):
        self._store_current_prompt()
        self._add_custom_mode(self.prompt_edit.toPlainText())

    def _reset_prompt(self):
        mode = self.mode_combo.currentData()
        if mode not in presets.BUILTIN_PRESETS:
            return
        overrides = dict(self.config.get("reformat_prompt_overrides", {}))
        overrides.pop(mode, None)
        self.config["reformat_prompt_overrides"] = overrides
        self.prompt_edit.setPlainText(presets.BUILTIN_PRESETS[mode]["prompt"])

    def _delete_custom_mode(self):
        mode = self.mode_combo.currentData()
        if not mode or not mode.startswith(presets.CUSTOM_PREFIX):
            return
        name = mode[len(presets.CUSTOM_PREFIX):]
        self.config["reformat_custom_modes"] = [
            custom for custom in self.config.get("reformat_custom_modes", [])
            if custom.get("name") != name]
        if self.config.get("reformat_mode") == mode:
            self.config["reformat_mode"] = presets.DISABLED
        self._current_mode = None
        self._rebuild_mode_combo()
        self._on_mode_changed()

    # ─── Onglet Général ──────────────────────────────────────────────────────

    def _build_general_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(12)

        box = QGroupBox("Collage")
        box_layout = QVBoxLayout(box)
        self.auto_paste_check = QCheckBox("Coller automatiquement après la dictée (Ctrl+V simulé)")
        self.auto_paste_check.setChecked(self.config.get("auto_paste", True))
        box_layout.addWidget(self.auto_paste_check)

        self.auto_paste_picker_check = QCheckBox(
            "Coller aussi quand le sélecteur de format a été utilisé")
        self.auto_paste_picker_check.setChecked(
            self.config.get("auto_paste_after_picker", True))
        box_layout.addWidget(self.auto_paste_picker_check)
        box_layout.addWidget(_hint("Le sélecteur prend le focus le temps du choix. Si ton "
                                   "gestionnaire de fenêtres ne rend pas le focus à la bonne "
                                   "application, décoche cette case : le texte reste dans le "
                                   "presse-papier."))
        layout.addWidget(box)

        picker_box = QGroupBox("Sélecteur de format")
        picker_layout = QVBoxLayout(picker_box)
        self.picker_language_check = QCheckBox(
            "Afficher le choix de la langue de sortie dans le sélecteur")
        self.picker_language_check.setChecked(self.config.get("picker_shows_language", True))
        picker_layout.addWidget(self.picker_language_check)
        layout.addWidget(picker_box)

        layout.addStretch()
        return page

    # ─── Utilitaires ─────────────────────────────────────────────────────────

    @staticmethod
    def _select(combo, value):
        for index in range(combo.count()):
            if combo.itemData(index) == value:
                combo.setCurrentIndex(index)
                return True
        return False

    def _save(self):
        self._store_current_prompt()

        self.config["model"] = self.model_combo.currentData()
        self.config["compute_type"] = self.compute_combo.currentData()
        self.config["language"] = self.lang_combo.currentData()
        self.config["gpu_index"] = self.gpu_combo.currentData()
        self.config["audio_device"] = self.audio_combo.currentData()
        self.config["whisper_idle_unload_min"] = self.unload_spin.value()

        self.config["vocab_biasing"] = self.bias_check.isChecked()
        self.config["vocabulary"] = vocabulary.parse_list_text(self.vocab_edit.toPlainText())
        self.config["corrections_enabled"] = self.corrections_check.isChecked()
        self.config["corrections"] = vocabulary.parse_corrections_text(
            self.corrections_edit.toPlainText())
        self.config["cloud_rule_enabled"] = self.cloud_check.isChecked()
        self.config["cloud_exceptions"] = vocabulary.parse_list_text(
            self.cloud_edit.toPlainText())

        self.config["artifact_filter"] = self.artifact_check.isChecked()
        self.config["artifact_patterns"] = vocabulary.parse_list_text(
            self.artifact_edit.toPlainText())
        self.config["artifact_ambiguous_enabled"] = self.ambiguous_check.isChecked()
        self.config["artifact_ambiguous"] = vocabulary.parse_list_text(
            self.ambiguous_edit.toPlainText())
        self.config["artifact_no_speech_threshold"] = self.no_speech_spin.value()
        self.config["artifact_logprob_threshold"] = self.logprob_spin.value()
        self.config["collapse_repetitions"] = self.collapse_check.isChecked()

        self.config["reformat_backend"] = self.backend_combo.currentData()
        self.config["ollama_host"] = self.host_edit.text().strip()
        # Si Ollama est éteint la combo peut être vide : on garde le modèle précédent plutôt que
        # d'enregistrer une chaîne vide, qui casserait la reformulation sans message clair.
        chosen_model = self.ollama_model_combo.currentText().strip()
        if chosen_model:
            self.config["ollama_model"] = chosen_model
        self.config["ollama_keep_alive"] = self.keep_alive_edit.text().strip() or "30m"
        self.config["ollama_timeout_s"] = self.timeout_spin.value()
        self.config["ollama_num_ctx"] = self.num_ctx_combo.currentData()
        self.config["reformat_mode"] = self.mode_combo.currentData() or presets.DISABLED
        self.config["target_language"] = self.target_lang_combo.currentData()

        self.config["auto_paste"] = self.auto_paste_check.isChecked()
        self.config["auto_paste_after_picker"] = self.auto_paste_picker_check.isChecked()
        self.config["picker_shows_language"] = self.picker_language_check.isChecked()

        # Une configuration appliquée depuis l'onglet « Modèles » écrase les combos concernées
        pending = getattr(self.models_tab, "pending_apply", None)
        if pending:
            self.config.update(pending)

        save_config(self.config)
        self.accept()
