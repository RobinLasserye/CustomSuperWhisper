"""Onglet « Modèles » : rapport de performance mesuré, recommandation selon la VRAM, et
téléchargement en un clic.

Les chiffres affichés viennent de `sw.models_catalog`, lui-même alimenté par les benchmarks du
dépôt. L'objectif est qu'on puisse choisir une configuration adaptée à une petite carte sans
deviner : VRAM réelle, vitesse, et qualité mesurée côte à côte.
"""

import threading

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView, QGroupBox, QHBoxLayout, QHeaderView, QLabel, QProgressBar,
    QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from .. import models_catalog
from ..hardware import gpu_vram_mib
from . import style


class DownloadSignals(QObject):
    progress = Signal(str, int)          # message, pourcentage (-1 = indéterminé)
    finished = Signal(bool, str)         # succès, message


class ModelsTab(QWidget):
    """Onglet autonome : il lit la config pour connaître le GPU et le host Ollama, et écrit
    `pending_apply` quand l'utilisateur veut appliquer une recommandation."""

    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.config = config
        self.pending_apply = None
        self.signals = DownloadSignals()
        self.signals.progress.connect(self._on_progress)
        self.signals.finished.connect(self._on_finished)
        self._busy = False

        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        layout.addWidget(self._build_recommendation())
        layout.addWidget(self._build_whisper_table())
        layout.addWidget(self._build_llm_table())

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        self.status = QLabel("")
        self.status.setProperty("hint", "true")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        self.refresh_states()

    # ─── Recommandation ──────────────────────────────────────────────────────

    def _build_recommendation(self):
        box = QGroupBox("Ta carte graphique")
        layout = QVBoxLayout(box)
        layout.setSpacing(8)

        vram = gpu_vram_mib(self.config.get("gpu_index", "0"))
        tier = models_catalog.recommend(vram or 0)
        whisper_model, whisper_compute = tier["whisper"]
        llm = tier["llm"]
        total = models_catalog.total_vram_mib(whisper_model, whisper_compute, llm,
                                             tier["num_ctx"])

        if vram:
            headline = f"GPU {self.config.get('gpu_index', '0')} — {vram / 1024:.1f} Go de VRAM"
        else:
            headline = "VRAM non détectée (nvidia-smi indisponible)"
        title = QLabel(headline)
        title.setStyleSheet(f"color: {style.TEXT}; font-weight: bold;")
        layout.addWidget(title)

        strategy = ("les deux modèles peuvent rester chargés en même temps"
                    if tier["strategy"] == "cohabitation"
                    else "Whisper doit être déchargé entre deux dictées")
        detail = QLabel(
            f"Configuration conseillée : <b>{whisper_model}</b> en {whisper_compute} + "
            f"<b>{llm}</b> (contexte {tier['num_ctx']}) — "
            f"{total / 1024:.1f} Go au total, {strategy}.<br>{tier['comment']}")
        detail.setWordWrap(True)
        detail.setTextFormat(Qt.RichText)
        layout.addWidget(detail)

        row = QHBoxLayout()
        apply_button = QPushButton("Appliquer cette configuration")
        apply_button.setProperty("muted", "true")
        apply_button.clicked.connect(lambda: self._apply_tier(tier))
        row.addWidget(apply_button)
        row.addStretch()
        layout.addLayout(row)
        return box

    def _apply_tier(self, tier):
        whisper_model, whisper_compute = tier["whisper"]
        self.pending_apply = {
            "model": whisper_model,
            "compute_type": whisper_compute,
            "ollama_model": tier["llm"],
            "ollama_num_ctx": tier["num_ctx"],
            "whisper_idle_unload_min": 5 if tier["strategy"] == "alternance" else 0,
        }
        self.status.setText(
            f"Configuration préparée : {whisper_model} / {whisper_compute} + {tier['llm']}. "
            "Elle sera enregistrée en cliquant sur « Sauvegarder ».")

    # ─── Tableau Whisper ─────────────────────────────────────────────────────

    def _build_whisper_table(self):
        box = QGroupBox("Transcription — mesures sur 30 clips français")
        layout = QVBoxLayout(box)

        profiles = sorted(models_catalog.WHISPER_PROFILES.items(),
                          key=lambda item: item[1]["vram_mib"])
        self.whisper_table = QTableWidget(len(profiles), 6)
        self.whisper_table.setHorizontalHeaderLabels(
            ["Modèle", "Précision", "VRAM", "Vitesse", "Qualité", "État"])
        self.whisper_table.verticalHeader().setVisible(False)
        self.whisper_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.whisper_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.whisper_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.whisper_table.setFixedHeight(min(len(profiles), 7) * 30 + 34)

        self._whisper_rows = []
        for row, ((model, compute), data) in enumerate(profiles):
            label = models_catalog.WHISPER_MODELS.get(model, {}).get("label", model)
            cells = [
                label,
                compute,
                f"{data['vram_mib'] / 1024:.1f} Go",
                models_catalog.speed_label(data["rtf"]),
                f"{models_catalog.quality_stars(data['wer_biased'])} "
                f"(WER {data['wer_biased']:.2f})",
                "",
            ]
            for column, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if column == 0 and not models_catalog.WHISPER_MODELS.get(
                        model, {}).get("multilingual", True):
                    item.setToolTip("Modèle anglais uniquement : WER catastrophique en français "
                                    "(0,91 mesuré)")
                self.whisper_table.setItem(row, column, item)
            self._whisper_rows.append((row, model))

        layout.addWidget(self.whisper_table)

        row_layout = QHBoxLayout()
        download = QPushButton("Télécharger le modèle sélectionné")
        download.setProperty("muted", "true")
        download.clicked.connect(self._download_selected_whisper)
        row_layout.addWidget(download)
        hint = QLabel("WER mesuré avec le biais vocabulaire activé, sur voix synthétique : "
                      "comparez les modèles entre eux, pas la valeur absolue.")
        hint.setProperty("hint", "true")
        hint.setWordWrap(True)
        row_layout.addWidget(hint, 1)
        layout.addLayout(row_layout)
        return box

    def _download_selected_whisper(self):
        row = self.whisper_table.currentRow()
        model = next((name for index, name in self._whisper_rows if index == row), None)
        if not model:
            self.status.setText("Sélectionne d'abord une ligne dans le tableau.")
            return
        if models_catalog.whisper_is_downloaded(model):
            self.status.setText(f"{model} est déjà présent dans le cache.")
            return
        self._start(f"Téléchargement de {model}…",
                    lambda: models_catalog.download_whisper(model),
                    f"{model} téléchargé")

    # ─── Tableau des modèles de reformulation ────────────────────────────────

    def _build_llm_table(self):
        box = QGroupBox("Reformulation — mesures et jugement à l'aveugle (note sur 5)")
        layout = QVBoxLayout(box)

        def rank(item):
            data = item[1]
            traps = data.get("traps") or (0, 9)
            jury = data.get("jury") or {}
            return -traps[0], -jury.get("fidelite", 0)

        models = sorted(models_catalog.LLM_MODELS.items(), key=rank)
        self.llm_table = QTableWidget(len(models), 6)
        self.llm_table.setHorizontalHeaderLabels(
            ["Modèle", "VRAM", "Latence", "Pièges", "Fidélité", "État"])
        self.llm_table.verticalHeader().setVisible(False)
        self.llm_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.llm_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.llm_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.llm_table.setFixedHeight(len(models) * 30 + 34)

        self._llm_rows = []
        for row, (name, data) in enumerate(models):
            jury = data.get("jury") or {}
            traps = data.get("traps")
            cells = [
                name,
                f"{data['vram_mib'] / 1024:.1f} Go",
                f"{data['latency_s']:.2f} s",
                f"{traps[0]}/{traps[1]}" if traps else "—",
                f"{jury['fidelite']:.1f}/5" if jury else "non jugé",
                "",
            ]
            risky = (traps and traps[0] <= 2) or (jury and jury["fidelite"] < 2.5)
            for column, text in enumerate(cells):
                item = QTableWidgetItem(text)
                item.setToolTip(data.get("note", ""))
                if risky:
                    item.setForeground(Qt.GlobalColor.red)
                self.llm_table.setItem(row, column, item)
            self._llm_rows.append((row, name))

        layout.addWidget(self.llm_table)

        row_layout = QHBoxLayout()
        pull = QPushButton("Télécharger le modèle sélectionné")
        pull.setProperty("muted", "true")
        pull.clicked.connect(self._pull_selected_llm)
        row_layout.addWidget(pull)
        use = QPushButton("Utiliser pour la reformulation")
        use.setProperty("muted", "true")
        use.clicked.connect(self._use_selected_llm)
        row_layout.addWidget(use)
        row_layout.addStretch()
        layout.addLayout(row_layout)

        note = QLabel("« Pièges » = test automatique de 9 dictées piégées "
                      "(<code>tools/eval_models.py</code>) : chiffres dictés à l'oral, noms "
                      "propres, négations, raccourcis. « Fidélité » = note de six jurés à "
                      "l'aveugle sur 12 dictées. Survolez une ligne pour le détail.")
        note.setProperty("hint", "true")
        note.setWordWrap(True)
        layout.addWidget(note)
        return box

    def _selected_llm(self):
        row = self.llm_table.currentRow()
        return next((name for index, name in self._llm_rows if index == row), None)

    def _pull_selected_llm(self):
        model = self._selected_llm()
        if not model:
            self.status.setText("Sélectionne d'abord une ligne dans le tableau.")
            return
        host = self.config.get("ollama_host", "http://127.0.0.1:11434")

        def task():
            def report(status, completed, total):
                if total:
                    self.signals.progress.emit(status, int(completed or 0) * 100 // int(total))
                else:
                    self.signals.progress.emit(status, -1)
            return models_catalog.pull_ollama_model(host, model, report)

        self._start(f"Téléchargement de {model} via Ollama…", task, f"{model} téléchargé")

    def _use_selected_llm(self):
        model = self._selected_llm()
        if not model:
            self.status.setText("Sélectionne d'abord une ligne dans le tableau.")
            return
        self.pending_apply = dict(self.pending_apply or {}, ollama_model=model)
        self.status.setText(f"{model} sera utilisé pour la reformulation après « Sauvegarder ».")

    # ─── Téléchargements ─────────────────────────────────────────────────────

    def _start(self, message, task, success_message):
        if self._busy:
            self.status.setText("Un téléchargement est déjà en cours.")
            return
        self._busy = True
        self.status.setText(message)
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)

        def run():
            try:
                task()
                self.signals.finished.emit(True, success_message)
            except Exception as exc:
                self.signals.finished.emit(False, f"Échec : {exc}")

        threading.Thread(target=run, daemon=True).start()

    def _on_progress(self, message, percent):
        if percent < 0:
            self.progress.setRange(0, 0)
        else:
            self.progress.setRange(0, 100)
            self.progress.setValue(percent)
        self.status.setText(message)

    def _on_finished(self, ok, message):
        self._busy = False
        self.progress.setVisible(False)
        self.status.setText(message)
        if ok:
            self.refresh_states()

    # ─── États ───────────────────────────────────────────────────────────────

    def refresh_states(self):
        for row, model in self._whisper_rows:
            present = models_catalog.whisper_is_downloaded(model)
            item = self.whisper_table.item(row, 5)
            if item:
                item.setText("téléchargé" if present else "absent")

        host = self.config.get("ollama_host", "http://127.0.0.1:11434")
        try:
            installed = set(models_catalog.ollama_installed_models(host))
        except Exception:
            installed = None
        for row, model in self._llm_rows:
            item = self.llm_table.item(row, 5)
            if not item:
                continue
            if installed is None:
                item.setText("Ollama éteint")
            else:
                item.setText("installé" if model in installed else "absent")
