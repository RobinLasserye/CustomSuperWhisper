"""Local, bounded, in-memory execution inspector. Never steals dictation focus."""
from datetime import datetime

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView, QComboBox, QDialog, QHBoxLayout, QHeaderView, QLabel,
    QPlainTextEdit, QPushButton, QSplitter, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget,
)
from . import style


class ExecutionDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.reports = []
        self.setWindowTitle("Exécutions locales — SuperWhisper")
        self.resize(980, 740)
        self.setStyleSheet(style.DIALOG)
        layout = QVBoxLayout(self)
        hint = QLabel(
            "Whisper → filtre d'hallucinations → corrections : étapes communes en amont.\n"
            "Ici : traitement du texte nettoyé. Un nœud transforme l'état ; une flèche choisit "
            "le prochain nœud.\nLes 10 dernières exécutions restent en mémoire uniquement. "
            "Fermer cette fenêtre conserve l'historique ; Effacer le supprime.")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        row = QHBoxLayout()
        self.runs = QComboBox()
        self.runs.currentIndexChanged.connect(self._select_run)
        row.addWidget(self.runs, 1)
        clear = QPushButton("Effacer")
        clear.clicked.connect(self.clear_history)
        row.addWidget(clear)
        layout.addLayout(row)
        self.summary = QLabel("Aucune exécution pour le moment. Lance une dictée.")
        self.summary.setTextFormat(Qt.PlainText)
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)
        self.steps = QTableWidget(0, 4)
        self.steps.setHorizontalHeaderLabels(["Nœud / étape", "Durée (ms)", "Tentative", "Décision / arête"])
        self.steps.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        self.steps.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.steps.setSelectionMode(QAbstractItemView.SingleSelection)
        self.steps.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.steps.itemSelectionChanged.connect(self._select_step)
        layout.addWidget(self.steps, 1)
        split = QSplitter()
        self.before = self._text_panel(split, "Avant le nœud")
        self.after = self._text_panel(split, "Après le nœud")
        layout.addWidget(split, 1)
        layout.addWidget(QLabel("Consigne envoyée au modèle (nœud format)"))
        self.prompt = QPlainTextEdit()
        self.prompt.setReadOnly(True)
        self.prompt.setPlaceholderText("Consigne du modèle (sélectionne un appel format)")
        self.prompt.setMaximumHeight(110)
        layout.addWidget(self.prompt)

    @staticmethod
    def _text_panel(split, title):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.addWidget(QLabel(title))
        text = QPlainTextEdit()
        text.setReadOnly(True)
        layout.addWidget(text)
        split.addWidget(panel)
        return text

    def add_execution(self, report):
        self.runs.blockSignals(True)
        self.reports.insert(0, report)
        del self.reports[10:]
        self.runs.insertItem(0, f"{datetime.now():%H:%M:%S} · {report.pipeline} · {report.mode}")
        while self.runs.count() > 10:
            self.runs.removeItem(10)
        self.runs.setCurrentIndex(0)
        self.runs.blockSignals(False)
        self._select_run(0)

    def _select_run(self, index):
        self.steps.setRowCount(0)
        self.before.clear()
        self.after.clear()
        self.prompt.clear()
        if index < 0 or index >= len(self.reports):
            return
        report = self.reports[index]
        total = sum(step.duration_ms for step in report.steps)
        self.summary.setText(
            f"{report.pipeline} · {report.backend} · {report.model} · langue : {report.language}\n"
            f"Temps cumulé des étapes : {total:.1f} ms · "
            f"{report.warning or 'Résultat prêt'}\n"
            + " → ".join(step.node for step in report.steps))
        self.steps.setRowCount(len(report.steps))
        for row, step in enumerate(report.steps):
            for col, value in enumerate((step.node, f"{step.duration_ms:.1f}",
                                         str(step.attempt), step.detail)):
                self.steps.setItem(row, col, QTableWidgetItem(value))
        self.steps.selectRow(0)

    def _select_step(self):
        index, row = self.runs.currentIndex(), self.steps.currentRow()
        if index < 0 or row < 0 or index >= len(self.reports):
            return
        report = self.reports[index]
        if row >= len(report.steps):
            return
        step = report.steps[row]
        self.before.setPlainText(step.before)
        self.after.setPlainText(step.after)
        self.prompt.setPlainText(step.prompt)

    def clear_history(self):
        self.reports.clear()
        self.runs.clear()
        self.steps.setRowCount(0)
        self.before.clear()
        self.after.clear()
        self.prompt.clear()
        self.summary.setText("Historique effacé.")
