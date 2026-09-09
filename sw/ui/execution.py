"""Local, bounded, in-memory execution inspector. Never steals dictation focus."""
from datetime import datetime
import json

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView, QComboBox, QDialog, QHBoxLayout, QHeaderView, QLabel,
    QPlainTextEdit, QPushButton, QSplitter, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget, QMessageBox, QTabWidget,
)
from . import style


class ExecutionDialog(QDialog):
    def __init__(self, parent=None, store=None):
        super().__init__(parent)
        self.reports = []
        self.store = store
        self.limit = 200 if store else 10
        self.setWindowTitle("Exécutions locales — SuperWhisper")
        self.resize(1040, 820)
        self.setStyleSheet(style.DIALOG)
        layout = QVBoxLayout(self)
        self.hint = hint = QLabel(
            "Whisper → filtre d'hallucinations → corrections : étapes communes en amont.\n"
            "Ici : traitement du texte nettoyé. Un nœud transforme l'état ; une flèche choisit "
            "le prochain nœud.\nLes 10 dernières exécutions restent en mémoire uniquement. "
            "Fermer cette fenêtre conserve l'historique ; Effacer le supprime.")
        hint.setWordWrap(True)
        if store:
            hint.setText("Historique privé sur cet ordinateur. Textes, consignes, mesures et audio si activé. "
                         "Aucun envoi automatique. Les réglages de collecte sont dans Général. "
                         "Les données restent conservées jusqu’à suppression.")
        layout.addWidget(hint)
        row = QHBoxLayout()
        self.runs = QComboBox()
        self.runs.currentIndexChanged.connect(self._select_run)
        row.addWidget(self.runs, 1)
        clear = QPushButton("Effacer")
        clear.clicked.connect(self._confirm_clear)
        row.addWidget(clear)
        if store:
            stats = QPushButton("Bilan")
            stats.clicked.connect(self.show_statistics)
            row.addWidget(stats)
            more = QPushButton("Charger davantage")
            more.clicked.connect(self.load_more)
            row.addWidget(more)
            delete = QPushButton("Supprimer cette exécution")
            delete.clicked.connect(self.delete_selected)
            row.addWidget(delete)
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
        self.prompt.setMaximumHeight(90)
        layout.addWidget(self.prompt)
        self.measurements = QPlainTextEdit()
        self.measurements.setReadOnly(True)
        self.measurements.setMaximumHeight(100)
        layout.addWidget(QLabel("Mesures disponibles (absence = non mesuré)"))
        layout.addWidget(self.measurements)
        self.rating = QComboBox()
        for label, value in (("Non évalué", "unrated"), ("Correct", "good"), ("À corriger", "incorrect")):
            self.rating.addItem(label, value)
        review_row = QHBoxLayout()
        review_row.addWidget(QLabel("Qualité — évaluation manuelle :"))
        review_row.addWidget(self.rating)
        save = QPushButton("Enregistrer l’évaluation")
        save.clicked.connect(self.save_review)
        review_row.addWidget(save)
        listen = QPushButton("Écouter l’audio")
        listen.clicked.connect(self.play_audio)
        listen.setVisible(store is not None)
        review_row.addWidget(listen)
        stop = QPushButton("Arrêter l’écoute")
        stop.clicked.connect(self.stop_audio)
        stop.setVisible(store is not None)
        review_row.addWidget(stop)
        layout.addLayout(review_row)
        self.correction = QPlainTextEdit()
        self.correction.setPlaceholderText("Version corrigée / attendue (facultatif). Enregistrer avant de changer d’exécution.")
        self.correction.setMaximumHeight(65)
        references = QTabWidget()
        references.addTab(self.correction, "Sortie finale attendue")
        self.reference_transcript = QPlainTextEdit()
        self.reference_transcript.setPlaceholderText("Transcription exacte de l’audio, avant reformulation (facultatif).")
        self.reference_transcript.setMaximumHeight(65)
        references.addTab(self.reference_transcript, "Transcription de référence")
        layout.addWidget(references)
        if store:
            self.reload_history()

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
        del self.reports[self.limit:]
        self.runs.insertItem(0, f"{datetime.fromisoformat(report.created_at).astimezone():%d/%m %H:%M:%S} · {report.pipeline} · {report.mode}")
        while self.runs.count() > self.limit:
            self.runs.removeItem(self.limit)
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
        self.rating.setCurrentIndex(max(0, self.rating.findData(report.review.get('rating', 'unrated'))))
        self.correction.setPlainText(report.review.get('correction', ''))
        self.reference_transcript.setPlainText(report.review.get('reference_transcript', ''))
        self.measurements.setPlainText(json.dumps({'execution': report.metrics, 'settings': report.settings}, ensure_ascii=False, indent=2))
        total = sum(step.duration_ms for step in report.steps)
        self.summary.setText(
            f"{report.pipeline} · {report.backend} · {report.model} · langue : {report.language}\n"
            f"Temps cumulé des étapes : {total:.1f} ms · "
            f"{report.warning or 'Résultat prêt'}\n"
            + " → ".join(step.node for step in report.steps)
            + ("\nHistorique NON enregistré : stockage local indisponible."
               if report.metrics.get("history_saved") is False else "")
            + ("\nAudio non enregistré : encodeur indisponible (texte conservé)."
               if report.metrics.get("audio_storage", {}).get("saved") is False else ""))
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
        self.measurements.setPlainText(json.dumps({'step': step.metrics, 'execution': report.metrics,
                                                   'settings': report.settings}, ensure_ascii=False, indent=2))

    def clear_history(self):
        if self.store:
            try:
                self.store.clear()
            except Exception:
                self.summary.setText("Suppression impossible — historique conservé.")
                return
        self.stop_audio()
        self.reports.clear()
        self.runs.clear()
        self.steps.setRowCount(0)
        self.before.clear()
        self.after.clear()
        self.prompt.clear()
        self.measurements.clear()
        self.correction.clear()
        self.reference_transcript.clear()
        self.summary.setText("Historique effacé.")

    def _confirm_clear(self):
        if self.store and QMessageBox.question(self, "Effacer l’historique local",
                "Supprimer tous les textes, évaluations et enregistrements audio locaux ?") != QMessageBox.Yes:
            return
        self.clear_history()

    def reload_history(self):
        try:
            reports = self.store.recent(self.limit)
        except Exception:
            self.summary.setText("Historique local indisponible. Les dictées restent utilisables.")
            return
        self.reports.clear()
        self.runs.clear()
        for report in reversed(reports):
            self.add_execution(report)
        if not reports:
            self.steps.setRowCount(0)
            self.before.clear()
            self.after.clear()
            self.prompt.clear()
            self.correction.clear()
            self.reference_transcript.clear()
            self.measurements.clear()
            self.summary.setText("Aucune exécution enregistrée.")

    def show_statistics(self):
        try:
            stats = self.store.statistics()
            lines = [f"{stats['runs']} exécutions · {stats['runs_with_warnings']} avec avertissement",
                     f"Base locale : {stats['database_bytes'] / 1_000_000:.1f} Mo",
                     f"Audio compressé : {stats['compressed_audio_bytes'] / 1_000_000:.1f} Mo"]
            ratings = stats['manual_ratings']
            lines.append(f"Évaluation manuelle : {ratings.get('good', 0)} correctes, "
                         f"{ratings.get('incorrect', 0)} à corriger, {ratings.get('unrated', 0)} non évaluées")
            for key, label in (("transcription_total_ms", "Transcription"),
                               ("pipeline_duration_ms", "Pipeline texte"),
                               ("stop_to_result_ms", "Arrêt micro → résultat (attente sélecteur incluse)")):
                if key in stats:
                    item = stats[key]
                    lines.append(f"{label} : médiane {item['median']:.0f} ms, P95 {item['p95']:.0f} ms "
                                 f"({item['measured_runs']} mesures)")
            QMessageBox.information(self, "Bilan local — toutes les exécutions", "\n".join(lines))
        except Exception:
            self.summary.setText("Bilan indisponible.")

    def load_more(self):
        self.limit += 200
        self.reload_history()

    def save_review(self):
        index = self.runs.currentIndex()
        if index < 0:
            return
        report = self.reports[index]
        rating, correction = self.rating.currentData(), self.correction.toPlainText()
        if self.store:
            try:
                if not self.store.review(report.run_id, rating, correction, self.reference_transcript.toPlainText()):
                    self.summary.setText("Exécution non enregistrée sur disque — collecte désactivée ou erreur d’écriture.")
                    return
            except Exception:
                self.summary.setText("Évaluation non enregistrée : stockage indisponible.")
                return
        report.review = {'rating': rating, 'correction': correction,
                         'reference_transcript': self.reference_transcript.toPlainText()}
        self.summary.setText("Évaluation enregistrée localement." if self.store else "Évaluation conservée en mémoire seulement.")

    def delete_selected(self):
        index = self.runs.currentIndex()
        if index < 0:
            return
        if QMessageBox.question(self, "Supprimer", "Supprimer cette exécution et son audio local ?") != QMessageBox.Yes:
            return
        try:
            self.store.delete(self.reports[index].run_id)
            self.stop_audio()
            self.reload_history()
        except Exception:
            self.summary.setText("Suppression impossible.")

    def play_audio(self):
        index = self.runs.currentIndex()
        if index < 0 or not self.store:
            return
        try:
            recording = self.store.audio(self.reports[index].run_id)
            if recording is None:
                self.summary.setText("Aucun audio conservé pour cette exécution.")
                return
            import sounddevice
            sounddevice.play(recording[0], recording[1])
        except Exception:
            self.summary.setText("Lecture audio indisponible.")

    @staticmethod
    def stop_audio():
        try:
            import sounddevice
            sounddevice.stop()
        except Exception:
            pass

    def closeEvent(self, event):
        self.stop_audio()
        super().closeEvent(event)
