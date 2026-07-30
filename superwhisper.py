#!/usr/bin/env python3
"""SuperWhisper Custom — transcription vocale locale, reformulation locale, traduction.

Raccourcis :
  Ctrl + Alt + Espace         dicter, puis appliquer le format par défaut
  Ctrl + Alt + Maj + Espace   dicter, puis choisir le format et la langue dans un sélecteur
"""

import os
import sys

# Doit tourner avant toute importation de faster_whisper : sous Linux, le loader dynamique ne
# relit pas LD_LIBRARY_PATH après le démarrage du processus, donc on se ré-exécute.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sw.runtime import ensure_cuda_libs                                   # noqa: E402

ensure_cuda_libs(os.path.abspath(__file__))

import threading                                                          # noqa: E402
import time                                                               # noqa: E402

from PySide6.QtCore import QObject, QSize, QTimer, Qt, Signal             # noqa: E402
from PySide6.QtGui import QAction, QActionGroup, QColor, QIcon, QPainter, QPixmap  # noqa: E402
from PySide6.QtWidgets import (                                           # noqa: E402
    QApplication, QDialog, QMenu, QSystemTrayIcon,
)

from sw import backends, config as config_module, instance, presets       # noqa: E402
from sw.audio import SAMPLE_RATE, AudioRecorder                           # noqa: E402
from sw.clipboard import auto_paste, clipboard_copy                       # noqa: E402
from sw.runtime import IS_WINDOWS, log                                    # noqa: E402
from sw.transcriber import Transcriber                                    # noqa: E402
from sw.ui.overlay import Overlay                                         # noqa: E402
from sw.ui.picker import PresetPicker                                     # noqa: E402
from sw.ui.settings import SettingsDialog                                 # noqa: E402

if not IS_WINDOWS:
    import signal

MIN_RECORDING_SECONDS = 0.3
PASTE_DELAY = 0.15
PASTE_DELAY_AFTER_PICKER = 0.35


class Signals(QObject):
    recording_started = Signal()
    transcription_started = Signal()
    reformulation_started = Signal(str)
    picker_requested = Signal(str)
    transcription_done = Signal(str, bool)
    warning = Signal(str)
    error = Signal(str)
    audio_level = Signal(object)


def create_icon(color, size=64):
    pixmap = QPixmap(QSize(size, size))
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.setBrush(QColor(color))
    painter.setPen(Qt.NoPen)
    painter.drawEllipse(8, 8, size - 16, size - 16)
    painter.end()
    return QIcon(pixmap)


class SuperWhisper(QObject):
    _open_settings_signal = Signal()

    def __init__(self):
        super().__init__()
        self.app = QApplication(sys.argv)
        self.app.setQuitOnLastWindowClosed(False)
        self.app.setApplicationName("SuperWhisper Custom")

        config_module.migrate_file()
        self.config = config_module.load_config()
        os.environ["CUDA_VISIBLE_DEVICES"] = self.config.get("gpu_index", "0")

        self.signals = Signals()
        self.recorder = AudioRecorder(self.signals, self.config.get("audio_device", "default"))
        self.transcriber = Transcriber()
        self.is_recording = False
        self.is_processing = False
        self.last_activity = time.monotonic()

        self.overlay = Overlay()
        self._build_tray()
        self._connect_signals()

        instance.write_pid()
        if IS_WINDOWS:
            self._setup_windows_ipc()
        else:
            self._setup_linux_ipc()

        if not os.path.exists(config_module.CONFIG_PATH):
            QTimer.singleShot(500, self._open_settings)

        self._idle_timer = QTimer(self)
        self._idle_timer.timeout.connect(self._check_idle)
        self._idle_timer.start(60_000)

        threading.Thread(target=self._preload, daemon=True).start()
        threading.Thread(target=self._hotkey_listener, daemon=True).start()

    # ─── Barre système ───────────────────────────────────────────────────────

    def _build_tray(self):
        self.tray = QSystemTrayIcon()
        self.icon_idle = create_icon("#a6e3a1")
        self.icon_recording = create_icon("#f38ba8")
        self.icon_working = create_icon("#89b4fa")
        self.tray.setIcon(self.icon_idle)
        self.tray.activated.connect(self._tray_activated)
        self.menu = QMenu()
        self._rebuild_tray_menu()
        self.tray.setContextMenu(self.menu)
        self.tray.show()

    def _rebuild_tray_menu(self):
        self.menu.clear()

        settings_action = QAction("Paramètres", self.menu)
        settings_action.triggered.connect(self._open_settings)
        self.menu.addAction(settings_action)
        self.menu.addSeparator()

        format_menu = self.menu.addMenu("Format par défaut")
        self._format_group = QActionGroup(self.menu)
        self._format_group.setExclusive(True)
        current_mode = self.config.get("reformat_mode", presets.DISABLED)
        for label, mode_id in presets.list_modes(self.config):
            action = QAction(label, self.menu, checkable=True)
            action.setChecked(mode_id == current_mode)
            action.triggered.connect(lambda _checked, m=mode_id: self._set_default_mode(m))
            self._format_group.addAction(action)
            format_menu.addAction(action)

        language_menu = self.menu.addMenu("Langue de sortie")
        self._language_group = QActionGroup(self.menu)
        self._language_group.setExclusive(True)
        current_language = self.config.get("target_language", "none")
        for label, code in presets.LANGUAGES:
            action = QAction(label, self.menu, checkable=True)
            action.setChecked(code == current_language)
            action.triggered.connect(lambda _checked, c=code: self._set_target_language(c))
            self._language_group.addAction(action)
            language_menu.addAction(action)

        self.menu.addSeparator()
        quit_action = QAction("Quitter", self.menu)
        quit_action.triggered.connect(self._quit)
        self.menu.addAction(quit_action)
        self._update_tooltip()

    def _update_tooltip(self):
        mode = presets.mode_label(self.config, self.config.get("reformat_mode"))
        language = self.config.get("target_language", "none")
        suffix = "" if language in (None, "none") else f" → {presets.language_label(language)}"
        self.tray.setToolTip(f"SuperWhisper — Ctrl+Alt+Espace · {mode}{suffix}")

    def _set_default_mode(self, mode):
        self.config["reformat_mode"] = mode
        config_module.save_config(self.config)
        self._update_tooltip()
        threading.Thread(target=self._warm_up_backend, daemon=True).start()

    def _set_target_language(self, code):
        self.config["target_language"] = code
        config_module.save_config(self.config)
        self._update_tooltip()

    def _tray_activated(self, reason):
        if reason == QSystemTrayIcon.ActivationReason.DoubleClick:
            self._open_settings()

    # ─── Signaux ─────────────────────────────────────────────────────────────

    def _connect_signals(self):
        self.signals.recording_started.connect(self._on_recording_started)
        self.signals.transcription_started.connect(self._on_transcription_started)
        self.signals.reformulation_started.connect(self._on_reformulation_started)
        self.signals.picker_requested.connect(self._on_picker_requested)
        self.signals.transcription_done.connect(self._on_transcription_done)
        self.signals.warning.connect(self._on_warning)
        self.signals.error.connect(self._on_error)
        self.signals.audio_level.connect(self.overlay.update_spectrum)
        self._open_settings_signal.connect(self._open_settings)

    def _setup_linux_ipc(self):
        self._sigusr_read, self._sigusr_write = os.pipe()
        signal.signal(signal.SIGUSR1, lambda *_: os.write(self._sigusr_write, b"\x00"))
        from PySide6.QtCore import QSocketNotifier
        self._notifier = QSocketNotifier(self._sigusr_read, QSocketNotifier.Type.Read)
        self._notifier.activated.connect(self._on_sigusr1)

    def _on_sigusr1(self):
        os.read(self._sigusr_read, 1)
        self._open_settings()

    def _setup_windows_ipc(self):
        import ctypes
        kernel32 = ctypes.windll.kernel32
        self._win_event = kernel32.CreateEventW(None, False, False, instance.WINDOWS_EVENT_NAME)
        self._win_timer = QTimer(self)
        self._win_timer.timeout.connect(self._check_windows_event)
        self._win_timer.start(500)

    def _check_windows_event(self):
        import ctypes
        if ctypes.windll.kernel32.WaitForSingleObject(self._win_event, 0) == 0:
            self._open_settings()

    # ─── Chargement et veille ────────────────────────────────────────────────

    def _preload(self):
        try:
            self.transcriber.load_model(self.config)
        except Exception as exc:
            self.signals.error.emit(f"Erreur modèle : {exc}")
        self._warm_up_backend()

    def _warm_up_backend(self):
        """Précharge le modèle de reformulation si un format est actif."""
        mode = self.config.get("reformat_mode", presets.DISABLED)
        translating = presets.is_translating(self.config.get("target_language"))
        if mode == presets.DISABLED and not translating:
            return
        if presets.mode_backend(self.config, mode) != "ollama":
            return
        backends.OllamaBackend.from_config(self.config).warm_up()

    def _check_idle(self):
        minutes = int(self.config.get("whisper_idle_unload_min", 0) or 0)
        if minutes <= 0 or self.is_recording or self.is_processing:
            return
        if not self.transcriber.is_loaded:
            return
        if time.monotonic() - self.last_activity >= minutes * 60:
            self.transcriber.unload()

    # ─── Réglages ────────────────────────────────────────────────────────────

    def _open_settings(self):
        previous = (self.config.get("model"), self.config.get("gpu_index"),
                    self.config.get("compute_type"))
        previous_llm = self.config.get("ollama_model")

        dialog = SettingsDialog(dict(self.config))
        if dialog.exec() != QDialog.Accepted:
            return

        self.config = config_module.load_config()
        os.environ["CUDA_VISIBLE_DEVICES"] = self.config.get("gpu_index", "0")
        self.recorder.device = self.config.get("audio_device", "default")
        self._rebuild_tray_menu()

        current = (self.config.get("model"), self.config.get("gpu_index"),
                   self.config.get("compute_type"))
        if current != previous:
            self.transcriber.unload()
            threading.Thread(target=self._preload, daemon=True).start()
        elif self.config.get("ollama_model") != previous_llm:
            threading.Thread(target=self._warm_up_backend, daemon=True).start()

    # ─── Raccourcis ──────────────────────────────────────────────────────────

    def _hotkey_listener(self):
        from pynput import keyboard
        pressed = {"ctrl": False, "alt": False, "shift": False}

        def on_press(key):
            if key in (keyboard.Key.ctrl_l, keyboard.Key.ctrl_r):
                pressed["ctrl"] = True
            elif key in (keyboard.Key.alt_l, keyboard.Key.alt_r, keyboard.Key.alt_gr):
                pressed["alt"] = True
            elif key in (keyboard.Key.shift, keyboard.Key.shift_l, keyboard.Key.shift_r):
                pressed["shift"] = True
            elif key == keyboard.Key.space and pressed["ctrl"] and pressed["alt"]:
                self._toggle(pick=pressed["shift"])

        def on_release(key):
            if key in (keyboard.Key.ctrl_l, keyboard.Key.ctrl_r):
                pressed["ctrl"] = False
            elif key in (keyboard.Key.alt_l, keyboard.Key.alt_r, keyboard.Key.alt_gr):
                pressed["alt"] = False
            elif key in (keyboard.Key.shift, keyboard.Key.shift_l, keyboard.Key.shift_r):
                pressed["shift"] = False

        log("écoute du raccourci (pynput) démarrée")
        while True:
            try:
                with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
                    listener.join()
            except Exception as exc:
                log(f"pynput a planté ({exc}), redémarrage dans 2 s")
                time.sleep(2)

    def _toggle(self, pick=False):
        self.last_activity = time.monotonic()
        if not self.is_recording:
            self.is_recording = True
            try:
                self.recorder.start()
            except Exception as exc:
                self.is_recording = False
                log(f"enregistrement impossible : {exc}")
                self.signals.error.emit("Micro indisponible")
                return
            self.signals.recording_started.emit()
            log(f"enregistrement démarré — micro={self.recorder.device}")
            return

        self.is_recording = False
        audio = self.recorder.stop()
        duration = len(audio) / SAMPLE_RATE
        log(f"enregistrement arrêté — {len(audio)} échantillons ({duration:.1f} s)")
        if duration < MIN_RECORDING_SECONDS:
            self.signals.error.emit("Trop court")
            return
        self.is_processing = True
        self.signals.transcription_started.emit()
        threading.Thread(target=self._transcribe, args=(audio, pick), daemon=True).start()

    # ─── Chaîne de traitement ────────────────────────────────────────────────

    def _transcribe(self, audio, pick):
        try:
            text, removed = self.transcriber.transcribe(audio, self.config)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            self.is_processing = False
            self.signals.error.emit(f"Erreur : {exc}")
            return

        if removed:
            log(f"artefacts retirés : {removed}")
        if not text:
            self.is_processing = False
            self.signals.error.emit("Aucun texte détecté")
            return
        log(f"transcription : {text[:100]}")

        if pick:
            self.signals.picker_requested.emit(text)
            return
        self._reformat_and_finish(text, self.config.get("reformat_mode", presets.DISABLED),
                                 self.config.get("target_language", "none"), False)

    def _on_picker_requested(self, text):
        """Ouvre le sélecteur — sur le fil graphique, obligatoirement."""
        self.overlay.hide_now()
        picker = PresetPicker(self.config, text)
        accepted = picker.exec() == QDialog.Accepted
        mode = picker.chosen_mode if accepted else presets.DISABLED
        language = picker.chosen_language if accepted else "none"
        threading.Thread(target=self._reformat_and_finish,
                         args=(text, mode, language, True), daemon=True).start()

    def _reformat_and_finish(self, text, mode, target_language, from_picker):
        system_prompt = presets.resolve(self.config, mode, target_language)
        if not system_prompt:
            self.signals.transcription_done.emit(text, from_picker)
            return

        effective_mode = presets.resolve_effective_mode(mode, target_language)
        effective_language = presets.effective_language(effective_mode, target_language)
        backend_name = presets.mode_backend(self.config, effective_mode)
        backend = backends.build_backend(self.config, backend_name)

        # Le texte brut est copié d'abord : si la reformulation échoue, rien n'est perdu.
        clipboard_copy(text)

        label = presets.mode_label(self.config, effective_mode)
        if backend_name == "ollama":
            label = f"{label} · {self.config.get('ollama_model', '')}"
        self.signals.reformulation_started.emit(label)

        try:
            result, warning = backends.reformat(backend, text, system_prompt, effective_language)
        except backends.ReformatError as exc:
            log(f"reformulation échouée : {exc}")
            self.is_processing = False
            self.signals.warning.emit(str(exc))
            return
        except Exception as exc:
            log(f"reformulation : erreur inattendue {exc}")
            self.is_processing = False
            self.signals.warning.emit("Reformulation impossible — texte brut collé")
            return

        self.signals.transcription_done.emit(result, from_picker)
        if warning:
            self.signals.warning.emit(warning)

    # ─── Retours visuels ─────────────────────────────────────────────────────

    def _on_recording_started(self):
        self.tray.setIcon(self.icon_recording)
        self.overlay.show_recording()

    def _on_transcription_started(self):
        self.tray.setIcon(self.icon_working)
        self.overlay.show_transcribing()

    def _on_reformulation_started(self, label):
        self.tray.setIcon(self.icon_working)
        self.overlay.show_reformulating(label)

    def _on_transcription_done(self, text, from_picker):
        self.is_processing = False
        self.last_activity = time.monotonic()
        self.tray.setIcon(self.icon_idle)
        clipboard_copy(text)
        self.overlay.show_done()
        if not self.config.get("auto_paste", True):
            return
        if from_picker and not self.config.get("auto_paste_after_picker", True):
            return
        delay = PASTE_DELAY_AFTER_PICKER if from_picker else PASTE_DELAY
        threading.Thread(target=auto_paste, args=(delay,), daemon=True).start()

    def _on_warning(self, text):
        self.is_processing = False
        self.tray.setIcon(self.icon_idle)
        self.overlay.show_warning(text)

    def _on_error(self, text):
        self.is_processing = False
        self.tray.setIcon(self.icon_idle)
        self.overlay.show_error(text)

    # ─── Cycle de vie ────────────────────────────────────────────────────────

    def _quit(self):
        instance.remove_pid()
        self.tray.hide()
        self.app.quit()

    def run(self):
        if not IS_WINDOWS:
            signal.signal(signal.SIGINT, signal.SIG_DFL)
        return self.app.exec()


if __name__ == "__main__":
    if instance.is_already_running():
        print("SuperWhisper déjà en cours — ouverture des paramètres.")
        sys.exit(0)
    sys.exit(SuperWhisper().run())
