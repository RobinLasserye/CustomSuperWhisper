#!/usr/bin/env python3
"""SuperWhisper Custom — Transcription vocale locale avec GUI."""

import os
import sys

# Ensure CUDA libraries are discoverable
_venv = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_site = os.path.join(_venv, ".venv", "lib", "python3.12", "site-packages")
_cuda_paths = [
    os.path.join(_site, "nvidia", "cublas", "lib"),
    os.path.join(_site, "nvidia", "cudnn", "lib"),
]
_existing = [p for p in _cuda_paths if os.path.isdir(p)]
if _existing:
    ld = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = ":".join(_existing) + (":" + ld if ld else "")
import json
import signal
import threading
import subprocess
import time
import numpy as np

from PySide6.QtWidgets import (
    QApplication, QWidget, QSystemTrayIcon, QMenu, QLabel, QVBoxLayout,
    QGraphicsOpacityEffect, QDialog, QComboBox, QPushButton,
    QFormLayout, QGroupBox, QPlainTextEdit, QHBoxLayout, QLineEdit,
    QInputDialog,
)
from PySide6.QtCore import Qt, QTimer, Signal, QObject, QSize
from PySide6.QtGui import QIcon, QPixmap, QPainter, QColor, QFont, QAction

SAMPLE_RATE = 16000
PIPEWIRE_RATE = 48000
NUM_BARS = 40
CONFIG_PATH = os.path.expanduser("~/.config/superwhisper-custom/config.json")
PID_FILE = os.path.expanduser("~/.config/superwhisper-custom/superwhisper.pid")

AVAILABLE_MODELS = [
    "large-v3", "large-v3-turbo", "distil-large-v3",
    "medium", "small", "base", "tiny",
]
AVAILABLE_LANGUAGES = [
    ("Français", "fr"), ("Anglais", "en"), ("Espagnol", "es"),
    ("Allemand", "de"), ("Italien", "it"), ("Portugais", "pt"),
    ("Japonais", "ja"), ("Chinois", "zh"), ("Auto-détection", None),
]

CLAUDE_BIN = os.path.expanduser("~/.local/bin/claude")
CLAUDE_BUILTIN_MODES = {
    "message": {
        "name": "Message",
        "prompt": (
            "Tu reçois une transcription vocale brute. Nettoie-la pour en faire un message "
            "prêt à envoyer (Facebook, SMS, Discord, etc.).\n\n"
            "CE QUE TU DOIS FAIRE :\n"
            "- Supprimer les hésitations (euh, bah, genre, en fait répété, du coup répété)\n"
            "- Corriger la grammaire et la ponctuation\n"
            "- Ajouter des retours à la ligne pour aérer quand le message est long\n"
            "- Supprimer les strictes répétitions (quand la même chose est dite deux fois de suite)\n"
            "- Rendre les phrases fluides et naturelles\n\n"
            "CE QUE TU NE DOIS PAS FAIRE :\n"
            "- NE PAS résumer, NE PAS raccourcir, NE PAS supprimer des informations ou des idées\n"
            "- NE PAS changer le sens ou le ton du message\n"
            "- NE PAS ajouter de contenu, de commentaire, de guillemets\n"
            "- NE PAS faire de résumé : CHAQUE idée et information du texte original doit être conservée\n\n"
            "Le message nettoyé doit faire à peu près la même longueur que l'original. "
            "Renvoie UNIQUEMENT le message nettoyé, rien d'autre."
        ),
    },
}

DEFAULT_CONFIG = {
    "model": "large-v3", "language": "fr", "compute_type": "float16",
    "gpu_index": "1", "audio_device": "default",
    "claude_mode": "disabled",
    "claude_custom_modes": [],
}


def find_existing_instances():
    """Find PIDs of other superwhisper.py processes (not ourselves)."""
    my_pid = os.getpid()
    pids = []
    try:
        out = subprocess.check_output(
            ["pgrep", "-f", "superwhisper.py"], text=True).strip()
        for line in out.split("\n"):
            pid = int(line.strip())
            if pid != my_pid:
                pids.append(pid)
    except (subprocess.CalledProcessError, ValueError):
        pass
    return pids


def is_already_running():
    """Check if another instance is running. If so, signal it and return True."""
    # First try PID file (faster, more reliable signal)
    if os.path.exists(PID_FILE):
        try:
            with open(PID_FILE) as f:
                pid = int(f.read().strip())
            os.kill(pid, 0)  # Check alive
            os.kill(pid, signal.SIGUSR1)  # Open settings
            return True
        except (OSError, ValueError):
            try:
                os.remove(PID_FILE)
            except OSError:
                pass

    # Fallback: scan for any running superwhisper process
    others = find_existing_instances()
    if others:
        for pid in others:
            try:
                os.kill(pid, signal.SIGUSR1)
                return True
            except OSError:
                continue
    return False


def write_pid():
    os.makedirs(os.path.dirname(PID_FILE), exist_ok=True)
    with open(PID_FILE, "w") as f:
        f.write(str(os.getpid()))


def remove_pid():
    try:
        os.remove(PID_FILE)
    except OSError:
        pass


def load_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH) as f:
            cfg = json.load(f)
            for k, v in DEFAULT_CONFIG.items():
                cfg.setdefault(k, v)
            return cfg
    return dict(DEFAULT_CONFIG)


def save_config(cfg):
    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)


def get_gpu_list():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,memory.total",
             "--format=csv,noheader"], text=True)
        gpus = []
        for line in out.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            gpus.append((parts[0], f"{parts[1]} ({parts[2]})"))
        return gpus
    except Exception:
        return [("0", "GPU 0")]


def get_audio_inputs():
    """Get PipeWire/PulseAudio sources via pactl."""
    devices = []
    try:
        out = subprocess.check_output(["pactl", "list", "sources", "short"], text=True)
        for line in out.strip().split("\n"):
            parts = line.split("\t")
            if len(parts) >= 2:
                name = parts[1]
                if "monitor" not in name:
                    devices.append((name, name.replace("alsa_input.", "").replace(".", " ")))
        # Add monitors at the end
        for line in out.strip().split("\n"):
            parts = line.split("\t")
            if len(parts) >= 2:
                name = parts[1]
                if "monitor" in name:
                    devices.append((name, f"[Monitor] {name.replace('alsa_output.', '').replace('.monitor', '')}"))
    except Exception:
        pass
    if not devices:
        devices.append(("default", "Par défaut"))
    return devices


class Signals(QObject):
    recording_started = Signal()
    recording_stopped = Signal()
    transcription_started = Signal()
    reformulation_started = Signal()
    transcription_done = Signal(str)
    error = Signal(str)
    audio_level = Signal(object)


class AudioRecorder:
    """Records audio using parec (PipeWire/PulseAudio) for reliable capture."""

    def __init__(self, signals, device=None):
        self.signals = signals
        self.device = device
        self.recording = False
        self.process = None
        self.frames = []
        self.thread = None

    def start(self):
        self.frames = []
        self.recording = True
        cmd = [
            "parec", "--format=float32le", "--rate=16000",
            "--channels=1", "--latency-msec=64",
        ]
        if self.device and self.device != "default":
            cmd += [f"--device={self.device}"]
        self.process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()

    def _read_loop(self):
        chunk_size = 1024 * 4  # 1024 float32 samples
        while self.recording and self.process:
            data = self.process.stdout.read(chunk_size)
            if not data:
                break
            samples = np.frombuffer(data, dtype=np.float32)
            self.frames.append(samples)
            fft = np.abs(np.fft.rfft(samples))[:NUM_BARS]
            mx = fft.max()
            if mx > 0:
                fft = fft / mx
            self.signals.audio_level.emit(fft)

    def stop(self):
        self.recording = False
        if self.process:
            self.process.terminate()
            self.process.wait()
            self.process = None
        if self.thread:
            self.thread.join(timeout=2)
        if self.frames:
            return np.concatenate(self.frames)
        return np.array([], dtype="float32")


class Transcriber:
    def __init__(self):
        self.model = None
        self.current_model_name = None
        self.current_gpu = None

    def load_model(self, config):
        gpu = config.get("gpu_index", "0")
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
        model_name = config["model"]
        if self.model and self.current_model_name == model_name and self.current_gpu == gpu:
            return
        from faster_whisper import WhisperModel
        self.model = WhisperModel(
            model_name, device="cuda",
            compute_type=config.get("compute_type", "float16"))
        self.current_model_name = model_name
        self.current_gpu = gpu

    def transcribe(self, audio, config):
        if self.model is None:
            self.load_model(config)
        lang = config.get("language")
        segments, _ = self.model.transcribe(
            audio, language=lang, beam_size=5, vad_filter=True)
        return " ".join(seg.text.strip() for seg in segments)


def create_icon(color, size=64):
    pixmap = QPixmap(QSize(size, size))
    pixmap.fill(Qt.transparent)
    p = QPainter(pixmap)
    p.setRenderHint(QPainter.Antialiasing)
    p.setBrush(QColor(color))
    p.setPen(Qt.NoPen)
    p.drawEllipse(8, 8, size - 16, size - 16)
    p.end()
    return QIcon(pixmap)


# ─── Settings Dialog ──────────────────────────────────────────────────────────

class SettingsDialog(QDialog):
    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.config = config
        self.setWindowTitle("SuperWhisper Custom")
        self.setMinimumWidth(480)
        self.setStyleSheet("""
            QDialog { background: #181825; color: #cdd6f4; }
            QGroupBox {
                border: 1px solid #313244; border-radius: 10px;
                margin-top: 14px; padding: 18px 12px 12px 12px;
                font-weight: bold; color: #a6adc8;
            }
            QGroupBox::title {
                subcontrol-origin: margin; left: 14px;
                padding: 0 6px; color: #89b4fa;
            }
            QComboBox {
                background: #1e1e2e; color: #cdd6f4; border: 1px solid #313244;
                border-radius: 8px; padding: 8px 14px; min-height: 30px;
            }
            QComboBox:hover { border-color: #89b4fa; }
            QComboBox::drop-down { border: none; width: 24px; }
            QComboBox QAbstractItemView {
                background: #1e1e2e; color: #cdd6f4;
                selection-background-color: #313244; border: 1px solid #313244;
            }
            QPushButton {
                background: #89b4fa; color: #11111b; border: none;
                border-radius: 10px; padding: 12px 32px;
                font-weight: bold; font-size: 14px;
            }
            QPushButton:hover { background: #b4d0fb; }
            QPushButton#btn_delete { background: #45475a; color: #cdd6f4; padding: 8px 16px; }
            QPushButton#btn_delete:hover { background: #585b70; }
            QPushButton#btn_add { background: #45475a; color: #cdd6f4; padding: 8px 16px; }
            QPushButton#btn_add:hover { background: #585b70; }
            QLabel { color: #a6adc8; }
            QPlainTextEdit {
                background: #1e1e2e; color: #cdd6f4; border: 1px solid #313244;
                border-radius: 8px; padding: 8px; font-size: 12px;
            }
            QPlainTextEdit:focus { border-color: #89b4fa; }
            QPlainTextEdit[readOnly="true"] { color: #a6adc8; }
            QLineEdit {
                background: #1e1e2e; color: #cdd6f4; border: 1px solid #313244;
                border-radius: 8px; padding: 8px 14px; min-height: 30px;
            }
            QLineEdit:focus { border-color: #89b4fa; }
        """)

        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(24, 20, 24, 20)

        title = QLabel("SuperWhisper Custom")
        title.setFont(QFont("Sans", 20, QFont.Bold))
        title.setStyleSheet("color: #cdd6f4;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        sub = QLabel("Ctrl + Alt + Space pour enregistrer")
        sub.setStyleSheet("color: #585b70; font-size: 12px;")
        sub.setAlignment(Qt.AlignCenter)
        layout.addWidget(sub)

        # Model
        mg = QGroupBox("Transcription")
        ml = QFormLayout(mg)
        ml.setSpacing(10)

        self.model_combo = QComboBox()
        for m in AVAILABLE_MODELS:
            self.model_combo.addItem(m, m)
        idx = AVAILABLE_MODELS.index(config["model"]) if config["model"] in AVAILABLE_MODELS else 0
        self.model_combo.setCurrentIndex(idx)
        ml.addRow("Modèle :", self.model_combo)

        self.lang_combo = QComboBox()
        for name, code in AVAILABLE_LANGUAGES:
            self.lang_combo.addItem(name, code)
        for i, (_, code) in enumerate(AVAILABLE_LANGUAGES):
            if code == config.get("language"):
                self.lang_combo.setCurrentIndex(i)
                break
        ml.addRow("Langue :", self.lang_combo)

        self.compute_combo = QComboBox()
        self.compute_combo.addItem("float16 — rapide", "float16")
        self.compute_combo.addItem("int8 — léger", "int8")
        if config.get("compute_type") == "int8":
            self.compute_combo.setCurrentIndex(1)
        ml.addRow("Précision :", self.compute_combo)
        layout.addWidget(mg)

        # Hardware
        hg = QGroupBox("Matériel")
        hl = QFormLayout(hg)
        hl.setSpacing(10)

        self.gpu_combo = QComboBox()
        for gid, gname in get_gpu_list():
            self.gpu_combo.addItem(f"GPU {gid} — {gname}", gid)
        for i in range(self.gpu_combo.count()):
            if self.gpu_combo.itemData(i) == config.get("gpu_index", "1"):
                self.gpu_combo.setCurrentIndex(i)
                break
        hl.addRow("GPU :", self.gpu_combo)

        self.audio_combo = QComboBox()
        for aid, aname in get_audio_inputs():
            self.audio_combo.addItem(aname, aid)
        for i in range(self.audio_combo.count()):
            if self.audio_combo.itemData(i) == config.get("audio_device"):
                self.audio_combo.setCurrentIndex(i)
                break
        hl.addRow("Microphone :", self.audio_combo)
        layout.addWidget(hg)

        # Claude post-processing
        cg = QGroupBox("Post-traitement Claude")
        cvl = QVBoxLayout(cg)
        cvl.setSpacing(10)
        cl = QFormLayout()
        cl.setSpacing(10)

        self.claude_combo = QComboBox()
        self._rebuild_claude_combo()
        self.claude_combo.currentIndexChanged.connect(self._on_claude_mode_changed)
        cl.addRow("Mode :", self.claude_combo)
        cvl.addLayout(cl)

        self.prompt_label = QLabel("Prompt :")
        self.prompt_label.setStyleSheet("color: #a6adc8; font-size: 12px;")
        cvl.addWidget(self.prompt_label)

        self.prompt_edit = QPlainTextEdit()
        self.prompt_edit.setMaximumHeight(120)
        cvl.addWidget(self.prompt_edit)

        btn_row = QHBoxLayout()
        self.btn_add_mode = QPushButton("+ Ajouter un mode")
        self.btn_add_mode.setObjectName("btn_add")
        self.btn_add_mode.clicked.connect(self._add_custom_mode)
        btn_row.addWidget(self.btn_add_mode)

        self.btn_del_mode = QPushButton("Supprimer")
        self.btn_del_mode.setObjectName("btn_delete")
        self.btn_del_mode.clicked.connect(self._del_custom_mode)
        btn_row.addWidget(self.btn_del_mode)
        btn_row.addStretch()
        cvl.addLayout(btn_row)

        layout.addWidget(cg)
        self._on_claude_mode_changed()

        btn = QPushButton("Sauvegarder")
        btn.clicked.connect(self._save)
        layout.addWidget(btn)

    def _rebuild_claude_combo(self):
        self.claude_combo.blockSignals(True)
        self.claude_combo.clear()
        self.claude_combo.addItem("Désactivé", "disabled")
        for mode_id, info in CLAUDE_BUILTIN_MODES.items():
            self.claude_combo.addItem(info["name"], mode_id)
        for cm in self.config.get("claude_custom_modes", []):
            self.claude_combo.addItem(cm["name"], f"custom:{cm['name']}")
        # Select current mode
        current = self.config.get("claude_mode", "disabled")
        for i in range(self.claude_combo.count()):
            if self.claude_combo.itemData(i) == current:
                self.claude_combo.setCurrentIndex(i)
                break
        self.claude_combo.blockSignals(False)

    def _on_claude_mode_changed(self):
        mode = self.claude_combo.currentData()
        if mode == "disabled":
            self.prompt_label.hide()
            self.prompt_edit.hide()
            self.btn_del_mode.hide()
        elif mode in CLAUDE_BUILTIN_MODES:
            self.prompt_label.show()
            self.prompt_edit.show()
            self.prompt_edit.setPlainText(CLAUDE_BUILTIN_MODES[mode]["prompt"])
            self.prompt_edit.setReadOnly(True)
            self.btn_del_mode.hide()
        elif mode and mode.startswith("custom:"):
            name = mode[len("custom:"):]
            self.prompt_label.show()
            self.prompt_edit.show()
            self.prompt_edit.setReadOnly(False)
            for cm in self.config.get("claude_custom_modes", []):
                if cm["name"] == name:
                    self.prompt_edit.setPlainText(cm["prompt"])
                    break
            self.btn_del_mode.show()

    def _add_custom_mode(self):
        name, ok = QInputDialog.getText(
            self, "Nouveau mode", "Nom du mode :",
            text="Mon mode")
        if not ok or not name.strip():
            return
        name = name.strip()
        custom = self.config.get("claude_custom_modes", [])
        if any(cm["name"] == name for cm in custom):
            return
        custom.append({"name": name, "prompt": ""})
        self.config["claude_custom_modes"] = custom
        self._rebuild_claude_combo()
        # Select the new mode
        for i in range(self.claude_combo.count()):
            if self.claude_combo.itemData(i) == f"custom:{name}":
                self.claude_combo.setCurrentIndex(i)
                break
        self._on_claude_mode_changed()

    def _del_custom_mode(self):
        mode = self.claude_combo.currentData()
        if not mode or not mode.startswith("custom:"):
            return
        name = mode[len("custom:"):]
        custom = self.config.get("claude_custom_modes", [])
        self.config["claude_custom_modes"] = [
            cm for cm in custom if cm["name"] != name]
        self.config["claude_mode"] = "disabled"
        self._rebuild_claude_combo()
        self._on_claude_mode_changed()

    def _save(self):
        self.config["model"] = self.model_combo.currentData()
        self.config["language"] = self.lang_combo.currentData()
        self.config["compute_type"] = self.compute_combo.currentData()
        self.config["gpu_index"] = self.gpu_combo.currentData()
        self.config["audio_device"] = self.audio_combo.currentData()
        # Save Claude mode
        mode = self.claude_combo.currentData()
        self.config["claude_mode"] = mode if mode else "disabled"
        # Save custom mode prompt if editing one
        if mode and mode.startswith("custom:"):
            name = mode[len("custom:"):]
            for cm in self.config.get("claude_custom_modes", []):
                if cm["name"] == name:
                    cm["prompt"] = self.prompt_edit.toPlainText()
                    break
        save_config(self.config)
        self.accept()


# ─── Spectrum Widget ──────────────────────────────────────────────────────────

class SpectrumWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.smooth_bars = np.zeros(NUM_BARS)
        self.velocity = np.zeros(NUM_BARS)
        self.peak_bars = np.zeros(NUM_BARS)
        self.peak_decay = np.zeros(NUM_BARS)
        self.setMinimumHeight(70)
        self.setMinimumWidth(360)
        self._anim_timer = QTimer(self)
        self._anim_timer.timeout.connect(self._decay)
        self._anim_timer.setInterval(16)  # ~60fps

    def update_bars(self, fft_data):
        target = np.zeros(NUM_BARS)
        n = min(len(fft_data), NUM_BARS)
        target[:n] = fft_data[:n]
        # Smooth spring-like interpolation
        diff = target - self.smooth_bars
        self.velocity = self.velocity * 0.6 + diff * 0.4
        self.smooth_bars = np.clip(self.smooth_bars + self.velocity, 0, 1)
        # Peak hold
        higher = self.smooth_bars > self.peak_bars
        self.peak_bars[higher] = self.smooth_bars[higher]
        self.peak_decay[higher] = 0
        if not self._anim_timer.isActive():
            self._anim_timer.start()
        self.update()

    def _decay(self):
        self.peak_decay += 0.02
        self.peak_bars = np.maximum(self.peak_bars - self.peak_decay * 0.04, 0)
        if np.max(self.smooth_bars) < 0.005 and np.max(self.peak_bars) < 0.005:
            self._anim_timer.stop()
        self.update()

    def reset(self):
        self.smooth_bars = np.zeros(NUM_BARS)
        self.velocity = np.zeros(NUM_BARS)
        self.peak_bars = np.zeros(NUM_BARS)
        self.peak_decay = np.zeros(NUM_BARS)
        self._anim_timer.stop()
        self.update()

    def paintEvent(self, event):
        from PySide6.QtGui import QLinearGradient
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        bar_w = max(3, (w - (NUM_BARS - 1) * 2) // NUM_BARS)
        gap = 2
        total = NUM_BARS * bar_w + (NUM_BARS - 1) * gap
        ox = (w - total) // 2
        cy = h // 2

        for i in range(NUM_BARS):
            val = self.smooth_bars[i]
            half_h = max(2, int(val * cy * 0.92))
            x = ox + i * (bar_w + gap)

            # Color gradient from center outward: blue -> purple -> pink
            t = i / max(NUM_BARS - 1, 1)
            # Center bars are brighter
            center_weight = 1.0 - abs(t - 0.5) * 2.0
            intensity = 0.6 + center_weight * 0.4

            r = int((137 * (1 - t) + 203 * t) * intensity)
            g = int((180 * (1 - t) + 166 * t) * intensity)
            b = int((250 * (1 - t) + 250 * t) * intensity)
            alpha = int(140 + val * 100)

            # Vertical gradient for each bar
            grad = QLinearGradient(x, cy - half_h, x, cy + half_h)
            grad.setColorAt(0.0, QColor(r, g, b, int(alpha * 0.4)))
            grad.setColorAt(0.35, QColor(r, g, b, alpha))
            grad.setColorAt(0.5, QColor(min(r + 40, 255), min(g + 40, 255), min(b + 20, 255), alpha))
            grad.setColorAt(0.65, QColor(r, g, b, alpha))
            grad.setColorAt(1.0, QColor(r, g, b, int(alpha * 0.4)))

            p.setPen(Qt.NoPen)
            p.setBrush(grad)
            # Symmetric: draw from center up and down
            p.drawRoundedRect(x, cy - half_h, bar_w, half_h * 2, bar_w // 2, bar_w // 2)

            # Peak dots (symmetric)
            peak_val = self.peak_bars[i]
            if peak_val > 0.05:
                peak_h = int(peak_val * cy * 0.92)
                dot_size = bar_w
                p.setBrush(QColor(r, g, b, 90))
                p.drawEllipse(x, cy - peak_h - dot_size // 2, dot_size, dot_size)
                p.drawEllipse(x, cy + peak_h - dot_size // 2, dot_size, dot_size)
        p.end()


# ─── Overlay ──────────────────────────────────────────────────────────────────

class Overlay(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(
            Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint
            | Qt.Tool | Qt.WindowTransparentForInput)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.setWindowTitle("SuperWhisper Overlay")
        self.setFixedSize(480, 110)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.container = QWidget()
        self.container.setStyleSheet(
            "background-color: rgba(17,17,27,230); border-radius: 22px;")
        cl = QVBoxLayout(self.container)
        cl.setContentsMargins(24, 12, 24, 12)
        cl.setSpacing(6)

        self.label = QLabel()
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setFont(QFont("Sans", 11, QFont.DemiBold))
        self.label.setStyleSheet("background: transparent; color: #585b70;")
        cl.addWidget(self.label)

        self.spectrum = SpectrumWidget()
        self.spectrum.setStyleSheet("background: transparent;")
        cl.addWidget(self.spectrum)

        layout.addWidget(self.container)

        # Opacity for fade animations
        self._opacity = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self._opacity)
        self._opacity.setOpacity(1.0)

        # Fade animation
        from PySide6.QtCore import QPropertyAnimation, QEasingCurve
        self._fade_anim = QPropertyAnimation(self._opacity, b"opacity")
        self._fade_anim.setDuration(350)
        self._fade_anim.setEasingCurve(QEasingCurve.Type.InOutCubic)

        # Auto-hide timer
        self._hide_timer = QTimer(self)
        self._hide_timer.setSingleShot(True)
        self._hide_timer.timeout.connect(self._fade_out)

        # Keep-on-top via KWin scripting (Wayland workaround)
        self._raise_timer = QTimer(self)
        self._raise_timer.setSingleShot(True)
        self._raise_timer.timeout.connect(self._ensure_on_top)

        self._state = "idle"  # idle, recording, transcribing

    def _center(self):
        s = QApplication.primaryScreen().geometry()
        self.move((s.width() - self.width()) // 2, int(s.height() * 0.08))

    def _ensure_on_top(self):
        """Force keepAbove via KWin scripting (Wayland-compatible)."""
        if not self.isVisible():
            return
        import tempfile
        script = 'workspace.windowList().forEach(function(w){if(w.caption==="SuperWhisper Overlay")w.keepAbove=true;});'
        try:
            with tempfile.NamedTemporaryFile(
                    mode='w', suffix='.js', delete=False, dir='/tmp',
                    prefix='sw_keepabove_') as f:
                f.write(script)
                path = f.name
            r = subprocess.run(
                ["qdbus", "org.kde.KWin", "/Scripting",
                 "org.kde.kwin.Scripting.loadScript", path,
                 "superwhisper-keepabove"],
                capture_output=True, text=True, timeout=2)
            if r.returncode == 0:
                subprocess.run(
                    ["qdbus", "org.kde.KWin", "/Scripting",
                     "org.kde.kwin.Scripting.start"],
                    capture_output=True, text=True, timeout=2)
                subprocess.run(
                    ["qdbus", "org.kde.KWin", "/Scripting",
                     "org.kde.kwin.Scripting.unloadScript",
                     "superwhisper-keepabove"],
                    capture_output=True, text=True, timeout=2)
            os.unlink(path)
        except Exception as e:
            print(f"[SW] KWin keepAbove error: {e}", flush=True)

    def _fade_in(self):
        self._fade_anim.stop()
        self._fade_anim.setStartValue(self._opacity.opacity())
        self._fade_anim.setEndValue(1.0)
        self._fade_anim.start()

    def _fade_out(self):
        self._fade_anim.stop()
        self._fade_anim.setStartValue(self._opacity.opacity())
        self._fade_anim.setEndValue(0.0)
        self._fade_anim.finished.connect(self._on_fade_out_done)
        self._fade_anim.start()

    def _on_fade_out_done(self):
        try:
            self._fade_anim.finished.disconnect(self._on_fade_out_done)
        except RuntimeError:
            pass
        self.hide()
        self._opacity.setOpacity(1.0)

    def show_recording(self):
        self._state = "recording"
        self.setFixedSize(480, 110)
        self.label.setText("  Enregistrement...")
        self.label.setStyleSheet("background: transparent; color: #bac2de;")
        self.container.setStyleSheet(
            "background-color: rgba(17,17,27,230); border-radius:22px;"
            "border: 1.5px solid rgba(205,214,244,40);")
        self.spectrum.show()
        self.spectrum.reset()
        self._hide_timer.stop()
        self._center()
        self.show()
        self._raise_timer.start(200)  # KWin keepAbove after surface is mapped
        self._fade_in()

    def update_spectrum(self, fft_data):
        self.spectrum.update_bars(fft_data)

    def show_transcribing(self):
        self._state = "transcribing"
        self.label.setText("  Transcription...")
        self.label.setStyleSheet("background: transparent; color: #89b4fa;")
        self.container.setStyleSheet(
            "background-color: rgba(17,17,27,230); border-radius:22px;"
            "border: 1.5px solid rgba(137,180,250,50);")
        self.spectrum.hide()
        self.setFixedSize(480, 50)
        self._center()
        self.show()
        self._raise_timer.start(200)

    def show_reformulating(self):
        self._state = "reformulating"
        self.label.setText("  Reformulation Claude...")
        self.label.setStyleSheet("background: transparent; color: #cba6f7;")
        self.container.setStyleSheet(
            "background-color: rgba(17,17,27,230); border-radius:22px;"
            "border: 1.5px solid rgba(203,166,247,50);")
        self.spectrum.hide()
        self.setFixedSize(480, 50)
        self._center()
        self.show()
        self._raise_timer.start(200)

    def show_done(self):
        """Brief confirmation then fade out."""
        self._state = "idle"
        self.label.setText("  Copié")
        self.label.setStyleSheet("background: transparent; color: #a6e3a1;")
        self.container.setStyleSheet(
            "background-color: rgba(17,17,27,230); border-radius:22px;"
            "border: 1.5px solid rgba(166,227,161,50);")
        self.spectrum.hide()
        self.setFixedSize(480, 50)
        self._center()
        self.show()
        self._raise_timer.start(200)
        self._hide_timer.start(1000)

    def show_error(self, text):
        self._state = "idle"
        self.label.setText(f"  {text}")
        self.label.setStyleSheet("background: transparent; color: #f38ba8;")
        self.container.setStyleSheet(
            "background-color: rgba(17,17,27,230); border-radius:22px;"
            "border: 1.5px solid rgba(243,139,168,50);")
        self.spectrum.hide()
        self.setFixedSize(480, 50)
        self._center()
        self.show()
        self._hide_timer.start(2500)


# ─── Main App ─────────────────────────────────────────────────────────────────

class SuperWhisper(QObject):
    def __init__(self):
        super().__init__()
        self.app = QApplication(sys.argv)
        self.app.setQuitOnLastWindowClosed(False)
        self.app.setApplicationName("SuperWhisper Custom")

        self.config = load_config()
        os.environ["CUDA_VISIBLE_DEVICES"] = self.config.get("gpu_index", "1")

        self.signals = Signals()
        self.recorder = AudioRecorder(self.signals, self.config.get("audio_device", "default"))
        self.transcriber = Transcriber()
        self.is_recording = False

        self.overlay = Overlay()

        self.tray = QSystemTrayIcon()
        self.icon_idle = create_icon("#a6e3a1")
        self.icon_rec = create_icon("#f38ba8")
        self.icon_work = create_icon("#89b4fa")
        self.tray.setIcon(self.icon_idle)
        self.tray.setToolTip("SuperWhisper — Ctrl+Alt+Space")

        menu = QMenu()
        sa = QAction("Paramètres", menu)
        sa.triggered.connect(self._open_settings)
        menu.addAction(sa)
        menu.addSeparator()
        qa = QAction("Quitter", menu)
        qa.triggered.connect(self._quit)
        menu.addAction(qa)
        self.tray.setContextMenu(menu)
        self.tray.activated.connect(self._tray_activated)
        self.tray.show()

        self.signals.recording_started.connect(self._on_rec_start)
        self.signals.recording_stopped.connect(lambda: None)
        self.signals.transcription_started.connect(self._on_trans_start)
        self.signals.reformulation_started.connect(self._on_reform_start)
        self.signals.transcription_done.connect(self._on_trans_done)
        self.signals.error.connect(self._on_error)
        self.signals.audio_level.connect(self._on_audio)

        # Single-instance: write PID and handle SIGUSR1 to open settings
        write_pid()
        self._sigusr_notifier_r, self._sigusr_notifier_w = os.pipe()
        signal.signal(signal.SIGUSR1, lambda *_: os.write(self._sigusr_notifier_w, b'\x00'))
        from PySide6.QtCore import QSocketNotifier
        self._sock_notifier = QSocketNotifier(self._sigusr_notifier_r, QSocketNotifier.Type.Read)
        self._sock_notifier.activated.connect(self._on_sigusr1)

        if not os.path.exists(CONFIG_PATH):
            QTimer.singleShot(500, self._open_settings)

        threading.Thread(target=self._preload, daemon=True).start()
        threading.Thread(target=self._hotkey_listener, daemon=True).start()

    def _on_sigusr1(self):
        os.read(self._sigusr_notifier_r, 1)
        self._open_settings()

    def _preload(self):
        try:
            self.transcriber.load_model(self.config)
        except Exception as e:
            self.signals.error.emit(f"Erreur modèle: {e}")

    def _tray_activated(self, reason):
        if reason == QSystemTrayIcon.ActivationReason.DoubleClick:
            self._open_settings()

    def _open_settings(self):
        dlg = SettingsDialog(dict(self.config))
        if dlg.exec() == QDialog.Accepted:
            old = (self.config.get("model"), self.config.get("gpu_index"))
            self.config = load_config()
            os.environ["CUDA_VISIBLE_DEVICES"] = self.config.get("gpu_index", "1")
            self.recorder.device = self.config.get("audio_device", "default")
            if (self.config["model"], self.config["gpu_index"]) != old:
                self.transcriber.model = None
                threading.Thread(target=self._preload, daemon=True).start()

    def _hotkey_listener(self):
        """Listen for Ctrl+Alt+Space via pynput with auto-restart on crash."""
        from pynput import keyboard
        ctrl = False
        alt = False

        def on_press(key):
            nonlocal ctrl, alt
            if key in (keyboard.Key.ctrl_l, keyboard.Key.ctrl_r):
                ctrl = True
            elif key in (keyboard.Key.alt_l, keyboard.Key.alt_r):
                alt = True
            elif key == keyboard.Key.space and ctrl and alt:
                self._toggle()

        def on_release(key):
            nonlocal ctrl, alt
            if key in (keyboard.Key.ctrl_l, keyboard.Key.ctrl_r):
                ctrl = False
            elif key in (keyboard.Key.alt_l, keyboard.Key.alt_r):
                alt = False

        print("[SW] Hotkey listener (pynput) started", flush=True)
        while True:
            try:
                with keyboard.Listener(on_press=on_press, on_release=on_release) as l:
                    l.join()
            except Exception as e:
                print(f"[SW] pynput listener crashed ({e}), restarting in 2s", flush=True)
                time.sleep(2)

    def _toggle(self):
        if not self.is_recording:
            self.is_recording = True
            self.recorder.start()
            self.signals.recording_started.emit()
            print(f"[SW] Recording started — device={self.recorder.device}", flush=True)
        else:
            self.is_recording = False
            audio = self.recorder.stop()
            self.signals.recording_stopped.emit()
            print(f"[SW] Recording stopped — {len(audio)} samples ({len(audio)/SAMPLE_RATE:.1f}s)", flush=True)
            if len(audio) < SAMPLE_RATE * 0.3:
                self.signals.error.emit("Trop court")
                return
            self.signals.transcription_started.emit()
            threading.Thread(target=self._transcribe, args=(audio,), daemon=True).start()

    def _transcribe(self, audio):
        try:
            print(f"[SW] Transcribing {len(audio)/SAMPLE_RATE:.1f}s audio...", flush=True)
            text = self.transcriber.transcribe(audio, self.config)
            if not text.strip():
                print("[SW] No text detected", flush=True)
                self.signals.error.emit("Aucun texte détecté")
                return
            text = text.strip()
            print(f"[SW] Result: {text[:80]}", flush=True)
            # Claude post-processing if enabled
            mode = self.config.get("claude_mode", "disabled")
            if mode != "disabled":
                prompt = self._get_claude_prompt(mode)
                if prompt:
                    self.signals.reformulation_started.emit()
                    text = self._claude_reformat(text, prompt)
            self.signals.transcription_done.emit(text)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[SW] Error: {e}", flush=True)
            self.signals.error.emit(f"Erreur: {e}")

    def _get_claude_prompt(self, mode):
        if mode in CLAUDE_BUILTIN_MODES:
            return CLAUDE_BUILTIN_MODES[mode]["prompt"]
        if mode.startswith("custom:"):
            name = mode[len("custom:"):]
            for cm in self.config.get("claude_custom_modes", []):
                if cm["name"] == name:
                    return cm["prompt"]
        return None

    def _claude_reformat(self, text, prompt):
        try:
            print(f"[SW] Claude reformulating...", flush=True)
            result = subprocess.run(
                [CLAUDE_BIN, "-p", prompt],
                input=text, capture_output=True, text=True, timeout=30)
            if result.returncode == 0 and result.stdout.strip():
                reformulated = result.stdout.strip()
                print(f"[SW] Claude result: {reformulated[:80]}", flush=True)
                return reformulated
            print(f"[SW] Claude failed: rc={result.returncode} err={result.stderr.strip()[:100]}", flush=True)
        except subprocess.TimeoutExpired:
            print("[SW] Claude timeout (30s)", flush=True)
        except Exception as e:
            print(f"[SW] Claude error: {e}", flush=True)
        return text  # fallback to original transcription

    def _on_rec_start(self):
        self.tray.setIcon(self.icon_rec)
        self.overlay.show_recording()

    def _on_trans_start(self):
        self.tray.setIcon(self.icon_work)
        self.overlay.show_transcribing()

    def _on_reform_start(self):
        self.tray.setIcon(self.icon_work)
        self.overlay.show_reformulating()

    def _on_trans_done(self, text):
        self.tray.setIcon(self.icon_idle)
        # Kill previous wl-copy if still running (it stays alive to serve clipboard)
        if hasattr(self, '_wl_copy_proc') and self._wl_copy_proc:
            try:
                self._wl_copy_proc.kill()
            except OSError:
                pass
        # Start wl-copy — don't wait, it stays alive to serve clipboard on Wayland
        try:
            self._wl_copy_proc = subprocess.Popen(
                ["wl-copy", "--", text],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"[SW] wl-copy started (pid={self._wl_copy_proc.pid})", flush=True)
        except Exception as e:
            print(f"[SW] wl-copy error: {e}", flush=True)
        # Show "Copié" immediately
        self.overlay.show_done()
        # Auto-paste in background (don't block UI)
        threading.Thread(target=self._auto_paste, daemon=True).start()

    def _on_error(self, text):
        self.tray.setIcon(self.icon_idle)
        self.overlay.show_error(text)

    def _on_audio(self, fft):
        self.overlay.update_spectrum(fft)

    def _auto_paste(self):
        """Auto-paste from clipboard after a short delay (runs in background thread)."""
        try:
            time.sleep(0.15)
            r = subprocess.run(["xdotool", "key", "ctrl+v"], capture_output=True, text=True, timeout=3)
            print(f"[SW] xdotool rc={r.returncode} err={r.stderr.strip()}", flush=True)
        except Exception as e:
            print(f"[SW] auto-paste error: {e}", flush=True)

    def _quit(self):
        remove_pid()
        self.tray.hide()
        self.app.quit()

    def run(self):
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        return self.app.exec()


if __name__ == "__main__":
    if is_already_running():
        print("SuperWhisper déjà en cours — ouverture des paramètres.")
        sys.exit(0)
    app = SuperWhisper()
    sys.exit(app.run())
