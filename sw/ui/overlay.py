"""Overlay flottant : spectre pendant l'enregistrement, états ensuite."""

import os
import subprocess
import tempfile

import numpy as np
from PySide6.QtCore import QEasingCurve, QPropertyAnimation, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QLinearGradient, QPainter
from PySide6.QtWidgets import (
    QApplication, QGraphicsOpacityEffect, QLabel, QVBoxLayout, QWidget,
)

from ..audio import NUM_BARS
from ..runtime import IS_WINDOWS, log
from . import style

WINDOW_TITLE = "SuperWhisper Overlay"


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
        self._anim_timer.setInterval(16)

    def update_bars(self, fft_data):
        target = np.zeros(NUM_BARS)
        count = min(len(fft_data), NUM_BARS)
        target[:count] = fft_data[:count]
        diff = target - self.smooth_bars
        self.velocity = self.velocity * 0.6 + diff * 0.4
        self.smooth_bars = np.clip(self.smooth_bars + self.velocity, 0, 1)
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

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        width, height = self.width(), self.height()
        bar_width = max(3, (width - (NUM_BARS - 1) * 2) // NUM_BARS)
        gap = 2
        total = NUM_BARS * bar_width + (NUM_BARS - 1) * gap
        origin_x = (width - total) // 2
        center_y = height // 2

        for index in range(NUM_BARS):
            value = self.smooth_bars[index]
            half_height = max(2, int(value * center_y * 0.92))
            x = origin_x + index * (bar_width + gap)

            position = index / max(NUM_BARS - 1, 1)
            center_weight = 1.0 - abs(position - 0.5) * 2.0
            intensity = 0.6 + center_weight * 0.4

            red = int((137 * (1 - position) + 203 * position) * intensity)
            green = int((180 * (1 - position) + 166 * position) * intensity)
            blue = int((250 * (1 - position) + 250 * position) * intensity)
            alpha = int(140 + value * 100)

            gradient = QLinearGradient(x, center_y - half_height, x, center_y + half_height)
            gradient.setColorAt(0.0, QColor(red, green, blue, int(alpha * 0.4)))
            gradient.setColorAt(0.35, QColor(red, green, blue, alpha))
            gradient.setColorAt(0.5, QColor(min(red + 40, 255), min(green + 40, 255),
                                           min(blue + 20, 255), alpha))
            gradient.setColorAt(0.65, QColor(red, green, blue, alpha))
            gradient.setColorAt(1.0, QColor(red, green, blue, int(alpha * 0.4)))

            painter.setPen(Qt.NoPen)
            painter.setBrush(gradient)
            painter.drawRoundedRect(x, center_y - half_height, bar_width, half_height * 2,
                                    bar_width // 2, bar_width // 2)

            peak = self.peak_bars[index]
            if peak > 0.05:
                peak_height = int(peak * center_y * 0.92)
                dot = bar_width
                painter.setBrush(QColor(red, green, blue, 90))
                painter.drawEllipse(x, center_y - peak_height - dot // 2, dot, dot)
                painter.drawEllipse(x, center_y + peak_height - dot // 2, dot, dot)
        painter.end()


class Overlay(QWidget):
    def __init__(self):
        super().__init__()
        flags = Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool
        if not IS_WINDOWS:
            flags |= Qt.WindowTransparentForInput
        self.setWindowFlags(flags)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.setWindowTitle(WINDOW_TITLE)
        self.setFixedSize(480, 110)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.container = QWidget()
        container_layout = QVBoxLayout(self.container)
        container_layout.setContentsMargins(24, 12, 24, 12)
        container_layout.setSpacing(6)

        self.label = QLabel()
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setFont(QFont("Sans", 11, QFont.DemiBold))
        container_layout.addWidget(self.label)

        self.spectrum = SpectrumWidget()
        self.spectrum.setStyleSheet("background: transparent;")
        container_layout.addWidget(self.spectrum)

        layout.addWidget(self.container)

        self._opacity = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self._opacity)
        self._opacity.setOpacity(1.0)

        self._fade_anim = QPropertyAnimation(self._opacity, b"opacity")
        self._fade_anim.setDuration(350)
        self._fade_anim.setEasingCurve(QEasingCurve.Type.InOutCubic)

        self._hide_timer = QTimer(self)
        self._hide_timer.setSingleShot(True)
        self._hide_timer.timeout.connect(self._fade_out)

        self._raise_timer = QTimer(self)
        self._raise_timer.setSingleShot(True)
        self._raise_timer.timeout.connect(self._ensure_on_top)

    # — Placement —

    def _center(self):
        screen = QApplication.primaryScreen().geometry()
        self.move((screen.width() - self.width()) // 2, int(screen.height() * 0.08))

    def _ensure_on_top(self):
        """Force keepAbove via le scripting KWin (Qt ne suffit pas sous Wayland)."""
        if not self.isVisible() or IS_WINDOWS:
            return
        script = ('workspace.windowList().forEach(function(w){'
                  f'if(w.caption==="{WINDOW_TITLE}")w.keepAbove=true;}});')
        path = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".js", delete=False,
                                             dir="/tmp", prefix="sw_keepabove_") as handle:
                handle.write(script)
                path = handle.name
            loaded = subprocess.run(
                ["qdbus", "org.kde.KWin", "/Scripting",
                 "org.kde.kwin.Scripting.loadScript", path, "superwhisper-keepabove"],
                capture_output=True, text=True, timeout=2)
            if loaded.returncode == 0:
                for method in ("start", "unloadScript"):
                    args = ["qdbus", "org.kde.KWin", "/Scripting",
                            f"org.kde.kwin.Scripting.{method}"]
                    if method == "unloadScript":
                        args.append("superwhisper-keepabove")
                    subprocess.run(args, capture_output=True, text=True, timeout=2)
        except Exception as exc:
            log(f"KWin keepAbove : {exc}")
        finally:
            if path:
                try:
                    os.unlink(path)
                except OSError:
                    pass

    # — Fondus —

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

    def hide_now(self):
        """Masque immédiatement (avant d'ouvrir le sélecteur, qui prend le focus)."""
        self._hide_timer.stop()
        self._fade_anim.stop()
        self.hide()
        self._opacity.setOpacity(1.0)

    # — États —

    def _show_state(self, text, color, border_rgba, height=50, hide_after=None):
        self.label.setText(text)
        self.label.setStyleSheet(f"background: transparent; color: {color};")
        self.container.setStyleSheet(
            "background-color: rgba(17,17,27,230); border-radius: 22px;"
            f"border: 1.5px solid {border_rgba};")
        self.spectrum.hide()
        self.setFixedSize(480, height)
        self._center()
        self.show()
        self._raise_timer.start(200)
        if hide_after:
            self._hide_timer.start(hide_after)
        else:
            self._hide_timer.stop()

    def show_recording(self):
        self.label.setText("  Enregistrement...")
        self.label.setStyleSheet(f"background: transparent; color: {style.TEXT};")
        self.container.setStyleSheet(
            "background-color: rgba(17,17,27,230); border-radius: 22px;"
            "border: 1.5px solid rgba(205,214,244,40);")
        self.setFixedSize(480, 110)
        self.spectrum.show()
        self.spectrum.reset()
        self._hide_timer.stop()
        self._center()
        self.show()
        self._raise_timer.start(200)
        self._fade_in()

    def update_spectrum(self, fft_data):
        self.spectrum.update_bars(fft_data)

    def show_transcribing(self):
        self._show_state("  Transcription...", style.ACCENT, "rgba(137,180,250,50)")

    def show_reformulating(self, label="Reformulation locale"):
        self._show_state(f"  {label}...", style.MAUVE, "rgba(203,166,247,50)")

    def show_done(self, label="Copié"):
        self._show_state(f"  {label}", style.GREEN, "rgba(166,227,161,50)", hide_after=1200)

    def show_warning(self, text):
        self._show_state(f"  {text}", style.ORANGE, "rgba(250,179,135,50)", hide_after=3500)

    def show_error(self, text):
        self._show_state(f"  {text}", style.RED, "rgba(243,139,168,50)", hide_after=2500)
