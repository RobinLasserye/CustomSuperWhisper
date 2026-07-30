"""Presse-papier et collage automatique."""

import subprocess
import time

from .runtime import IS_LINUX, log


def clipboard_copy(text):
    """Copie dans le presse-papier. Sous Wayland, QClipboard n'est pas fiable sans focus."""
    if IS_LINUX:
        for command, use_stdin in (
                (["wl-copy", "--", text], False),
                (["xclip", "-selection", "clipboard"], True)):
            try:
                subprocess.run(
                    command,
                    input=text.encode() if use_stdin else None,
                    stdin=None if use_stdin else subprocess.DEVNULL,
                    timeout=3, check=True,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return True
            except (FileNotFoundError, subprocess.SubprocessError):
                continue

    try:
        from PySide6.QtWidgets import QApplication
        clipboard = QApplication.clipboard()
        if clipboard is not None:
            clipboard.setText(text)
            return True
    except Exception as exc:
        log(f"presse-papier : échec ({exc})")
    return False


def auto_paste(delay=0.15):
    """Simule Ctrl+V. `delay` laisse le temps au focus de revenir sur l'application cible."""
    try:
        time.sleep(delay)
        from pynput.keyboard import Controller, Key
        keyboard = Controller()
        keyboard.press(Key.ctrl_l)
        keyboard.press("v")
        keyboard.release("v")
        keyboard.release(Key.ctrl_l)
        return True
    except Exception as exc:
        log(f"collage automatique : échec ({exc})")
        return False
