"""Instance unique : détection, signalement et fichier de PID.

Piège documenté : `pgrep -f superwhisper.py` matche aussi le shell parent qui contient ce texte
dans sa ligne de commande, et un `pkill` sur le même motif se suicide. D'où la vérification de
l'exécutable et l'exclusion du processus courant et de son parent.
"""

import os
import subprocess
import sys

from .runtime import IS_WINDOWS, PID_FILE, log

if not IS_WINDOWS:
    import signal

WINDOWS_EVENT_NAME = "SuperWhisperCustomShowSettings"


def find_existing_instances():
    """PID des autres processus superwhisper.py (ni nous, ni notre parent)."""
    my_pid = os.getpid()
    my_ppid = os.getppid()
    pids = []

    if IS_WINDOWS:
        try:
            output = subprocess.check_output(
                ["wmic", "process", "where",
                 "name like '%python%' and commandline like '%superwhisper.py%'",
                 "get", "processid,commandline"],
                text=True, stderr=subprocess.DEVNULL)
        except Exception:
            return pids
        for line in output.strip().split("\n")[1:]:
            line = line.strip()
            if not line:
                continue
            parts = line.rsplit(None, 1)
            if len(parts) < 2 or not parts[1].isdigit():
                continue
            cmdline, pid = parts[0], int(parts[1])
            if pid in (my_pid, my_ppid):
                continue
            if " -c " in cmdline or ' -c"' in cmdline or " -c'" in cmdline:
                continue
            pids.append(pid)
        return pids

    try:
        output = subprocess.check_output(
            ["pgrep", "-f", "superwhisper[.]py"], text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return pids

    for line in output.split("\n"):
        line = line.strip()
        if not line.isdigit():
            continue
        pid = int(line)
        if pid in (my_pid, my_ppid):
            continue
        if _is_python_process(pid):
            pids.append(pid)
    return pids


def _is_python_process(pid):
    """Évite de compter un shell dont la ligne de commande mentionne le script."""
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as handle:
            argv = handle.read().split(b"\0")
    except OSError:
        return False
    if not argv or not argv[0]:
        return False
    executable = os.path.basename(argv[0].decode(errors="replace"))
    return executable.startswith("python")


def signal_existing_instance(pid=None):
    """Demande à l'instance en cours d'ouvrir ses réglages."""
    if IS_WINDOWS:
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            event = kernel32.OpenEventW(0x2, False, WINDOWS_EVENT_NAME)
            if event:
                kernel32.SetEvent(event)
                kernel32.CloseHandle(event)
                return True
        except Exception:
            pass
        return False

    if pid is None:
        return False
    try:
        os.kill(pid, signal.SIGUSR1)
        return True
    except OSError:
        return False


def _read_pid_file():
    try:
        with open(PID_FILE) as handle:
            return int(handle.read().strip())
    except (OSError, ValueError):
        return None


def _remove_pid_file():
    try:
        os.remove(PID_FILE)
    except OSError:
        pass


def is_already_running():
    """True si une autre instance tourne (et a été prévenue d'ouvrir ses réglages)."""
    pid = _read_pid_file() if os.path.exists(PID_FILE) else None

    if pid is not None and pid != os.getpid():
        if IS_WINDOWS:
            if _windows_process_alive(pid):
                signal_existing_instance()
                return True
            _remove_pid_file()
        else:
            try:
                os.kill(pid, 0)
            except OSError:
                _remove_pid_file()
            else:
                signal_existing_instance(pid)
                return True

    for other in find_existing_instances():
        if signal_existing_instance(other) or IS_WINDOWS:
            return True
    return False


def _windows_process_alive(pid):
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x1000, False, pid)
        if handle:
            kernel32.CloseHandle(handle)
            return True
    except Exception:
        pass
    return False


def write_pid():
    os.makedirs(os.path.dirname(PID_FILE), exist_ok=True)
    with open(PID_FILE, "w") as handle:
        handle.write(str(os.getpid()))


def remove_pid():
    _remove_pid_file()


def relaunch_detached(entry_script):
    """Relance l'application détachée du terminal courant (utilisé après un changement de GPU)."""
    try:
        subprocess.Popen([sys.executable, entry_script],
                         start_new_session=True,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except OSError as exc:
        log(f"relance impossible ({exc})")
        return False
