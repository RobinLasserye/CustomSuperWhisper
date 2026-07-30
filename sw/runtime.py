"""Plateforme, chemins de configuration et amorçage des bibliothèques CUDA.

Ce module ne doit importer que la bibliothèque standard : il est chargé avant tout le reste,
y compris avant `faster_whisper`, pour pouvoir corriger l'environnement du processus.
"""

import os
import platform
import sys

IS_WINDOWS = platform.system() == "Windows"
IS_LINUX = platform.system() == "Linux"

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if IS_WINDOWS:
    CONFIG_DIR = os.path.join(
        os.environ.get("APPDATA", os.path.expanduser("~")), "superwhisper-custom")
else:
    CONFIG_DIR = os.path.expanduser("~/.config/superwhisper-custom")

CONFIG_PATH = os.path.join(CONFIG_DIR, "config.json")
PID_FILE = os.path.join(CONFIG_DIR, "superwhisper.pid")

if IS_WINDOWS:
    _CLAUDE_BIN = os.path.join(os.environ.get("APPDATA", ""), "npm", "claude.cmd")
    CLAUDE_BIN = _CLAUDE_BIN if os.path.exists(_CLAUDE_BIN) else "claude"
else:
    CLAUDE_BIN = os.path.expanduser("~/.local/bin/claude")

_CUDA_PACKAGES = ("cublas", "cudnn")
_CUDA_SUBDIR = "bin" if IS_WINDOWS else "lib"


def log(message):
    print(f"[SW] {message}", flush=True)


def _nvidia_roots():
    """Emplacements possibles du paquet `nvidia` (wheels cublas/cudnn)."""
    roots = []
    try:
        import importlib.util
        spec = importlib.util.find_spec("nvidia")
        if spec and spec.submodule_search_locations:
            roots.extend(spec.submodule_search_locations)
    except Exception:
        pass
    if IS_WINDOWS:
        roots.append(os.path.join(PROJECT_DIR, ".venv", "Lib", "site-packages", "nvidia"))
    else:
        roots.append(os.path.join(
            PROJECT_DIR, ".venv", "lib",
            f"python{sys.version_info.major}.{sys.version_info.minor}",
            "site-packages", "nvidia"))
    return roots


def cuda_lib_dirs():
    """Répertoires de bibliothèques CUDA fournis par les wheels nvidia, s'ils existent."""
    dirs, seen = [], set()
    for root in _nvidia_roots():
        for package in _CUDA_PACKAGES:
            path = os.path.join(root, package, _CUDA_SUBDIR)
            if not os.path.isdir(path):
                continue
            # `lib` et `lib64` pointent souvent sur le même dossier : une seule entrée suffit
            key = os.path.realpath(path)
            if key in seen:
                continue
            seen.add(key)
            dirs.append(path)
    return dirs


def ensure_cuda_libs(entry_script=None):
    """Rendre libcublas/libcudnn trouvables par le loader dynamique.

    Sous Linux, modifier `LD_LIBRARY_PATH` dans `os.environ` n'a **aucun effet** : le loader ne
    relit pas la variable après le démarrage du processus. La seule solution fiable est de se
    ré-exécuter avec l'environnement corrigé, ce que fait cette fonction (une seule fois, la garde
    `SW_CUDA_REEXEC` empêche toute boucle). Sans ça, l'application ne marche que lancée depuis un
    `.desktop` qui pose la variable, et échoue sur
    `RuntimeError: Library libcublas.so.12 is not found` partout ailleurs.

    Retourne True si un ré-exec a été tenté (en pratique la fonction ne revient jamais dans ce
    cas), False s'il n'y avait rien à faire.
    """
    dirs = cuda_lib_dirs()
    if not dirs:
        return False

    if IS_WINDOWS:
        # Depuis Python 3.8, patcher PATH ne suffit plus pour la recherche de DLL.
        for path in dirs:
            try:
                os.add_dll_directory(path)
            except (OSError, AttributeError):
                pass
        os.environ["PATH"] = os.pathsep.join(dirs) + os.pathsep + os.environ.get("PATH", "")
        return False

    current = [p for p in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep) if p]
    if all(d in current for d in dirs):
        return False
    if os.environ.get("SW_CUDA_REEXEC") == "1":
        log("CUDA : ré-exec déjà tenté, on continue avec l'environnement actuel")
        return False

    script = entry_script or os.path.abspath(sys.argv[0])
    if not os.path.isfile(script):
        return False

    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = os.pathsep.join(dirs + current)
    env["SW_CUDA_REEXEC"] = "1"
    log(f"CUDA : ré-exec avec LD_LIBRARY_PATH={env['LD_LIBRARY_PATH']}")
    try:
        os.execve(sys.executable, [sys.executable, script] + sys.argv[1:], env)
    except OSError as exc:
        log(f"CUDA : ré-exec impossible ({exc})")
        return False
    return True
