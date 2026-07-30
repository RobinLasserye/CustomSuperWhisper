#!/usr/bin/env python3
"""
SuperWhisper Custom — Script d'installation universel (Linux + Windows).

Ce script :
  1. Détecte le système (Linux ou Windows)
  2. Crée un environnement virtuel Python
  3. Installe toutes les dépendances (PySide6, faster-whisper, pynput, sounddevice, etc.)
  4. Télécharge le modèle Whisper par défaut (large-v3)
  5. Configure le raccourci Ctrl+Alt+Space (hotkey globale via pynput, pas besoin de config système)
  6. Met en place le démarrage automatique au boot

Usage :
  python install.py              # Installation complète
  python install.py --model tiny # Installer avec un modèle plus léger
  python install.py --no-autostart  # Sans démarrage automatique
"""

import os
import sys
import platform
import subprocess
import shutil
import argparse

IS_WINDOWS = platform.system() == "Windows"
IS_LINUX = platform.system() == "Linux"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Colors for terminal output
if IS_WINDOWS:
    # Enable ANSI on Windows
    os.system("")

GREEN = "\033[92m"
BLUE = "\033[94m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
RESET = "\033[0m"


def info(msg):
    print(f"{BLUE}[INFO]{RESET} {msg}")


def success(msg):
    print(f"{GREEN}[OK]{RESET} {msg}")


def warn(msg):
    print(f"{YELLOW}[WARN]{RESET} {msg}")


def error(msg):
    print(f"{RED}[ERREUR]{RESET} {msg}")


def header(msg):
    print(f"\n{BOLD}{BLUE}{'=' * 60}{RESET}")
    print(f"{BOLD}{BLUE}  {msg}{RESET}")
    print(f"{BOLD}{BLUE}{'=' * 60}{RESET}\n")


def run(cmd, check=True, **kwargs):
    """Run a command and print it."""
    if isinstance(cmd, list):
        display = " ".join(cmd)
    else:
        display = cmd
    info(f"$ {display}")
    return subprocess.run(cmd, check=check, **kwargs)


def get_python():
    """Get the current Python executable."""
    return sys.executable


def get_venv_python():
    """Get the venv Python executable path."""
    if IS_WINDOWS:
        return os.path.join(SCRIPT_DIR, ".venv", "Scripts", "python.exe")
    return os.path.join(SCRIPT_DIR, ".venv", "bin", "python")


def get_venv_pip():
    """Get the venv pip executable path."""
    if IS_WINDOWS:
        return os.path.join(SCRIPT_DIR, ".venv", "Scripts", "pip.exe")
    return os.path.join(SCRIPT_DIR, ".venv", "bin", "pip")


def check_prerequisites():
    """Check that Python 3.10+ and NVIDIA GPU are available."""
    header("Vérification des prérequis")

    # Python version
    v = sys.version_info
    if v.major < 3 or (v.major == 3 and v.minor < 10):
        error(f"Python 3.10+ requis (trouvé: {v.major}.{v.minor}.{v.micro})")
        sys.exit(1)
    success(f"Python {v.major}.{v.minor}.{v.micro}")

    # NVIDIA GPU check
    nvidia_ok = False
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10)
        if result.returncode == 0 and result.stdout.strip():
            for line in result.stdout.strip().split("\n"):
                success(f"GPU détecté: {line.strip()}")
            nvidia_ok = True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    if not nvidia_ok:
        warn("Aucun GPU NVIDIA détecté. L'application fonctionnera en mode CPU (plus lent).")
        warn("Pour de meilleures performances, installez les pilotes NVIDIA + CUDA Toolkit.")

    # Check git (optional)
    try:
        subprocess.run(["git", "--version"], capture_output=True, check=True)
        success("Git disponible")
    except (FileNotFoundError, subprocess.CalledProcessError):
        warn("Git non trouvé (optionnel)")

    return nvidia_ok


def create_venv():
    """Create Python virtual environment."""
    header("Création de l'environnement virtuel")

    venv_path = os.path.join(SCRIPT_DIR, ".venv")
    if os.path.exists(venv_path):
        info("Environnement virtuel existant détecté, mise à jour...")
    else:
        run([get_python(), "-m", "venv", venv_path])
        success("Environnement virtuel créé")

    # Upgrade pip
    run([get_venv_python(), "-m", "pip", "install", "--upgrade", "pip"])
    success("pip mis à jour")


def install_dependencies(has_gpu):
    """Install all Python dependencies."""
    header("Installation des dépendances")

    pip = get_venv_pip()

    # Core dependencies
    deps = [
        "PySide6",
        "numpy",
        "sounddevice",
        "pynput",
    ]

    # faster-whisper with CUDA support
    if has_gpu:
        deps.extend([
            "faster-whisper",
            "nvidia-cublas-cu12",
            "nvidia-cudnn-cu12",
        ])
    else:
        deps.append("faster-whisper")

    info("Installation des paquets Python...")
    run([pip, "install", "--upgrade"] + deps)
    success("Toutes les dépendances installées")

    # Linux-specific system dependencies
    if IS_LINUX:
        info("Vérification des dépendances système Linux...")
        missing = []
        for cmd, pkg in [("xdotool", "xdotool")]:
            if not shutil.which(cmd):
                missing.append(pkg)
        if missing:
            warn(f"Paquets système manquants (optionnels): {', '.join(missing)}")
            warn("Installez-les avec: sudo dnf install " + " ".join(missing))
            warn("  ou: sudo apt install " + " ".join(missing))


def download_model(model_name):
    """Pre-download the Whisper model."""
    header(f"Téléchargement du modèle Whisper ({model_name})")

    venv_python = get_venv_python()

    # Use faster-whisper to download the model
    download_script = f"""
import sys
print("Téléchargement du modèle {model_name}...")
print("(cela peut prendre quelques minutes selon votre connexion)")
try:
    from faster_whisper import WhisperModel
    # This will download the model if not cached
    model = WhisperModel("{model_name}", device="cpu", compute_type="int8")
    print("Modèle téléchargé et vérifié avec succès!")
except Exception as e:
    print(f"Erreur lors du téléchargement: {{e}}", file=sys.stderr)
    print("Le modèle sera téléchargé au premier lancement.", file=sys.stderr)
"""
    result = run([venv_python, "-c", download_script], check=False)
    if result.returncode == 0:
        success(f"Modèle {model_name} prêt")
    else:
        warn(f"Le modèle {model_name} sera téléchargé au premier lancement")


def setup_autostart():
    """Configure the application to start at boot."""
    header("Configuration du démarrage automatique")

    venv_python = get_venv_python()
    app_script = os.path.join(SCRIPT_DIR, "superwhisper.py")

    if IS_WINDOWS:
        _setup_autostart_windows(venv_python, app_script)
    elif IS_LINUX:
        _setup_autostart_linux(venv_python, app_script)


def _setup_autostart_windows(python_exe, app_script):
    """Create a shortcut in the Windows Startup folder."""
    startup_dir = os.path.join(
        os.environ.get("APPDATA", ""),
        "Microsoft", "Windows", "Start Menu", "Programs", "Startup")

    if not os.path.exists(startup_dir):
        warn(f"Dossier Startup introuvable: {startup_dir}")
        return

    # Create a .bat launcher for reliability
    bat_path = os.path.join(startup_dir, "SuperWhisperCustom.bat")
    bat_content = f'@echo off\r\nstart "" /B "{python_exe}" "{app_script}"\r\n'

    with open(bat_path, "w") as f:
        f.write(bat_content)
    success(f"Démarrage automatique configuré: {bat_path}")

    # Also create a .vbs wrapper to hide the console window
    vbs_path = os.path.join(startup_dir, "SuperWhisperCustom.vbs")
    vbs_content = (
        f'Set WshShell = CreateObject("WScript.Shell")\r\n'
        f'WshShell.Run """{python_exe}"" ""{app_script}""", 0, False\r\n'
    )
    with open(vbs_path, "w") as f:
        f.write(vbs_content)

    # Remove the .bat since we have the .vbs (which hides the console)
    try:
        os.remove(bat_path)
    except OSError:
        pass

    success(f"Lanceur sans console: {vbs_path}")


def _setup_autostart_linux(python_exe, app_script):
    """Create a .desktop autostart entry on Linux."""
    autostart_dir = os.path.expanduser("~/.config/autostart")
    os.makedirs(autostart_dir, exist_ok=True)

    desktop_path = os.path.join(autostart_dir, "superwhisper-custom.desktop")
    # Le `sleep 3` et la phase KDE 2 sont nécessaires sur Plasma : lancée trop tôt, l'application
    # démarre avant que la barre système existe et son icône n'apparaît jamais.
    desktop_content = f"""[Desktop Entry]
Type=Application
Name=SuperWhisper Custom
Comment=Transcription vocale locale avec Ctrl+Alt+Space
Exec=bash -c 'sleep 3 && exec "{python_exe}" "{app_script}"'
Hidden=false
NoDisplay=false
X-GNOME-Autostart-enabled=true
X-KDE-autostart-phase=2
X-KDE-autostart-after=panel
StartupNotify=false
Terminal=false
Categories=Utility;Audio;
"""
    with open(desktop_path, "w") as f:
        f.write(desktop_content)

    success(f"Démarrage automatique configuré: {desktop_path}")


def create_launcher():
    """Create convenient launcher scripts."""
    header("Création des lanceurs")

    venv_python = get_venv_python()
    app_script = os.path.join(SCRIPT_DIR, "superwhisper.py")

    if IS_WINDOWS:
        # .bat launcher
        bat_path = os.path.join(SCRIPT_DIR, "superwhisper.bat")
        bat_content = f'@echo off\r\n"{venv_python}" "{app_script}" %*\r\n'
        with open(bat_path, "w") as f:
            f.write(bat_content)
        success(f"Lanceur créé: {bat_path}")

        # .vbs launcher (no console window)
        vbs_path = os.path.join(SCRIPT_DIR, "superwhisper.vbs")
        vbs_content = (
            f'Set WshShell = CreateObject("WScript.Shell")\r\n'
            f'WshShell.Run """{venv_python}"" ""{app_script}""", 0, False\r\n'
        )
        with open(vbs_path, "w") as f:
            f.write(vbs_content)
        success(f"Lanceur sans console: {vbs_path}")
    else:
        # Shell launcher
        sh_path = os.path.join(SCRIPT_DIR, "superwhisper.sh")
        sh_content = f"""#!/bin/bash
exec "{venv_python}" "{app_script}" "$@"
"""
        with open(sh_path, "w") as f:
            f.write(sh_content)
        os.chmod(sh_path, 0o755)
        success(f"Lanceur créé: {sh_path}")


def print_summary():
    """Print final summary."""
    header("Installation terminée !")

    print(f"{GREEN}SuperWhisper Custom est prêt !{RESET}\n")
    print(f"  Raccourci global : {BOLD}Ctrl + Alt + Espace{RESET}")
    print(f"    - 1er appui  : démarre l'enregistrement")
    print(f"    - 2e appui   : arrête et transcrit")
    print(f"    - Le texte est automatiquement copié dans le presse-papier\n")

    if IS_WINDOWS:
        print(f"  Lancer maintenant : {BOLD}superwhisper.bat{RESET}")
        print(f"  Ou double-cliquer : {BOLD}superwhisper.vbs{RESET} (sans console)\n")
    else:
        print(f"  Lancer maintenant : {BOLD}./superwhisper.sh{RESET}\n")

    print(f"  L'application démarrera automatiquement au prochain boot.")
    print(f"  Clic droit sur l'icône du tray pour accéder aux paramètres.\n")


def main():
    parser = argparse.ArgumentParser(description="Installer SuperWhisper Custom")
    parser.add_argument("--model", default="large-v3",
                        choices=["large-v3", "large-v3-turbo", "distil-large-v3",
                                 "medium", "small", "base", "tiny"],
                        help="Modèle Whisper à télécharger (défaut: large-v3)")
    parser.add_argument("--no-autostart", action="store_true",
                        help="Ne pas configurer le démarrage automatique")
    parser.add_argument("--no-model", action="store_true",
                        help="Ne pas télécharger le modèle maintenant")
    args = parser.parse_args()

    print(f"\n{BOLD}{GREEN}SuperWhisper Custom — Installation{RESET}")
    print(f"  Système : {platform.system()} {platform.release()}")
    print(f"  Python  : {sys.version.split()[0]}")
    print(f"  Dossier : {SCRIPT_DIR}\n")

    has_gpu = check_prerequisites()
    create_venv()
    install_dependencies(has_gpu)

    if not args.no_model:
        download_model(args.model)

    create_launcher()

    if not args.no_autostart:
        setup_autostart()

    print_summary()


if __name__ == "__main__":
    main()
