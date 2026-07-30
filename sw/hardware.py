"""Inventaire matériel : GPU disponibles, VRAM, entrées audio."""

import subprocess

from .runtime import log


def get_gpu_list():
    """[(index, libellé, vram_mib)] — via nvidia-smi."""
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,memory.total",
             "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return [("0", "GPU 0", None)]

    gpus = []
    for line in output.strip().split("\n"):
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            vram = int(float(parts[2]))
        except ValueError:
            vram = None
        gpus.append((parts[0], parts[1], vram))
    return gpus or [("0", "GPU 0", None)]


def gpu_vram_mib(index):
    for gpu_index, _name, vram in get_gpu_list():
        if gpu_index == str(index):
            return vram
    return None


def gpu_free_mib(index):
    """VRAM libre sur le GPU demandé, ou None."""
    try:
        output = subprocess.check_output(
            ["nvidia-smi", f"--id={index}", "--query-gpu=memory.free",
             "--format=csv,noheader,nounits"], text=True, stderr=subprocess.DEVNULL)
        return int(float(output.strip().split("\n")[0]))
    except Exception:
        return None


def get_audio_inputs():
    """[(identifiant, libellé)] — « default » en tête, car les index numériques ne survivent pas
    à un redémarrage (voir le commentaire dans AudioRecorder.start)."""
    devices = [("default", "Par défaut (recommandé)")]
    try:
        import sounddevice as sd
        for index, device in enumerate(sd.query_devices()):
            if device["max_input_channels"] > 0:
                devices.append((str(index), device["name"]))
    except Exception as exc:
        log(f"énumération audio impossible ({exc})")
    return devices
