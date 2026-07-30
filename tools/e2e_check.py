#!/usr/bin/env python3
"""Vérification bout-en-bout : audio → transcription biaisée → corrections → reformulation.

Exige un GPU et Ollama en marche. La phrase de test est synthétisée avec espeak-ng, donc aucun
micro n'est nécessaire.

Usage :
  python tools/e2e_check.py
  python tools/e2e_check.py --model large-v3-turbo --compute int8 --llm qwen3:1.7b
"""

import argparse
import os
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sw.runtime import ensure_cuda_libs                                   # noqa: E402

ensure_cuda_libs(os.path.abspath(__file__))

from sw import artifacts, backends, config as config_module, presets      # noqa: E402
from sw.transcriber import Transcriber                                    # noqa: E402

PHRASES = [
    ("Je vais demander à Claude Code de corriger le bug puis je pousse la branche sur GitHub.",
     ["Claude Code", "GitHub"]),
    ("J'ai lancé Ollama sur Fedora avec PipeWire.", ["Ollama", "Fedora", "PipeWire"]),
]


def synthesize(text):
    path = tempfile.mktemp(suffix=".wav", prefix="sw_e2e_")
    subprocess.run(["espeak-ng", "-v", "fr", "-s", "150", "-w", path, text],
                   check=True, capture_output=True)
    resampled = path + ".16k.wav"
    subprocess.run(["ffmpeg", "-y", "-i", path, "-ar", "16000", "-ac", "1", resampled],
                   check=True, capture_output=True)
    os.unlink(path)
    return resampled


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model")
    parser.add_argument("--compute")
    parser.add_argument("--llm")
    parser.add_argument("--preset", default="message")
    args = parser.parse_args()

    config = config_module.load_config()
    if args.model:
        config["model"] = args.model
    if args.compute:
        config["compute_type"] = args.compute
    if args.llm:
        config["ollama_model"] = args.llm

    print(f"Whisper : {config['model']} / {config['compute_type']}   "
          f"Modèle local : {config['ollama_model']}")
    print(f"Biais vocabulaire : {config['vocab_biasing']}   "
          f"Règle cloud : {config['cloud_rule_enabled']}\n")

    transcriber = Transcriber()
    start = time.monotonic()
    transcriber.load_model(config)
    print(f"[1/4] Modèle chargé en {time.monotonic() - start:.1f} s")

    failures = []

    for phrase, expected in PHRASES:
        wav = synthesize(phrase)
        try:
            start = time.monotonic()
            segments = transcriber.transcribe_segments(wav, config)
            elapsed = time.monotonic() - start
            raw = " ".join(segment.text.strip() for segment in segments)
            text, removed = artifacts.filter_transcription(segments, config)
            from sw import vocabulary
            corrected = vocabulary.correct(text, config)
        finally:
            os.unlink(wav)

        print(f"\n[2/4] Dicté     : {phrase}")
        print(f"      Transcrit  : {raw.strip()}   ({elapsed:.2f} s)")
        if removed:
            print(f"      Artefacts  : {removed}")
        if corrected != text:
            print(f"      Corrigé    : {corrected}")
        for term in expected:
            if term.lower() not in corrected.lower():
                failures.append(f"« {term} » absent de « {corrected} »")
                print(f"      ÉCHEC : « {term} » manquant")
            else:
                print(f"      OK    : « {term} » présent")

    # Reformulation et traduction sur la première phrase
    sample = ("alors euh du coup je voulais te dire que pour demain euh je pense que je vais pas "
              "pouvoir venir on peut décaler à jeudi si ça t'arrange")
    backend = backends.build_backend(config, "ollama")
    if not backend.is_running():
        print("\n[3/4] Ollama éteint — reformulation non testée")
        return 1 if failures else 0

    for label, target in (("Reformulation", "none"), ("Traduction anglaise", "en"),
                          ("Traduction japonaise", "ja")):
        prompt = presets.resolve(config, args.preset, target)
        start = time.monotonic()
        try:
            result, warning = backends.reformat(backend, sample, prompt, target)
        except backends.ReformatError as exc:
            failures.append(f"{label} : {exc}")
            print(f"\n[3/4] {label} : ÉCHEC — {exc}")
            continue
        print(f"\n[3/4] {label} ({time.monotonic() - start:.2f} s)"
              f"{' — ' + warning if warning else ''}")
        print(f"      {result}")
        if warning:
            failures.append(f"{label} : {warning}")

    print("\n[4/4] " + ("Tout est passé" if not failures
                        else f"{len(failures)} problème(s) : " + " | ".join(failures)))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
