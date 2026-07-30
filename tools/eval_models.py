#!/usr/bin/env python3
"""Test de pièges : un modèle local est-il assez fidèle pour reformuler une dictée ?

Chaque cas est une transcription vocale réelle contenant un piège identifié lors du jugement à
l'aveugle du benchmark (voir docs/BENCHMARKS.md) : un chiffre dicté en lettres, un nom propre, un
terme métier, une négation, un raccourci clavier. Les vérifications sont automatiques :

- `requis` : groupes de variantes ; au moins une variante de chaque groupe doit apparaître
- `interdits` : ce qui trahit une invention ou une inversion de sens

Usage :
  python tools/eval_models.py                       # tous les modèles installés dans Ollama
  python tools/eval_models.py qwen3.5:4b granite4:micro
  python tools/eval_models.py --json resultats.json
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
import unicodedata
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sw import langcheck, presets                                       # noqa: E402

HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")

CASES = [
    {
        "id": "modalisation",
        "preset": "message",
        "lang": None,
        "texte": ("alors euh du coup je voulais te dire que en fait pour demain euh je pense que "
                  "je vais pas pouvoir venir parce que j'ai un truc de prévu avec ma sœur euh "
                  "donc voilà si jamais on peut décaler à jeudi ou vendredi ça m'arrangerait "
                  "bien et euh sinon bah tant pis on verra la semaine prochaine"),
        "requis": [["jeudi"], ["vendredi"], ["sœur", "soeur"],
                   ["pense", "crois", "pourrai pas", "peux pas", "ne pourrai", "pas pouvoir"]],
        "interdits": ["envie", "jeûne", "jeune ", "rendez-vous", " euh", "euh ", " bah"],
        "piege": "« je pense que je vais pas pouvoir venir » ne doit pas devenir « j'ai pas envie "
                 "de venir » ; « jeudi » ne doit pas devenir « le jeûne ».",
    },
    {
        "id": "chiffres_facture",
        "preset": "mail",
        "lang": None,
        "texte": ("bonjour euh je vous contacte parce que j'ai reçu la facture du mois de juin et "
                  "euh il y a un problème en fait le montant est de 340 euros alors que le devis "
                  "disait 280 donc euh j'aimerais bien comprendre d'où vient la différence et euh "
                  "savoir si c'est possible de faire un avoir sur les 60 euros merci d'avance"),
        "requis": [["340"], ["280"], ["60"], ["avoir"], ["juin"]],
        "interdits": ["[votre nom]", "[insert", "numéro de facture n°", "[signature]"],
        "piege": "Les trois montants doivent survivre, sans inventer de numéro de facture ni de "
                 "signature.",
    },
    {
        "id": "raccourci",
        "preset": "github",
        "lang": None,
        "texte": ("donc en gros quand je lance la transcription avec le raccourci ctrl alt espace "
                  "euh ça marche bien mais dès que je mets le mode reformulation sur Claude ça "
                  "timeout au bout de 60 secondes et du coup le texte brut est collé mais y a pas "
                  "de message d'erreur clair euh et faudrait aussi que ça marche même quand "
                  "ollama est éteint donc genre un fallback propre avec un warning dans l'overlay"),
        "requis": [["espace", "space", "espacio"], ["60"], ["ollama"], ["overlay"]],
        "interdits": ["ctrl+alt+e ", "ctrl + alt + e ", "ctrl+alt+entrée", "windows 11",
                      "version 1.", "ubuntu"],
        "piege": "Le raccourci dicté est « ctrl alt espace » : inventer « Ctrl+Alt+E » ou une "
                 "version d'OS est éliminatoire.",
    },
    {
        "id": "specs_orales",
        "preset": "message",
        "lang": None,
        "texte": ("euh salut alors pour ta question sur la carte graphique euh franchement si tu "
                  "as que huit giga de VRAM euh je te conseille de prendre le modèle turbo en int "
                  "huit parce que ça tient dans un giga six et euh la qualité est quasi identique "
                  "au large et euh par contre évite distil c'est que de l'anglais ça marche pas "
                  "en français"),
        "requis": [["int huit", "int8", "int 8"],
                   ["un giga six", "1,6", "1.6", "1 giga six"],
                   ["anglais"], ["huit giga", "8 giga", "8 go", "8 gb"], ["distil"]],
        "interdits": ["6 go de vram", "6 gb of vram", "tient dans 6", "fits within 6",
                      "8 go de vram pour le turbo", "n'est pas de l'anglais", "not english",
                      "pas en anglais"],
        "piege": "« int huit » est une quantification et « un giga six » une taille : les "
                 "confondre avec « 8 Go » ou « 6 Go » fabrique une spec fausse. « c'est que de "
                 "l'anglais » ne doit pas être inversé.",
    },
    {
        "id": "responsabilites",
        "preset": "notes",
        "lang": None,
        "texte": ("bon alors euh le point sur la réunion de ce matin donc euh Marie prend la "
                  "partie back-end elle a jusqu'au 15 septembre pour livrer l'API de paiement euh "
                  "moi je m'occupe du front et euh Thomas fait la recette euh par contre on a un "
                  "souci de budget il manque à peu près huit mille euros sur le lot deux donc euh "
                  "il faut qu'on en parle au comité de pilotage vendredi"),
        "requis": [["Marie"], ["Thomas"], ["recette"], ["15 septembre"],
                   ["lot deux", "lot 2"], ["API de paiement", "api de paiement"],
                   ["huit mille", "8 000", "8000"]],
        "interdits": ["réception du projet", "reception du projet", "ensemble du projet",
                      "vous (front)"],
        "piege": "« recette » est une phase de test, pas une réception ; le manque porte sur le "
                 "lot deux, pas sur tout le projet ; le front est pris par l'auteur.",
    },
    {
        "id": "identifiants",
        "preset": "notes",
        "lang": None,
        "texte": ("ok donc pour le projet tapis VR euh j'ai avancé sur la partie BLE le service "
                  "FTMS répond bien sur la caractéristique 2AD9 et euh j'ai validé que 08 02 "
                  "c'est bien le stop et que la vitesse c'est des km heure fois cent en little "
                  "endian par contre euh WiVRn c'est une impasse zéro input routé donc faut "
                  "passer par ALVR plus SteamVR et euh l'install d'ALVR est encore en attente "
                  "dans le dossier téléchargements et aussi faut se rappeler que le tapis se met "
                  "en veille au bout de vingt minutes environ"),
        "requis": [["2AD9", "2ad9"], ["08 02", "0802", "08/02"], ["little endian"],
                   ["ALVR", "alvr"], ["SteamVR", "steamvr"], ["vingt minutes", "20 minutes"],
                   ["input", "entrée"]],
        "interdits": ["0x0802", "entrée routière", "entree routiere", "impassable",
                      "alvr puis steamvr", " euh", "euh "],
        "piege": "« 08 02 » ne doit pas devenir « 0x0802 » (notation inventée), « zéro input "
                 "routé » ne doit pas devenir « aucune entrée routière ».",
    },
    {
        "id": "traduction_japonais",
        "preset": "github",
        "lang": "ja",
        "texte": ("quand je lance la transcription avec le raccourci ctrl alt espace ça marche "
                  "mais la reformulation timeout au bout de 60 secondes et le texte brut est "
                  "collé sans message d'erreur clair"),
        "requis": [["60"]],
        "interdits": [],
        "verifie_langue": "ja",
        "piege": "La sortie doit être réellement en japonais : un modèle a rendu le ticket "
                 "intégralement en français sans le signaler.",
    },
    {
        "id": "traduction_anglais_chiffres",
        "preset": "message",
        "lang": "en",
        "texte": ("si tu as que huit giga de VRAM prends le modèle turbo en int huit parce que ça "
                  "tient dans un giga six et la qualité est quasi identique au large"),
        "requis": [["int8", "int 8", "int eight"], ["1.6", "1,6", "one gigabyte six", "1.6gb"]],
        "interdits": ["fits within 6", "6 gb card", "6gb of vram"],
        "verifie_langue": "en",
        "piege": "Traduire ne doit pas transformer « int huit » en « 8 GB » ni « un giga six » en "
                 "« 6 GB ».",
    },
    {
        "id": "pas_de_bavardage",
        "preset": "compact",
        "lang": None,
        "texte": ("bon alors je te confirme qu'on se voit bien demain à quatorze heures devant le "
                  "cinéma et euh pense à prendre les billets qu'on a réservés la semaine dernière"),
        "requis": [["quatorze heures", "14h", "14 h", "14:00"], ["billets"]],
        "interdits": ["voici", "here is", "```", "message compact :", "texte reformulé",
                      " euh", "euh "],
        "piege": "Aucun préambule, aucune fence markdown, aucun commentaire.",
    },
]


def post(path, payload, timeout=600):
    request = urllib.request.Request(
        HOST.rstrip("/") + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode())


def capabilities(model):
    try:
        return post("/api/show", {"model": model}, timeout=60).get("capabilities", []) or []
    except Exception:
        return []


def installed_models():
    with urllib.request.urlopen(HOST.rstrip("/") + "/api/tags", timeout=10) as response:
        data = json.loads(response.read().decode())
    return [entry["name"] for entry in data.get("models", [])]


def normalize(text):
    decomposed = unicodedata.normalize("NFD", (text or "").lower())
    stripped = "".join(c for c in decomposed if unicodedata.category(c) != "Mn")
    # « 1 025 » / « 1 025 » → « 1025 », pour ne pas pénaliser la typographie française
    stripped = re.sub(r"(?<=\d)[\s  ](?=\d)", "", stripped)
    return re.sub(r"\s+", " ", stripped)


def check(case, output):
    """Retourne (manquants, interdits_trouvés, langue_ok)."""
    haystack = normalize(output)
    missing = []
    for group in case.get("requis", []):
        if not any(normalize(variant) in haystack for variant in group):
            missing.append(" / ".join(group))
    forbidden = [term for term in case.get("interdits", [])
                 if normalize(term) in haystack]
    language_ok = None
    if case.get("verifie_langue"):
        language_ok = langcheck.looks_like(output, case["verifie_langue"])
    return missing, forbidden, language_ok


def run_case(model, case, think):
    config = {"reformat_prompt_overrides": {}, "reformat_custom_modes": []}
    system = presets.resolve(config, case["preset"], case.get("lang") or "none")
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": case["texte"]}],
        "stream": False, "keep_alive": "10m",
        "options": {"temperature": 0.2, "top_p": 0.9, "num_ctx": 8192},
    }
    if think:
        payload["think"] = False
    start = time.monotonic()
    data = post("/api/chat", payload)
    latency = round(time.monotonic() - start, 2)
    from sw.backends import clean_output
    return clean_output((data.get("message") or {}).get("content", "")), latency


def evaluate(model, verbose=True):
    think = "thinking" in capabilities(model)
    results, score, total = [], 0.0, 0
    for case in CASES:
        total += 1
        try:
            output, latency = run_case(model, case, think)
        except Exception as exc:
            results.append({"cas": case["id"], "erreur": str(exc)})
            if verbose:
                print(f"    {case['id']:28s} ERREUR {exc}")
            continue

        missing, forbidden, language_ok = check(case, output)
        passed = not missing and not forbidden and language_ok is not False
        if passed:
            score += 1
        results.append({"cas": case["id"], "latence_s": latency, "reussi": passed,
                        "manquants": missing, "interdits": forbidden,
                        "langue_ok": language_ok, "sortie": output})
        if verbose:
            mark = "OK  " if passed else "ÉCHEC"
            print(f"    {case['id']:28s} {mark} {latency:5.2f}s", end="")
            details = []
            if missing:
                details.append("manque : " + ", ".join(missing))
            if forbidden:
                details.append("interdit : " + ", ".join(forbidden))
            if language_ok is False:
                details.append(f"pas en {case['verifie_langue']}")
            print(("  → " + " | ".join(details)) if details else "")
    return {"model": model, "score": score, "total": total, "cases": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("models", nargs="*", help="modèles Ollama à évaluer")
    parser.add_argument("--json", help="fichier de sortie JSON")
    parser.add_argument("--keep-loaded", action="store_true",
                        help="ne pas décharger le modèle après évaluation")
    args = parser.parse_args()

    models = args.models or installed_models()
    print(f"Évaluation de {len(models)} modèle(s) sur {len(CASES)} cas de pièges\n")

    report = []
    for model in models:
        print(f"=== {model}")
        result = evaluate(model)
        report.append(result)
        print(f"    → {result['score']:.0f}/{result['total']}\n")
        if not args.keep_loaded:
            subprocess.run(["ollama", "stop", model], capture_output=True)

    print("Classement :")
    for result in sorted(report, key=lambda r: -r["score"]):
        latencies = [c["latence_s"] for c in result["cases"] if c.get("latence_s")]
        median = sorted(latencies)[len(latencies) // 2] if latencies else 0
        print(f"  {result['score']:.0f}/{result['total']}  {result['model']:24s} "
              f"latence médiane {median:.2f}s")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
        print(f"\nDétail → {args.json}")


if __name__ == "__main__":
    main()
