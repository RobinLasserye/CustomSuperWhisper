"""Catalogue de modèles mesurés, recommandations selon la VRAM, et téléchargements.

Tous les chiffres viennent des benchmarks de `docs/BENCHMARKS.md`, mesurés sur une RTX 5090
(CUDA 12, ctranslate2 4.7.1, Ollama 0.20.7). Ils servent à deux choses : afficher un rapport
performance/qualité dans la fenêtre de réglages, et recommander une configuration qui tient dans
la VRAM de la carte choisie.

Conventions :
- `vram_mib` — VRAM du processus mesurée après chargement et un premier passage (contexte CUDA
  inclus), donc bien « ce que l'application consomme », pas la taille du fichier de poids.
- `rtf` — Real Time Factor : secondes de calcul par seconde d'audio. 0,02 = 50× temps réel.
- `wer` / `wer_biased` — taux d'erreur mot sur 30 clips synthétisés en français, sans puis avec le
  biais vocabulaire. Les valeurs absolues sont pessimistes (voix synthétique, hors distribution) ;
  c'est le classement entre modèles et l'écart entre les deux colonnes qui sont significatifs.
- `claude_hits` — clips (sur 9) où « Claude » est correctement transcrit, sans / avec biais.
"""

WHISPER_MODELS = {
    "tiny": {"label": "tiny", "multilingual": True},
    "base": {"label": "base", "multilingual": True},
    "small": {"label": "small", "multilingual": True},
    "medium": {"label": "medium", "multilingual": True},
    "large-v3": {"label": "large-v3", "multilingual": True},
    "large-v3-turbo": {"label": "large-v3-turbo", "multilingual": True},
    "distil-large-v3": {"label": "distil-large-v3 (anglais uniquement)", "multilingual": False},
}

# (modèle, précision) → mesures
WHISPER_PROFILES = {
    ("tiny", "float16"): {"vram_mib": 672, "load_s": 0.39, "rtf": 0.031,
                          "wer": 0.965, "wer_biased": 0.770, "claude_hits": (1, 9)},
    ("base", "float16"): {"vram_mib": 768, "load_s": 0.37, "rtf": 0.043,
                          "wer": 0.867, "wer_biased": 0.714, "claude_hits": (1, 8)},
    ("small", "int8"): {"vram_mib": 864, "load_s": 0.68, "rtf": 0.038,
                        "wer": 0.573, "wer_biased": 0.509, "claude_hits": (5, 9)},
    ("small", "float16"): {"vram_mib": 1216, "load_s": 0.47, "rtf": 0.029,
                           "wer": 0.646, "wer_biased": 0.537, "claude_hits": (5, 9)},
    ("distil-large-v3", "int8"): {"vram_mib": 1536, "load_s": 1.27, "rtf": 0.016,
                                  "wer": 0.927, "wer_biased": 0.913, "claude_hits": (3, 5)},
    ("large-v3-turbo", "int8"): {"vram_mib": 1632, "load_s": 1.46, "rtf": 0.018,
                                 "wer": 0.354, "wer_biased": 0.268, "claude_hits": (5, 9)},
    ("large-v3", "int8"): {"vram_mib": 2400, "load_s": 2.29, "rtf": 0.043,
                           "wer": 0.361, "wer_biased": 0.243, "claude_hits": (6, 9)},
    ("medium", "float16"): {"vram_mib": 2464, "load_s": 0.81, "rtf": 0.046,
                            "wer": 0.470, "wer_biased": 0.338, "claude_hits": (6, 9)},
    ("distil-large-v3", "float16"): {"vram_mib": 2496, "load_s": 0.88, "rtf": 0.017,
                                     "wer": 0.928, "wer_biased": 0.917, "claude_hits": (5, 4)},
    ("large-v3-turbo", "float16"): {"vram_mib": 2624, "load_s": 1.00, "rtf": 0.018,
                                    "wer": 0.440, "wer_biased": 0.298, "claude_hits": (5, 8)},
    ("large-v3", "float16"): {"vram_mib": 4288, "load_s": 1.26, "rtf": 0.042,
                              "wer": 0.370, "wer_biased": 0.251, "claude_hits": (6, 9)},
}

# Modèles de reformulation mesurés. `vram_mib` au contexte par défaut (8192), `vram_mib_min` au
# contexte réduit (2048). `latency_s` = médiane sur 24 appels, `latency_p90` = 90e centile.
# `long_s` = dictée de 1664 mots (≈ 4 min 30 de parole).
LLM_MODELS = {
    "qwen3:8b": {
        "label": "qwen3:8b", "size_gb": 5.2, "vram_mib": 6660, "vram_mib_min": 5710,
        "latency_s": 0.60, "latency_p90": 2.42, "long_s": 14.3, "thinking": True,
        "jury": {"fidelite": 4.52, "consigne": 3.73, "langue": 3.10}, "traps": (6.67, 9),
        "note": "Le plus fiable du catalogue : 6,67/9 au test de pièges (3 tirages), meilleure "
                "fidélité au "
                "jugement à l'aveugle (aucune invention de chiffre, de nom ou de raccourci "
                "relevée), et seul à respecter une demande de traduction en japonais. Son seul "
                "échec : « un giga six » perdu à la traduction. Modèle par défaut.",
    },
    "qwen3.5:4b": {
        "label": "qwen3.5:4b", "size_gb": 3.4, "vram_mib": 6350, "vram_mib_min": 5840,
        "latency_s": 0.67, "latency_p90": 1.83, "long_s": 11.9, "thinking": True,
        "jury": {"fidelite": 4.04, "consigne": 3.88, "langue": 4.00}, "traps": (5.33, 9),
        "note": "Meilleur français du panel et meilleur respect des consignes ; conserve "
                "1664/1664 mots et 60/60 chiffres sur une dictée de 4 min 30. Deux faiblesses "
                "mesurées : il réécrit des chiffres dictés à l'oral (« un giga six » → « 6 GB »), "
                "fausse un terme métier (« recette » → « réception »), et il a ignoré une demande "
                "de traduction en japonais — rattrapé par la seconde tentative automatique. "
                "À préférer quand la qualité de la prose française prime sur la restitution "
                "exacte des chiffres.",
    },
    "qwen3.5:2b": {
        "label": "qwen3.5:2b", "size_gb": 1.8, "vram_mib": 4390, "vram_mib_min": 4110,
        "latency_s": 0.53, "latency_p90": 1.70, "long_s": None, "thinking": True,
        "jury": None, "traps": (4.0, 9),
        "note": "À éviter : 4,00/9 au test de pièges, moins fiable que qwen3:1.7b qui est pourtant "
                "deux fois plus léger. Il perd des montants dictés et des identifiants techniques "
                "(« little endian », « SteamVR »), et ne traduit pas en japonais.",
    },
    "qwen3:1.7b": {
        "label": "qwen3:1.7b", "size_gb": 1.4, "vram_mib": 2740, "vram_mib_min": 1990,
        "latency_s": 0.23, "latency_p90": 0.89, "long_s": None, "thinking": True,
        "jury": None, "traps": (6.0, 9),
        "note": "Le meilleur rapport fiabilité/VRAM du catalogue : 6,00/9 au test de pièges pour "
                "2,0 Go seulement et 0,23 s de latence — juste derrière qwen3:8b, et devant tous "
                "les modèles plus gros. Il échoue sur les identifiants techniques denses et ne "
                "traduit pas en japonais. Le choix pour toute carte de moins de 12 Go.",
    },
    "gemma3:4b-it-qat": {
        "label": "gemma3:4b-it-qat", "size_gb": 4.0, "vram_mib": 6140, "vram_mib_min": 6040,
        "latency_s": 0.48, "latency_p90": 1.91, "long_s": 5.8, "thinking": False,
        "jury": {"fidelite": 2.82, "consigne": 3.59, "langue": 3.85}, "traps": (4.33, 9),
        "note": "Français propre et seul japonais vraiment idiomatique, mais il invente : "
                "raccourci « Ctrl+Alt+E » jamais dicté, « int8 » devenu « with 8GB », numéro de "
                "facture fabriqué. À éviter si le message part sans relecture.",
    },
    "granite4:micro": {
        "label": "granite4:micro", "size_gb": 2.1, "vram_mib": 3380, "vram_mib_min": 2460,
        "latency_s": 0.42, "latency_p90": 0.95, "long_s": 5.3, "thinking": False,
        "jury": {"fidelite": 1.61, "consigne": 2.21, "langue": 2.10}, "traps": (2.0, 9),
        "note": "DÉCONSEILLÉ malgré son poids : il produit des faux. « je pense que je vais pas "
                "pouvoir venir » devient « j'ai pas vraiment envie de venir », « jeudi » devient "
                "« le jeûne », un budget passe du lot deux à tout le projet. Le plus léger, mais "
                "inutilisable pour un message qu'on envoie.",
    },
}

# Modèle écarté, gardé documenté pour ne pas refaire l'erreur.
LLM_REJECTED = {
    "qwen3:4b": "Son raisonnement fuit dans la réponse malgré `think:false` : 9 à 28 s par appel "
                "et le texte livré commence par « Okay, let's tackle this… ».",
}

DEFAULT_LLM_MODEL = "qwen3:8b"

# Paliers de recommandation. `whisper` = (modèle, précision).
VRAM_TIERS = [
    {"max_gb": 4, "whisper": ("large-v3-turbo", "int8"), "llm": "qwen3:1.7b", "num_ctx": 2048,
     "strategy": "alternance",
     "comment": "1,6 Go pour Whisper et 1,9 Go pour le modèle de reformulation : sur 4 Go il faut "
                "décharger Whisper entre deux dictées (réglage « Décharger Whisper après »). "
                "Évitez les dictées de plus de deux minutes, ce modèle perd des chiffres au-delà."},
    {"max_gb": 6, "whisper": ("large-v3-turbo", "int8"), "llm": "qwen3:1.7b", "num_ctx": 8192,
     "strategy": "cohabitation",
     "comment": "1,6 + 2,7 Go : les deux modèles tiennent ensemble. turbo en int8 transcrit 56× "
                "plus vite que le temps réel pour un WER de 0,27, presque celui de large-v3."},
    {"max_gb": 8, "whisper": ("large-v3", "int8"), "llm": "qwen3:1.7b", "num_ctx": 8192,
     "strategy": "cohabitation",
     "comment": "2,4 + 2,7 Go : on peut se payer le meilleur modèle de transcription du catalogue "
                "(WER 0,243) tout en gardant 3 Go de marge. Mesuré : qwen3:1.7b est plus fiable "
                "que qwen3.5:2b, pourtant deux fois plus lourd."},
    {"max_gb": 12, "whisper": ("large-v3", "int8"), "llm": "qwen3:8b", "num_ctx": 8192,
     "strategy": "cohabitation",
     "comment": "large-v3 en int8 atteint le meilleur WER mesuré (0,243 avec biais) pour 2,4 Go, "
                "et qwen3:8b est le modèle le plus fidèle du catalogue."},
    {"max_gb": None, "whisper": ("large-v3", "int8"), "llm": "qwen3:8b", "num_ctx": 8192,
     "strategy": "cohabitation",
     "comment": "large-v3 en int8 obtient le meilleur WER mesuré du catalogue (0,243) pour 2,4 Go, "
                "donc rien ne justifie float16 (0,251 pour 4,3 Go) même quand la VRAM est là. "
                "float16 reste sélectionnable dans l'onglet Transcription."},
]


def whisper_profile(model, compute_type):
    return WHISPER_PROFILES.get((model, compute_type))


def quality_stars(wer):
    """Traduit un WER mesuré en repère lisible (comparatif entre modèles, pas absolu)."""
    if wer is None:
        return "?"
    for bound, stars in ((0.28, "★★★★★"), (0.34, "★★★★"), (0.45, "★★★"), (0.60, "★★")):
        if wer <= bound:
            return stars
    return "★"


def speed_label(rtf):
    if not rtf:
        return "?"
    return f"{1 / rtf:.0f}× temps réel"


def recommend(vram_mib):
    """Palier conseillé pour une carte de `vram_mib` Mio."""
    gb = (vram_mib or 0) / 1024
    for tier in VRAM_TIERS:
        if tier["max_gb"] is None or gb <= tier["max_gb"]:
            return tier
    return VRAM_TIERS[-1]


def total_vram_mib(whisper_model, compute_type, llm_model, num_ctx=8192,
                   strategy="cohabitation"):
    """VRAM nécessaire pour la combinaison.

    En « cohabitation » les deux modèles restent chargés : les besoins s'additionnent. En
    « alternance » Whisper est déchargé entre deux dictées, donc seul le plus gourmand des deux
    doit tenir à un instant donné.
    """
    profile = whisper_profile(whisper_model, compute_type)
    whisper_mib = profile["vram_mib"] if profile else 0
    llm = LLM_MODELS.get(llm_model, {})
    llm_mib = llm.get("vram_mib_min" if num_ctx <= 2048 else "vram_mib", 0)
    if strategy == "alternance":
        return max(whisper_mib, llm_mib)
    return whisper_mib + llm_mib


def fits(vram_mib, whisper_model, compute_type, llm_model, num_ctx=8192, headroom_mib=1024,
         strategy="cohabitation"):
    """La combinaison tient-elle dans la carte, en gardant de la marge pour le bureau ?"""
    needed = total_vram_mib(whisper_model, compute_type, llm_model, num_ctx, strategy)
    return needed + headroom_mib <= vram_mib


# ─── Téléchargements ──────────────────────────────────────────────────────────

def whisper_is_downloaded(model):
    """Le modèle est-il déjà dans le cache Hugging Face ?"""
    try:
        from faster_whisper.utils import download_model
        download_model(model, local_files_only=True)
        return True
    except Exception:
        return False


def download_whisper(model):
    """Télécharge un modèle Whisper. Bloquant : à appeler dans un thread."""
    from faster_whisper.utils import download_model
    return download_model(model)


def ollama_installed_models(host, timeout=5):
    """Modèles présents localement dans Ollama. Le délai est court quand l'appel vient de
    l'interface : la fenêtre ne doit pas geler si l'hôte configuré ne répond pas."""
    import json
    import urllib.request
    with urllib.request.urlopen(host.rstrip("/") + "/api/tags", timeout=timeout) as response:
        data = json.loads(response.read().decode())
    return [entry["name"] for entry in data.get("models", [])]


def pull_ollama_model(host, model, progress=None):
    """Télécharge un modèle Ollama en suivant la progression.

    `progress` est appelé avec (statut, octets_reçus, octets_totaux). Bloquant. Lève si le flux
    s'interrompt avant l'événement final.
    """
    import json
    import urllib.error
    import urllib.request

    request = urllib.request.Request(
        host.rstrip("/") + "/api/pull",
        data=json.dumps({"model": model, "stream": True}).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    finished = False
    try:
        with urllib.request.urlopen(request, timeout=3600) as response:
            for raw_line in response:
                line = raw_line.decode().strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                if event.get("error"):
                    raise RuntimeError(event["error"])
                status = event.get("status", "")
                if status == "success":
                    finished = True
                if progress:
                    progress(status, event.get("completed"), event.get("total"))
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode()[:200]
        except Exception:
            pass
        raise RuntimeError(f"Ollama a répondu {exc.code}{' : ' + detail if detail else ''}") from exc

    if not finished:
        # Le flux s'est arrêté sans l'événement « success » : le modèle n'est pas complet, et
        # annoncer une réussite ferait croire à un téléchargement terminé.
        raise RuntimeError("téléchargement interrompu avant la fin")
    return True
