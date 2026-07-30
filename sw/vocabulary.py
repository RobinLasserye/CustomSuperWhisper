"""Biais vocabulaire côté Whisper et corrections appliquées après transcription.

Trois couches indépendantes, dans cet ordre :

1. `build_hotwords` / `build_initial_prompt` — biaisent le décodage de Whisper vers un vocabulaire
   fourni. Les deux mécanismes se cumulent (`faster_whisper/transcribe.py`, `get_prompt`) : les
   hotwords et le contexte issu de `initial_prompt` atterrissent dans le même bloc `sot_prev`.
2. `apply_corrections` — un dictionnaire ordonné de règles littérales ou regex.
3. `apply_cloud_rule` — la règle « cloud → Claude », neutralisée sur les zones couvertes par une
   liste d'exceptions (les vrais usages d'hébergement).

Aucune dépendance en dehors de la bibliothèque standard : ce module est testable seul.
"""

import re

# Vocabulaire injecté dans le décodage. Mesuré au benchmark : « Claude » passe de 6/9 à 9/9
# clips correctement transcrits sur large-v3, et le WER global baisse (0,37 → 0,25).
DEFAULT_VOCABULARY = [
    "Claude", "Claude Code", "Claude Desktop", "Anthropic", "Opus", "Sonnet", "Haiku",
    "Ollama", "Fedora", "KDE Plasma", "PipeWire", "Wayland", "systemd", "GitHub", "GitLab",
    "WhatsApp", "Messenger", "Slack", "Python", "Docker", "SuperWhisper", "faster-whisper",
    "PySide", "Qt", "Whisper", "VS Code", "npm", "CUDA", "NVIDIA",
]

# Règles appliquées dans l'ordre : les règles contextuelles doivent précéder les règles nues.
DEFAULT_CORRECTIONS = [
    {"from": "cloud code", "to": "Claude Code", "regex": False, "enabled": True},
    {"from": "cloud desktop", "to": "Claude Desktop", "regex": False, "enabled": True},
    {"from": "cloud opus", "to": "Claude Opus", "regex": False, "enabled": True},
    {"from": "cloud sonnet", "to": "Claude Sonnet", "regex": False, "enabled": True},
    {"from": "clode", "to": "Claude", "regex": False, "enabled": True},
    {"from": "cloude", "to": "Claude", "regex": False, "enabled": True},
    {"from": "claud", "to": "Claude", "regex": False, "enabled": True},
    {"from": "anthropique", "to": "Anthropic", "regex": False, "enabled": True},
    {"from": "olama", "to": "Ollama", "regex": False, "enabled": True},
    {"from": "ollamma", "to": "Ollama", "regex": False, "enabled": True},
    {"from": "git hub", "to": "GitHub", "regex": False, "enabled": True},
    {"from": "pipe wire", "to": "PipeWire", "regex": False, "enabled": True},
    {"from": "système d", "to": "systemd", "regex": False, "enabled": True},
    {"from": "py side", "to": "PySide", "regex": False, "enabled": True},
    {"from": "watsapp", "to": "WhatsApp", "regex": False, "enabled": True},
    {"from": "whatsap", "to": "WhatsApp", "regex": False, "enabled": True},
    {"from": "super whisper", "to": "SuperWhisper", "regex": False, "enabled": True},
    {"from": "vs codes", "to": "VS Code", "regex": False, "enabled": True},
]

# Zones où « cloud » veut vraiment dire « cloud ». Évaluées après les règles contextuelles :
# « cloud code » est déjà devenu « Claude Code » quand « le cloud » est testé.
DEFAULT_CLOUD_EXCEPTIONS = [
    "dans le cloud", "sur le cloud", "vers le cloud", "au cloud", "du cloud", "le cloud",
    "un cloud", "cloud public", "cloud privé", "cloud hybride", "cloud gaming",
    "cloud computing", "cloud souverain", "cloud AWS", "cloud Azure", "cloud GCP",
    "cloud Google", "cloud OVH", "cloud Scaleway", "hébergement cloud", "stockage cloud",
    "serveur cloud", "solution cloud", "offre cloud",
]

CLOUD_PATTERN = re.compile(r"(?<!\w)cloud(?!\w)", re.IGNORECASE)


# ─── Biais Whisper ────────────────────────────────────────────────────────────

def build_hotwords(vocabulary):
    """Chaîne passée à `hotwords=` de faster-whisper, ou None si le vocabulaire est vide."""
    terms = [t.strip() for t in (vocabulary or []) if t and t.strip()]
    return ", ".join(terms) if terms else None


def build_initial_prompt(vocabulary):
    """Contexte passé à `initial_prompt=`. Volontairement court : un prompt long augmente le
    risque que Whisper le recrache dans la transcription."""
    terms = [t.strip() for t in (vocabulary or []) if t and t.strip()]
    if not terms:
        return None
    return "Transcription d'un développeur français. Vocabulaire : " + ", ".join(terms) + "."


# ─── Moteur de corrections ────────────────────────────────────────────────────

def literal_pattern(text):
    """Motif regex pour une expression littérale : insensible à la casse, espaces souples,
    et bornée par des gardes de mot (`\\b` échouerait sur un motif finissant par une apostrophe
    ou un point)."""
    words = [re.escape(w) for w in text.split()]
    body = r"\s+".join(words) if words else re.escape(text)
    return rf"(?<!\w){body}(?!\w)"


def compile_rule(rule):
    """Compile une règle en (pattern compilé, fonction de remplacement)."""
    source = rule["from"] if not rule.get("regex") else rule["from"]
    if rule.get("regex"):
        compiled = re.compile(source, re.IGNORECASE)
        replacement = rule["to"]          # les backréférences sont voulues en mode regex
        return compiled, replacement
    compiled = re.compile(literal_pattern(source), re.IGNORECASE)
    literal = rule["to"]
    return compiled, (lambda _match, value=literal: value)


def apply_corrections(text, corrections):
    """Applique les règles dans l'ordre. Les règles désactivées sont ignorées."""
    if not text:
        return text
    for rule in corrections or []:
        if not rule.get("enabled", True) or not rule.get("from"):
            continue
        try:
            compiled, replacement = compile_rule(rule)
        except re.error:
            continue                      # une regex utilisateur invalide ne casse pas la dictée
        text = compiled.sub(replacement, text)
    return text


def exception_spans(text, exceptions):
    """Intervalles (début, fin) du texte couverts par une expression d'exception."""
    spans = []
    for exception in exceptions or []:
        if not exception.strip():
            continue
        for match in re.finditer(literal_pattern(exception), text, re.IGNORECASE):
            spans.append(match.span())
    return spans


def apply_cloud_rule(text, exceptions=None):
    """Remplace « cloud » par « Claude » sauf dans les zones d'exception."""
    if not text:
        return text
    spans = exception_spans(text, exceptions if exceptions is not None
                            else DEFAULT_CLOUD_EXCEPTIONS)

    def replace(match):
        for start, end in spans:
            if match.start() >= start and match.end() <= end:
                return match.group(0)
        return "Claude"

    return CLOUD_PATTERN.sub(replace, text)


def correct(text, config):
    """Chaîne complète de correction, telle que l'application l'utilise."""
    if not config.get("corrections_enabled", True):
        return text
    text = apply_corrections(text, config.get("corrections"))
    if config.get("cloud_rule_enabled", True):
        text = apply_cloud_rule(text, config.get("cloud_exceptions"))
    return text


# ─── Édition sous forme de texte (pour la fenêtre de réglages) ────────────────

def parse_corrections_text(raw):
    """Lit `motif => remplacement` ligne par ligne.

    - `re:motif => remplacement` pour une règle regex
    - une ligne commençant par `#` est une règle désactivée (ou un simple commentaire)
    """
    rules = []
    for line in (raw or "").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        enabled = True
        if stripped.startswith("#"):
            enabled = False
            stripped = stripped.lstrip("#").strip()
            if not stripped:
                continue
        if "=>" not in stripped:
            continue
        source, _, target = stripped.partition("=>")
        source, target = source.strip(), target.strip()
        if not source:
            continue
        is_regex = source.lower().startswith("re:")
        if is_regex:
            source = source[3:].strip()
        rules.append({"from": source, "to": target, "regex": is_regex, "enabled": enabled})
    return rules


def format_corrections_text(rules):
    """Rend les règles éditables sous la forme lue par `parse_corrections_text`."""
    lines = []
    for rule in rules or []:
        prefix = "" if rule.get("enabled", True) else "# "
        source = ("re:" + rule["from"]) if rule.get("regex") else rule["from"]
        lines.append(f"{prefix}{source} => {rule.get('to', '')}")
    return "\n".join(lines)


def parse_list_text(raw):
    """Une entrée par ligne, lignes vides et commentaires ignorés."""
    items = []
    for line in (raw or "").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            items.append(stripped)
    return items
