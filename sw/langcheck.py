"""Vérification sommaire de la langue de sortie.

Motivation mesurée : au benchmark, un modèle a rendu un ticket **entièrement en français** alors
que le japonais était demandé, sans le signaler. Le contrôle ci-dessous sert à détecter ce cas et
à déclencher une seconde tentative, pas à faire de la détection de langue fine.

`looks_like` est volontairement asymétrique : elle ne renvoie `False` que si elle est sûre, et
`None` quand elle ne sait pas. Un doute ne doit jamais bloquer une dictée.
"""

# Plages Unicode caractéristiques d'une écriture non latine
SCRIPT_RANGES = {
    "ja": [(0x3040, 0x30FF), (0x4E00, 0x9FFF), (0xFF66, 0xFF9D)],
    "zh": [(0x4E00, 0x9FFF), (0x3400, 0x4DBF)],
    "ko": [(0xAC00, 0xD7AF), (0x1100, 0x11FF), (0x3130, 0x318F)],
    "ru": [(0x0400, 0x04FF)],
    "ar": [(0x0600, 0x06FF), (0x0750, 0x077F)],
}

MIN_SCRIPT_RATIO = 0.15

# Mots très fréquents, utilisés seulement pour comparer une langue latine au français.
STOPWORDS = {
    "fr": {"le", "la", "les", "des", "une", "est", "et", "que", "pour", "dans", "avec", "vous",
           "nous", "sur", "pas", "plus", "mais", "être", "cette", "aux"},
    "en": {"the", "and", "is", "are", "to", "of", "for", "with", "you", "that", "this", "in",
           "on", "it", "be", "not", "have", "will", "would", "please"},
    "es": {"el", "la", "los", "las", "de", "que", "y", "para", "con", "una", "es", "por", "en",
           "no", "se", "su", "más", "pero", "como", "puedo"},
    "de": {"der", "die", "das", "und", "ist", "für", "mit", "nicht", "ich", "sie", "ein", "eine",
           "zu", "auf", "von", "im", "dem", "den", "aber", "auch"},
    "it": {"il", "la", "le", "di", "che", "e", "per", "con", "una", "è", "non", "in", "sono",
           "questo", "ma", "come", "anche", "del", "alla", "più"},
    "pt": {"o", "a", "os", "as", "de", "que", "e", "para", "com", "uma", "é", "não", "em", "do",
           "da", "mas", "como", "mais", "seu", "por"},
    "nl": {"de", "het", "een", "en", "is", "van", "voor", "met", "niet", "dat", "op", "te", "ik",
           "je", "we", "maar", "ook", "aan", "er", "zijn"},
    "pl": {"i", "w", "nie", "na", "że", "to", "jest", "do", "się", "z", "o", "ale", "jak",
           "dla", "od", "po", "przez", "bardzo", "tak", "co"},
    "tr": {"ve", "bir", "bu", "için", "ile", "değil", "olarak", "daha", "gibi", "ama", "veya",
           "da", "de", "en", "çok", "her", "ne", "olan", "kadar", "sonra"},
}


def script_ratio(text, code):
    """Part de caractères appartenant à l'écriture attendue, parmi les caractères non espaces."""
    ranges = SCRIPT_RANGES.get(code)
    if not ranges:
        return None
    total = sum(1 for char in text if not char.isspace())
    if not total:
        return 0.0
    hits = 0
    for char in text:
        point = ord(char)
        if any(start <= point <= end for start, end in ranges):
            hits += 1
    return hits / total


def _words(text):
    return [w.strip(".,;:!?…()[]«»\"'`*_-").lower() for w in text.split()]


def stopword_hits(text, code):
    words = set(_words(text))
    reference = STOPWORDS.get(code)
    if not reference:
        return None
    return len(words & reference)


def looks_like(text, code):
    """True / False / None (indéterminé) : le texte est-il bien dans la langue `code` ?"""
    if not text or not text.strip() or not code or code == "none":
        return None

    ratio = script_ratio(text, code)
    if ratio is not None:
        return ratio >= MIN_SCRIPT_RATIO

    if code not in STOPWORDS:
        return None

    words = _words(text)
    if len(words) < 8:
        return None                       # trop court pour conclure

    target = stopword_hits(text, code) or 0
    french = stopword_hits(text, "fr") or 0
    if code == "fr":
        return french >= 1

    if target == 0 and french >= 2:
        return False                      # visiblement resté en français
    if target == 0 and french == 0:
        return None                       # langue inconnue, on ne juge pas
    return target >= french
