"""Filtrage des hallucinations de Whisper.

Whisper a été entraîné sur des sous-titres YouTube : sur du silence, du bruit de fond ou une fin
de phrase coupée, il produit spontanément des crédits de sous-titrage ou des remerciements de
chaîne, qui n'ont jamais été prononcés. Les motifs français les plus fréquents sont documentés,
notamment « Sous-titrage ST' 501 » et la famille « Sous-titres réalisés par la communauté
d'Amara.org » (cf. la liste de scraibe/hallucinations.py sur Hugging Face).

Deux niveaux :

- **Motifs certains** — des phrases qu'on ne dicte jamais. Retirées inconditionnellement, et
  seulement elles : le reste du segment est conservé.
- **Motifs ambigus** — « merci », « au revoir »… qui peuvent être réellement dictés. Retirés
  uniquement si les métriques de Whisper trahissent le segment (`no_speech_prob` élevé ou
  `avg_logprob` très bas).

Le module travaille sur une copie normalisée du texte (minuscules, accents retirés, ponctuation
réduite à des espaces) tout en gardant la correspondance des positions, pour pouvoir découper le
texte **original** avec précision.
"""

import re
import unicodedata

# Motifs certains, écrits sous forme normalisée (minuscules, sans accent, sans ponctuation).
DEFAULT_ARTIFACT_PATTERNS = [
    "sous titrage st 501",
    "sous titrage st501",
    "sous titrage societe radio canada",
    "sous titrage mfp",
    "sous titrage fr",
    "sous titres realises par la communaute d amara org",
    "sous titres realises para la communaute d amara org",
    "sous titres realises pour la communaute d amara org",
    "sous titres realises par la communaute de l amara org",
    "sous titres faits par la communaute d amara org",
    "sous titres fait par la communaute d amara org",
    "sous titres par la communaute d amara org",
    "sous titres par amara org",
    "sous titres par l amara org",
    "sous titres fait par sous titres par amara org",
    "sous titres realises par les soustitres d amara org",
    "cliquez vous sur les sous titres et abonnez vous a la chaine d amara org",
    "amara org",
    "par soustitreur com",
    "soustitreur com",
    "merci d avoir regarde cette video",
    "merci d avoir regarde",
    "merci a tous d avoir regarde cette video",
    "merci beaucoup d avoir regarde cette video",
    "n hesitez pas a vous abonner",
    "abonnez vous a la chaine",
    "abonnez vous",
    "a la prochaine video",
    "thanks for watching",
    "thank you for watching",
    "subtitles by the amara org community",
    "please subscribe",
    "www mooji org",
]

# Motifs plausibles en dictée réelle : retirés seulement si les métriques les trahissent.
DEFAULT_AMBIGUOUS_PATTERNS = [
    "merci", "merci beaucoup", "merci a tous", "au revoir", "a bientot",
    "a la prochaine", "voila", "sous titres", "bye bye",
]

DEFAULT_NO_SPEECH_THRESHOLD = 0.6
DEFAULT_LOGPROB_THRESHOLD = -1.0

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?…])\s+")


# Marque de fin de phrase dans la forme normalisée : un motif ne doit pas pouvoir enjamber un
# point, sinon « … merci. D'avoir regardé cette vidéo … » serait pris pour l'artefact
# « merci d avoir regarde cette video ».
_BOUNDARY = "\x00"
_SENTENCE_ENDERS = ".!?…"


def _normalize_with_map(text):
    """Retourne (texte normalisé, index d'origine de chaque caractère normalisé)."""
    chars, mapping = [], []
    for index, char in enumerate(text):
        decomposed = unicodedata.normalize("NFD", char)
        base = "".join(c for c in decomposed if unicodedata.category(c) != "Mn").lower()
        if not base:
            continue
        if base.isalnum():
            for piece in base:
                chars.append(piece)
                mapping.append(index)
            continue
        # Un point suivi d'une espace (ou en fin de texte) termine une phrase ; un point collé
        # comme dans « amara.org » n'est qu'un séparateur.
        following = text[index + 1] if index + 1 < len(text) else ""
        ends_sentence = char in _SENTENCE_ENDERS and (following == "" or following.isspace())
        marker = _BOUNDARY if ends_sentence else " "
        if chars and chars[-1] not in (" ", _BOUNDARY):
            chars.append(marker)
            mapping.append(index)
        elif chars and marker == _BOUNDARY and chars[-1] == " ":
            chars[-1] = _BOUNDARY
    return "".join(chars), mapping


def normalize(text):
    """Forme normalisée utilisée pour comparer un texte à un motif."""
    return _normalize_with_map(text)[0].replace(_BOUNDARY, " ").strip()


def _pattern_regex(pattern):
    """Motif normalisé → regex tolérant plusieurs espaces, mais pas une fin de phrase."""
    words = [re.escape(w) for w in pattern.split()]
    if not words:
        return None
    return re.compile(r"(?<![a-z0-9])" + r"[ ]+".join(words) + r"(?![a-z0-9])")


def find_artifacts(text, patterns):
    """Occurrences des motifs dans le texte, en positions du texte **original**.

    Retourne une liste de (début, fin, motif) triée, sans chevauchement.
    """
    normalized, mapping = _normalize_with_map(text)
    found = []
    for pattern in patterns or []:
        regex = _pattern_regex(normalize(pattern))
        if regex is None:
            continue
        for match in regex.finditer(normalized):
            start, end = match.span()
            if start >= len(mapping) or end - 1 >= len(mapping):
                continue
            stop = mapping[end - 1] + 1
            while stop < len(text) and unicodedata.category(text[stop]) == "Mn":
                stop += 1
            found.append((mapping[start], stop, pattern))
    if not found:
        return []
    # Garder les occurrences les plus longues et supprimer les chevauchements
    found.sort(key=lambda item: (item[0], -(item[1] - item[0])))
    result = []
    for start, end, pattern in found:
        if result and start < result[-1][1]:
            # Chevauchement : on fusionne les deux intervalles au lieu de garder le plus long,
            # sinon le début du premier artefact resterait dans le texte.
            previous_start, previous_end, previous_pattern = result[-1]
            longest = pattern if (end - start) > (previous_end - previous_start) \
                else previous_pattern
            result[-1] = (min(previous_start, start), max(previous_end, end), longest)
            continue
        result.append((start, end, pattern))
    return result


_ORPHAN_PUNCTUATION = " \t.,;:!?…-–—"


def _join_around_cut(left, right):
    """Recolle deux morceaux séparés par un artefact retiré.

    Le nettoyage est **local** : nettoyer la ponctuation sur tout le texte casserait la
    typographie française du contenu conservé (l'espace avant « ! » ou « : » doit rester).
    """
    left = left.rstrip()
    right = right.lstrip()
    if not left or left[-1] in _SENTENCE_ENDERS:
        # La phrase précédente est finie (ou il n'y a rien avant) : la ponctuation qui ouvrait le
        # morceau suivant appartenait à l'artefact.
        right = right.lstrip(_ORPHAN_PUNCTUATION)
    if not left:
        return right
    if not right:
        return left
    return f"{left} {right}"


def strip_artifacts(text, patterns):
    """Retire les motifs du texte. Retourne (texte nettoyé, motifs retirés)."""
    occurrences = find_artifacts(text, patterns)
    if not occurrences:
        return text, []
    pieces, last, removed = [], 0, []
    for start, end, _pattern in occurrences:
        pieces.append(text[last:start])
        removed.append(text[start:end].strip())
        last = end
    pieces.append(text[last:])

    cleaned = pieces[0]
    for piece in pieces[1:]:
        cleaned = _join_around_cut(cleaned, piece)
    return cleaned.strip(), removed


def is_only_punctuation(text):
    return not any(c.isalnum() for c in text)


def segment_is_suspect(no_speech_prob, avg_logprob,
                       no_speech_threshold=DEFAULT_NO_SPEECH_THRESHOLD,
                       logprob_threshold=DEFAULT_LOGPROB_THRESHOLD):
    """Les métriques de Whisper trahissent-elles un segment halluciné ?"""
    if no_speech_prob is not None and no_speech_prob >= no_speech_threshold:
        return True
    if avg_logprob is not None and avg_logprob <= logprob_threshold:
        return True
    return False


def clean_segment(text, no_speech_prob=None, avg_logprob=None, patterns=None,
                  ambiguous_patterns=None, ambiguous_enabled=True,
                  no_speech_threshold=DEFAULT_NO_SPEECH_THRESHOLD,
                  logprob_threshold=DEFAULT_LOGPROB_THRESHOLD):
    """Nettoie un segment. Retourne (texte, motifs retirés)."""
    if patterns is None:
        patterns = DEFAULT_ARTIFACT_PATTERNS
    cleaned, removed = strip_artifacts(text, patterns)

    if ambiguous_enabled and cleaned.strip():
        candidates = (ambiguous_patterns if ambiguous_patterns is not None
                      else DEFAULT_AMBIGUOUS_PATTERNS)
        suspect = segment_is_suspect(no_speech_prob, avg_logprob,
                                     no_speech_threshold, logprob_threshold)
        if suspect and normalize(cleaned) in {normalize(p) for p in candidates}:
            removed.append(cleaned.strip())
            cleaned = ""
    if is_only_punctuation(cleaned):
        cleaned = ""
    return cleaned.strip(), removed


def collapse_repetitions(text, max_repeats=2):
    """Effondre les boucles de Whisper : une phrase répétée à l'identique plus de `max_repeats`
    fois d'affilée n'est gardée que `max_repeats` fois."""
    if not text:
        return text
    sentences = _SENTENCE_SPLIT.split(text.strip())
    kept, streak, previous = [], 0, None
    for sentence in sentences:
        key = normalize(sentence)
        if key and key == previous:
            streak += 1
            if streak > max_repeats:
                continue
        else:
            streak = 1
            previous = key
        kept.append(sentence)
    return " ".join(kept)


def filter_transcription(segments, config=None):
    """Applique le filtre à une liste de segments Whisper.

    `segments` : itérable d'objets ou de dictionnaires exposant `text`, `no_speech_prob` et
    `avg_logprob`. Retourne (texte assemblé, liste des artefacts retirés).
    """
    config = config or {}
    if not config.get("artifact_filter", True):
        texts = []
        for segment in segments:
            texts.append(_get(segment, "text", "").strip())
        return " ".join(t for t in texts if t), []

    patterns = config.get("artifact_patterns") or DEFAULT_ARTIFACT_PATTERNS
    ambiguous = config.get("artifact_ambiguous") or DEFAULT_AMBIGUOUS_PATTERNS
    ambiguous_enabled = config.get("artifact_ambiguous_enabled", True)
    no_speech_threshold = config.get("artifact_no_speech_threshold",
                                     DEFAULT_NO_SPEECH_THRESHOLD)
    logprob_threshold = config.get("artifact_logprob_threshold", DEFAULT_LOGPROB_THRESHOLD)

    kept, removed_all = [], []
    for segment in segments:
        text = _get(segment, "text", "") or ""
        cleaned, removed = clean_segment(
            text.strip(),
            no_speech_prob=_get(segment, "no_speech_prob", None),
            avg_logprob=_get(segment, "avg_logprob", None),
            patterns=patterns, ambiguous_patterns=ambiguous,
            ambiguous_enabled=ambiguous_enabled,
            no_speech_threshold=no_speech_threshold,
            logprob_threshold=logprob_threshold)
        removed_all.extend(removed)
        if cleaned:
            kept.append(cleaned)

    text = " ".join(kept)
    if config.get("collapse_repetitions", True):
        text = collapse_repetitions(text)
    return text.strip(), removed_all


def _get(segment, name, default):
    if isinstance(segment, dict):
        return segment.get(name, default)
    return getattr(segment, name, default)
