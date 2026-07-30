"""Filtre des hallucinations de Whisper (crédits de sous-titres, remerciements YouTube)."""

from sw import artifacts


class FakeSegment:
    """Imite un segment faster-whisper."""

    def __init__(self, text, no_speech_prob=0.1, avg_logprob=-0.3):
        self.text = text
        self.no_speech_prob = no_speech_prob
        self.avg_logprob = avg_logprob


# ─── Normalisation ────────────────────────────────────────────────────────────

def test_normalize_retire_accents_casse_et_ponctuation():
    assert artifacts.normalize("Sous-titrage ST' 501") == "sous titrage st 501"
    assert artifacts.normalize("Réalisés PAR la communauté d'Amara.org") \
        == "realises par la communaute d amara org"


# ─── Motifs certains ──────────────────────────────────────────────────────────

def test_retire_le_credit_de_sous_titrage():
    text, removed = artifacts.strip_artifacts(
        "On se voit demain à quinze heures. Sous-titrage ST' 501",
        artifacts.DEFAULT_ARTIFACT_PATTERNS)
    assert text == "On se voit demain à quinze heures."
    assert removed == ["Sous-titrage ST' 501"]


def test_retire_amara_dans_toutes_ses_variantes():
    for phrase in ("Sous-titres réalisés par la communauté d'Amara.org",
                   "Sous-titres réalisés para la communauté d'Amara.org",
                   "Sous-titres par Amara.org"):
        text, removed = artifacts.strip_artifacts(
            f"Le texte utile. {phrase}", artifacts.DEFAULT_ARTIFACT_PATTERNS)
        assert text == "Le texte utile.", phrase
        assert removed, phrase


def test_retire_les_remerciements_youtube():
    text, _ = artifacts.strip_artifacts(
        "Voilà pour aujourd'hui. Merci d'avoir regardé cette vidéo !",
        artifacts.DEFAULT_ARTIFACT_PATTERNS)
    assert text == "Voilà pour aujourd'hui."


def test_conserve_le_texte_quand_l_artefact_est_au_milieu():
    text, _ = artifacts.strip_artifacts(
        "Premier point. Sous-titrage ST' 501 Deuxième point.",
        artifacts.DEFAULT_ARTIFACT_PATTERNS)
    assert "Premier point." in text and "Deuxième point." in text
    assert "501" not in text


def test_ne_touche_pas_a_un_texte_normal():
    phrase = "Il faut vérifier les sous-titres du film avant de publier."
    text, removed = artifacts.strip_artifacts(phrase, artifacts.DEFAULT_ARTIFACT_PATTERNS)
    assert text == phrase
    assert removed == []


def test_segment_reduit_a_un_artefact_devient_vide():
    text, removed = artifacts.clean_segment("Sous-titrage ST' 501")
    assert text == ""
    assert removed


# ─── Motifs ambigus ───────────────────────────────────────────────────────────

def test_merci_seul_est_garde_si_whisper_est_confiant():
    text, _ = artifacts.clean_segment("Merci", no_speech_prob=0.05, avg_logprob=-0.2)
    assert text == "Merci"


def test_merci_seul_est_retire_si_whisper_doute():
    text, removed = artifacts.clean_segment("Merci", no_speech_prob=0.92, avg_logprob=-0.2)
    assert text == ""
    assert removed == ["Merci"]


def test_merci_est_retire_si_la_vraisemblance_est_basse():
    text, _ = artifacts.clean_segment("Merci beaucoup", no_speech_prob=0.1, avg_logprob=-1.8)
    assert text == ""


def test_merci_dans_une_phrase_reste_meme_si_whisper_doute():
    text, _ = artifacts.clean_segment("Merci de me rappeler demain",
                                      no_speech_prob=0.95, avg_logprob=-2.0)
    assert text == "Merci de me rappeler demain"


def test_filtre_ambigu_desactivable():
    text, _ = artifacts.clean_segment("Merci", no_speech_prob=0.99,
                                      ambiguous_enabled=False)
    assert text == "Merci"


# ─── Répétitions en boucle ────────────────────────────────────────────────────

def test_effondre_les_repetitions():
    text = artifacts.collapse_repetitions(
        "Je répète la phrase. Je répète la phrase. Je répète la phrase. "
        "Je répète la phrase. Fin.")
    assert text.count("Je répète la phrase.") == 2
    assert text.endswith("Fin.")


def test_ne_touche_pas_a_deux_repetitions():
    phrase = "Oui. Oui. On y va."
    assert artifacts.collapse_repetitions(phrase) == phrase


# ─── Chaîne sur des segments ──────────────────────────────────────────────────

def test_filter_transcription_assemble_et_nettoie():
    segments = [
        FakeSegment(" Bon alors le point sur la réunion."),
        FakeSegment(" Marie prend le back-end."),
        FakeSegment(" Sous-titrage ST' 501", no_speech_prob=0.8),
        FakeSegment(" Merci", no_speech_prob=0.95),
    ]
    text, removed = artifacts.filter_transcription(segments, {})
    assert text == "Bon alors le point sur la réunion. Marie prend le back-end."
    assert len(removed) == 2


def test_filter_transcription_accepte_des_dictionnaires():
    segments = [{"text": "Salut.", "no_speech_prob": 0.1, "avg_logprob": -0.2},
                {"text": "Merci d'avoir regardé cette vidéo", "no_speech_prob": 0.4}]
    text, removed = artifacts.filter_transcription(segments, {})
    assert text == "Salut."
    assert removed


def test_filtre_desactive_conserve_tout():
    segments = [FakeSegment("Salut."), FakeSegment("Sous-titrage ST' 501")]
    text, removed = artifacts.filter_transcription(segments, {"artifact_filter": False})
    assert "501" in text
    assert removed == []


def test_motifs_personnalises_pris_en_compte():
    segments = [FakeSegment("Bonjour. Générique de fin")]
    text, _ = artifacts.filter_transcription(
        segments, {"artifact_patterns": ["generique de fin"]})
    assert text == "Bonjour."


# ─── Régressions trouvées en revue ────────────────────────────────────────────

def test_la_typographie_francaise_du_texte_conserve_est_intacte():
    text, _ = artifacts.strip_artifacts(
        "Bonjour ! Ça va ? Sous-titrage ST' 501", artifacts.DEFAULT_ARTIFACT_PATTERNS)
    assert text == "Bonjour ! Ça va ?"          # les espaces avant ! et ? doivent rester


def test_un_motif_ne_traverse_pas_une_fin_de_phrase():
    phrase = "On dit merci. D'avoir regardé cette vidéo, on en reparlera demain."
    text, removed = artifacts.strip_artifacts(phrase, artifacts.DEFAULT_ARTIFACT_PATTERNS)
    assert text == phrase
    assert removed == []


def test_le_point_colle_reste_un_separateur():
    # « amara.org » doit continuer à correspondre au motif « amara org »
    text, removed = artifacts.strip_artifacts(
        "Fin. Sous-titres par Amara.org", artifacts.DEFAULT_ARTIFACT_PATTERNS)
    assert text == "Fin."
    assert removed


def test_un_chevauchement_ne_laisse_pas_de_fragment():
    text, removed = artifacts.strip_artifacts(
        "Bonjour. Sous-titres réalisés par la communauté d'Amara.org",
        artifacts.DEFAULT_ARTIFACT_PATTERNS)
    assert text == "Bonjour."
    assert "Sous-titres" not in text
    assert len(removed) == 1
