"""Moteur de corrections : c'est lui qui répare « cloud » → « Claude »."""

from sw import vocabulary


# ─── Biais Whisper ────────────────────────────────────────────────────────────

def test_hotwords_join_les_termes():
    assert vocabulary.build_hotwords(["Claude Code", "Ollama"]) == "Claude Code, Ollama"


def test_hotwords_vide_retourne_none():
    assert vocabulary.build_hotwords([]) is None
    assert vocabulary.build_hotwords(None) is None
    assert vocabulary.build_hotwords(["  ", ""]) is None


def test_initial_prompt_contient_les_termes_et_reste_court():
    prompt = vocabulary.build_initial_prompt(["Claude Code", "PipeWire"])
    assert "Claude Code" in prompt and "PipeWire" in prompt
    assert len(prompt) < 300


def test_initial_prompt_vide_retourne_none():
    assert vocabulary.build_initial_prompt([]) is None


# ─── Règles littérales ────────────────────────────────────────────────────────

def rule(source, target, regex=False, enabled=True):
    return {"from": source, "to": target, "regex": regex, "enabled": enabled}


def test_correction_litterale_insensible_a_la_casse():
    assert vocabulary.apply_corrections("CLOUD CODE marche", [rule("cloud code", "Claude Code")]) \
        == "Claude Code marche"


def test_correction_tolere_plusieurs_espaces():
    assert vocabulary.apply_corrections("git   hub", [rule("git hub", "GitHub")]) == "GitHub"


def test_correction_ne_coupe_pas_un_mot_plus_long():
    # « claud » ne doit pas transformer « Claudette » ni « claudication »
    result = vocabulary.apply_corrections("Claudette claudique", [rule("claud", "Claude")])
    assert result == "Claudette claudique"


def test_regle_desactivee_est_ignoree():
    assert vocabulary.apply_corrections("cloud code", [rule("cloud code", "Claude Code",
                                                            enabled=False)]) == "cloud code"


def test_remplacement_litteral_ne_lit_pas_les_antislash():
    assert vocabulary.apply_corrections("truc", [rule("truc", r"a\1b")]) == r"a\1b"


def test_regle_regex_autorise_les_groupes():
    rules = [rule(r"version (\d+)", r"v\1", regex=True)]
    assert vocabulary.apply_corrections("version 12", rules) == "v12"


def test_regex_invalide_ne_casse_pas_la_dictee():
    rules = [rule("(non fermé", "x", regex=True), rule("ok", "OK")]
    assert vocabulary.apply_corrections("ok", rules) == "OK"


def test_ordre_des_regles_respecte():
    rules = [rule("cloud code", "Claude Code"), rule("cloud", "Claude")]
    assert vocabulary.apply_corrections("cloud code et cloud", rules) == "Claude Code et Claude"


# ─── Règle « cloud » et ses exceptions ────────────────────────────────────────

def test_cloud_devient_claude_par_defaut():
    assert vocabulary.apply_cloud_rule("demande à cloud de corriger") \
        == "demande à Claude de corriger"


def test_cloud_capitalise_devient_claude():
    assert vocabulary.apply_cloud_rule("Cloud est rapide") == "Claude est rapide"


def test_exception_hebergement_preserve_cloud():
    for phrase in ("je déploie dans le cloud",
                   "on migre sur le cloud",
                   "un cloud public",
                   "le cloud AWS de l'agence",
                   "hébergement cloud mutualisé"):
        assert "cloud" in vocabulary.apply_cloud_rule(phrase), phrase


def test_exception_ne_bloque_pas_les_autres_occurrences():
    result = vocabulary.apply_cloud_rule("je demande à cloud de déployer dans le cloud")
    assert result == "je demande à Claude de déployer dans le cloud"


def test_exceptions_personnalisees_remplacent_les_defauts():
    result = vocabulary.apply_cloud_rule("dans le cloud", exceptions=[])
    assert result == "dans le Claude"


def test_cloud_dans_un_mot_compose_non_touche():
    assert vocabulary.apply_cloud_rule("soundcloud et cloudflare") == "soundcloud et cloudflare"


# ─── Chaîne complète ──────────────────────────────────────────────────────────

def test_correct_applique_regles_puis_regle_cloud():
    config = {
        "corrections_enabled": True,
        "corrections": vocabulary.DEFAULT_CORRECTIONS,
        "cloud_rule_enabled": True,
        "cloud_exceptions": vocabulary.DEFAULT_CLOUD_EXCEPTIONS,
    }
    texte = "je lance cloud code puis je pousse sur git hub et je déploie dans le cloud"
    attendu = ("je lance Claude Code puis je pousse sur GitHub et je déploie dans le cloud")
    assert vocabulary.correct(texte, config) == attendu


def test_correct_desactive_ne_touche_a_rien():
    config = {"corrections_enabled": False}
    assert vocabulary.correct("cloud code", config) == "cloud code"


def test_correct_sans_regle_cloud():
    config = {"corrections_enabled": True, "corrections": [], "cloud_rule_enabled": False}
    assert vocabulary.correct("demande à cloud", config) == "demande à cloud"


def test_les_defauts_corrigent_les_variantes_phonetiques():
    config = {"corrections_enabled": True, "corrections": vocabulary.DEFAULT_CORRECTIONS,
              "cloud_rule_enabled": False}
    assert vocabulary.correct("clode et cloude", config) == "Claude et Claude"


# ─── Sérialisation pour l'interface ───────────────────────────────────────────

def test_parse_corrections_text():
    rules = vocabulary.parse_corrections_text(
        "cloud code => Claude Code\n"
        "re:\\bv(\\d+)\\b => version \\1\n"
        "# desactive => rien\n"
        "\n"
        "ligne sans fleche\n")
    assert rules == [
        {"from": "cloud code", "to": "Claude Code", "regex": False, "enabled": True},
        {"from": r"\bv(\d+)\b", "to": r"version \1", "regex": True, "enabled": True},
        {"from": "desactive", "to": "rien", "regex": False, "enabled": False},
    ]


def test_aller_retour_texte_regles():
    rules = vocabulary.DEFAULT_CORRECTIONS
    assert vocabulary.parse_corrections_text(
        vocabulary.format_corrections_text(rules)) == rules


def test_format_prefixe_les_regles_desactivees():
    texte = vocabulary.format_corrections_text([rule("a", "b", enabled=False)])
    assert texte == "# a => b"


def test_parse_list_text_ignore_vides_et_commentaires():
    assert vocabulary.parse_list_text("Claude\n\n# note\n  Ollama  \n") == ["Claude", "Ollama"]
