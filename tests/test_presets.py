"""Résolution des formats, des surcharges de consignes et de la traduction."""

from sw import presets

EMPTY = {"reformat_prompt_overrides": {}, "reformat_custom_modes": []}


# ─── Consignes ────────────────────────────────────────────────────────────────

def test_les_formats_historiques_existent_toujours():
    # Les prompts personnalisés déjà enregistrés par l'utilisateur pointent sur ces identifiants
    for mode in ("message", "compact", "dev"):
        assert mode in presets.BUILTIN_PRESETS


def test_les_nouveaux_formats_sont_complets():
    for mode in ("readable", "prompt"):
        assert presets.BUILTIN_PRESETS[mode]["name"]
        assert presets.BUILTIN_PRESETS[mode]["prompt"].strip()


def test_les_formats_de_prompt_gardent_leur_liste_de_consignes_bannies():
    # « vérifie ton travail », « étape par étape » et l'orchestration d'agents imposée dégradent
    # la réponse des modèles Claude 5 : les deux formats doivent interdire de les écrire.
    for mode in ("prompt", "dev"):
        assert "À NE JAMAIS ÉCRIRE" in presets.BUILTIN_PRESETS[mode]["prompt"]


def test_instruction_dev_ne_prescrit_plus_de_strategie_dexecution():
    prompt = presets.BUILTIN_PRESETS["dev"]["prompt"]
    assert "Stratégie d'exécution" not in prompt
    assert "Relire son propre code" not in prompt


def test_message_soigne_borne_les_emojis():
    prompt = presets.BUILTIN_PRESETS["readable"]["prompt"]
    assert "UN SEUL emoji" in prompt
    assert "🙂" in prompt          # la liste fermée est bien dans la consigne
    # qwen3:8b ne sait pas s'abstenir d'emoji : la liste exclut donc les emojis festifs, qui
    # sont les seuls à vraiment détonner sur un message factuel (mesuré, voir le design doc).
    for festif in ("🎉", "🙌"):
        assert festif not in prompt


def test_prompt_par_defaut():
    prompt = presets.preset_prompt(EMPTY, "mail")
    assert "e-mail" in prompt.lower()


def test_surcharge_prioritaire_sur_le_defaut():
    config = {"reformat_prompt_overrides": {"message": "Ma consigne"}}
    assert presets.preset_prompt(config, "message") == "Ma consigne"


def test_format_personnalise():
    config = {"reformat_custom_modes": [{"name": "Tweet", "prompt": "Fais court"}]}
    assert presets.preset_prompt(config, "custom:Tweet") == "Fais court"


def test_format_inconnu_retourne_none():
    assert presets.preset_prompt(EMPTY, "custom:Absent") is None
    assert presets.preset_prompt(EMPTY, "disabled") is None
    assert presets.preset_prompt(EMPTY, None) is None


# ─── Libellés et backends ─────────────────────────────────────────────────────

def test_libelle_du_mode():
    assert presets.mode_label(EMPTY, "disabled") == "Brut"
    assert presets.mode_label(EMPTY, "github") == "Ticket GitHub"
    assert presets.mode_label(EMPTY, "custom:Tweet") == "Tweet"


def test_backend_par_defaut_puis_par_mode():
    config = dict(EMPTY, reformat_backend="ollama",
                  reformat_mode_backends={"dev": "claude"})
    assert presets.mode_backend(config, "message") == "ollama"
    assert presets.mode_backend(config, "dev") == "claude"


def test_backend_d_un_mode_personnalise():
    config = {"reformat_backend": "ollama",
              "reformat_custom_modes": [{"name": "X", "prompt": "", "backend": "claude"}]}
    assert presets.mode_backend(config, "custom:X") == "claude"


def test_liste_des_modes_commence_par_brut_et_finit_par_les_customs():
    config = {"reformat_custom_modes": [{"name": "Tweet", "prompt": ""}]}
    modes = presets.list_modes(config)
    assert modes[0][1] == presets.DISABLED
    assert modes[-1] == ("Tweet", "custom:Tweet")


# ─── Traduction ───────────────────────────────────────────────────────────────

def test_pas_de_directive_sans_langue():
    assert presets.translation_directive("none") == ""
    assert presets.translation_directive(None) == ""


def test_directive_nomme_la_langue():
    directive = presets.translation_directive("ja")
    assert "JAPONAIS" in directive
    assert "mise en forme" in directive


def test_resolve_sans_rien_a_faire():
    assert presets.resolve(EMPTY, "disabled", "none") is None


def test_resolve_ajoute_le_preambule_et_la_cloture():
    prompt = presets.resolve(EMPTY, "message", "none")
    assert prompt.startswith(presets.PREAMBLE)
    assert prompt.endswith(presets.CLOSING)


def test_resolve_cumule_format_et_langue():
    prompt = presets.resolve(EMPTY, "github", "en")
    assert "Markdown" in prompt
    assert "ANGLAIS" in prompt


def test_brut_plus_langue_bascule_sur_traduction_seule():
    prompt = presets.resolve(EMPTY, "disabled", "es")
    assert "ESPAGNOL" in prompt
    assert "Traduis" in prompt
    assert presets.resolve_effective_mode("disabled", "es") == "translate"


def test_traduction_seule_sans_langue_retombe_sur_l_anglais():
    prompt = presets.resolve(EMPTY, "translate", "none")
    assert "ANGLAIS" in prompt


def test_mode_effectif_inchange_sans_traduction():
    assert presets.resolve_effective_mode("message", "none") == "message"


def test_is_translating():
    assert presets.is_translating("en")
    assert not presets.is_translating("none")
    assert not presets.is_translating(None)
    assert not presets.is_translating("")


def test_toutes_les_langues_proposees_ont_un_nom_pour_la_directive():
    for _label, code in presets.LANGUAGES:
        if code == "none":
            continue
        assert code in presets.LANGUAGE_NAMES, code


def test_libelle_de_langue():
    assert presets.language_label("ja") == "Japonais"
    assert presets.language_label("none") == "Sans traduction"


def test_langue_effective_pour_le_controle_de_sortie():
    # « Traduction seule » sans langue choisie retombe sur l'anglais : le contrôle doit le savoir
    assert presets.effective_language("translate", "none") == "en"
    assert presets.effective_language("translate", "ja") == "ja"
    assert presets.effective_language("message", "none") == "none"
    assert presets.effective_language("message", "es") == "es"
