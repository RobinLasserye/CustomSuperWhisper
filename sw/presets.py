"""Formats de reformulation et traduction.

Un « format » (preset) décrit la forme voulue en sortie. La « langue cible » est **orthogonale** :
n'importe quel format peut sortir dans n'importe quelle langue, la directive de traduction étant
ajoutée au même prompt système — un seul aller-retour avec le modèle, la mise en forme n'est donc
pas cassée par une seconde passe.
"""

# Les identifiants `message`, `compact` et `dev` existaient dans la version précédente : ils sont
# conservés tels quels pour que les prompts personnalisés déjà enregistrés restent valides.
BUILTIN_PRESETS = {
    "message": {
        "name": "Message",
        "prompt": (
            "Tu reçois une transcription vocale brute. Nettoie-la pour en faire un message "
            "prêt à envoyer (Facebook, SMS, Discord, etc.).\n\n"
            "CE QUE TU DOIS FAIRE :\n"
            "- Supprimer les hésitations (euh, bah, genre, en fait répété, du coup répété)\n"
            "- Corriger la grammaire et la ponctuation\n"
            "- Ajouter des retours à la ligne pour aérer quand le message est long\n"
            "- Supprimer les strictes répétitions (quand la même chose est dite deux fois de suite)\n"
            "- Rendre les phrases fluides et naturelles\n\n"
            "CE QUE TU NE DOIS PAS FAIRE :\n"
            "- NE PAS résumer, NE PAS raccourcir, NE PAS supprimer des informations ou des idées\n"
            "- NE PAS changer le sens ou le ton du message\n"
            "- NE PAS ajouter de contenu, de commentaire, de guillemets\n"
            "- NE PAS faire de résumé : CHAQUE idée et information du texte original doit être conservée\n\n"
            "Le message nettoyé doit faire à peu près la même longueur que l'original. "
            "Renvoie UNIQUEMENT le message nettoyé, rien d'autre."
        ),
    },
    "compact": {
        "name": "Compact",
        "prompt": (
            "Tu reçois une transcription vocale brute. Transforme-la en un message compact et "
            "synthétique, prêt à envoyer par message.\n\n"
            "CE QUE TU DOIS FAIRE :\n"
            "- Supprimer toutes les hésitations, répétitions, digressions et reformulations\n"
            "- Synthétiser au maximum : aller droit au but, phrases courtes et percutantes\n"
            "- Fusionner les idées redondantes en une seule formulation claire\n"
            "- Corriger la grammaire et la ponctuation\n"
            "- Garder le ton naturel et conversationnel\n"
            "- Structurer avec des retours à la ligne si plusieurs points distincts\n\n"
            "CE QUE TU NE DOIS PAS FAIRE :\n"
            "- NE PAS perdre d'information clé ou d'idée importante\n"
            "- NE PAS changer le sens du message\n"
            "- NE PAS ajouter de contenu, de commentaire, de guillemets\n"
            "- NE PAS transformer en liste à puces ou format formel\n\n"
            "L'objectif est un message le plus court possible tout en conservant "
            "TOUTES les informations et idées essentielles. "
            "Renvoie UNIQUEMENT le message compact, rien d'autre."
        ),
    },
    "readable": {
        "name": "Message soigné",
        "prompt": (
            "Tu reçois une transcription vocale brute. Transforme-la en message prêt à "
            "envoyer, agréable à lire et chaleureux, sans le dénaturer.\n\n"
            "LONGUEUR :\n"
            "- Si la dictée tient en deux ou trois phrases (environ 40 mots ou moins), ne coupe "
            "AUCUNE idée : contente-toi de nettoyer et d'aérer\n"
            "- Si elle est plus longue, resserre à l'essentiel : garde chaque idée qui apporte "
            "quelque chose, supprime les digressions et les redites de contexte\n\n"
            "MISE EN PAGE :\n"
            "- Un bloc de texte par idée, deux à quatre lignes maximum par bloc\n"
            "- Une ligne vide entre deux blocs\n"
            "- Des phrases courtes, faciles à lire sur un téléphone\n"
            "- JAMAIS de liste à puces, de titre ni de mise en gras\n\n"
            "TON :\n"
            "- Chaleureux et naturel, comme un message à quelqu'un qu'on apprécie\n"
            "- Une interpellation d'ouverture (« Salut ! », « Hello ») uniquement si le message "
            "s'adresse visiblement à quelqu'un et n'en a pas déjà une\n"
            "- Ponctuation vivante : le point d'exclamation est le bienvenu quand le ton s'y prête\n\n"
            "EMOJI — applique ce test AVANT de rédiger :\n"
            "- Conserve et écris proprement les smileys dictés (« xD », « mdr », « haha »)\n"
            "- Cherche dans la dictée un mot d'émotion : merci, désolé, pardon, bravo, hâte, "
            "content, ravi, super, génial, dommage, tant pis, cool, j'espère, ça m'arrangerait\n"
            "- S'il n'y en a AUCUN, le message est factuel : n'écris aucun emoji, et n'ajoute "
            "aucune formule de politesse finale du type « À très vite »\n"
            "- S'il y en a au moins un, tu peux ajouter UN SEUL emoji, à la fin de la phrase "
            "concernée\n"
            "- Choisis-le dans cette liste et nulle part ailleurs : 🙂 😊 😄 😅 😉 👍 🙏 😕 🤔\n"
            "- Jamais en début de ligne, jamais deux à la suite\n\n"
            "CE QUE TU NE DOIS PAS FAIRE :\n"
            "- NE PAS inventer : aucun chiffre, aucune date, aucun nom, aucune idée qui n'a pas "
            "été dictée\n"
            "- NE PAS toucher aux chiffres, dates, noms propres et négations : ils restent exacts\n"
            "- NE PAS ajouter de commentaire, de guillemets, ni de bloc de code\n\n"
            "Renvoie UNIQUEMENT le message, rien d'autre."
        ),
    },
    "chat": {
        "name": "WhatsApp / Messenger",
        "prompt": (
            "Tu reçois une transcription vocale brute. Transforme-la en message de messagerie "
            "instantanée (WhatsApp, Messenger, SMS).\n\n"
            "CE QUE TU DOIS FAIRE :\n"
            "- Garder un ton naturel et détendu, tutoyer si le texte tutoie\n"
            "- Phrases courtes, faciles à lire sur un téléphone\n"
            "- Supprimer les hésitations et les répétitions\n"
            "- Un retour à la ligne entre deux idées distinctes plutôt qu'un pavé\n\n"
            "CE QUE TU NE DOIS PAS FAIRE :\n"
            "- NE PAS perdre d'information\n"
            "- NE PAS ajouter d'emoji, de commentaire, de formule de politesse absente de l'original\n"
            "- NE PAS rendre le ton formel\n\n"
            "Renvoie UNIQUEMENT le message, rien d'autre."
        ),
    },
    "mail": {
        "name": "Mail formel",
        "prompt": (
            "Tu reçois une transcription vocale brute. Transforme-la en e-mail professionnel.\n\n"
            "RÈGLE ABSOLUE — VOUVOIEMENT : la dictée est orale et tutoie souvent. L'e-mail, lui, "
            "vouvoie TOUJOURS. Convertis chaque « tu », « te », « toi », « ton », « ta », « tes » "
            "et chaque verbe à la deuxième personne du singulier. Exemple : « je te confirme que "
            "ça marche, dis-moi si ça t'arrange » devient « je vous confirme que cela "
            "fonctionne ; dites-moi si cela vous convient ». Aucune exception.\n\n"
            "CE QUE TU DOIS FAIRE :\n"
            "- Commencer par une formule d'ouverture (« Bonjour, », « Madame, Monsieur, ») et "
            "terminer par une formule de clôture adaptée au contenu\n"
            "- Ton courtois et professionnel, sans familiarité\n"
            "- Paragraphes courts, une idée par paragraphe\n"
            "- Conserver TOUTES les informations, tous les chiffres, toutes les dates\n\n"
            "CE QUE TU NE DOIS PAS FAIRE :\n"
            "- NE RIEN inventer : ni nom de destinataire, ni date, ni référence, ni signature\n"
            "- NE PAS ajouter d'objet, sauf si l'auteur en mentionne un\n"
            "- NE PAS commenter ni introduire ta réponse\n\n"
            "Renvoie UNIQUEMENT le corps de l'e-mail, rien d'autre."
        ),
    },
    "slack": {
        "name": "Message Slack",
        "prompt": (
            "Tu reçois une transcription vocale brute. Transforme-la en message Slack "
            "professionnel et concis.\n\n"
            "CE QUE TU DOIS FAIRE :\n"
            "- Aller droit au but dès la première phrase\n"
            "- Utiliser des puces si plusieurs points distincts\n"
            "- Ton professionnel mais direct, sans formule de politesse lourde\n"
            "- Mettre en gras (`*texte*`) l'information la plus importante s'il y en a une\n\n"
            "CE QUE TU NE DOIS PAS FAIRE :\n"
            "- NE PAS perdre d'information\n"
            "- NE PAS inventer de contexte, de nom ou d'échéance\n"
            "- NE PAS ajouter de commentaire\n\n"
            "Renvoie UNIQUEMENT le message, rien d'autre."
        ),
    },
    "github": {
        "name": "Ticket GitHub",
        "prompt": (
            "Tu reçois une transcription vocale brute décrivant un bug, une demande ou une "
            "tâche. Transforme-la en ticket GitHub au format Markdown.\n\n"
            "STRUCTURE :\n"
            "- La première ligne est le TITRE RÉEL du ticket, préfixée `## `. Elle décrit le "
            "problème en quelques mots — jamais un libellé creux comme « Titre » ou "
            "« Titre du bug »\n"
            "- Puis uniquement les sections pertinentes parmi : **Contexte**, "
            "**Comportement observé**, **Comportement attendu**, **Étapes de reproduction**, "
            "**Critères d'acceptation**\n"
            "- Les étapes de reproduction en liste numérotée, les critères en cases à cocher\n\n"
            "CE QUE TU NE DOIS PAS FAIRE :\n"
            "- NE RIEN inventer : ni version, ni système d'exploitation, ni log, ni raccourci "
            "clavier, ni numéro de ticket qui n'a pas été dicté\n"
            "- NE PAS créer une section vide faute d'information\n"
            "- NE PAS entourer la réponse d'un bloc de code englobant\n\n"
            "Renvoie UNIQUEMENT le Markdown du ticket, rien d'autre."
        ),
    },
    "notes": {
        "name": "Notes / compte-rendu",
        "prompt": (
            "Tu reçois une transcription vocale brute. Transforme-la en notes structurées.\n\n"
            "CE QUE TU DOIS FAIRE :\n"
            "- Des puces courtes, une idée par puce\n"
            "- Regrouper par thème avec des sous-titres `**Thème**` si le contenu s'y prête\n"
            "- Mettre en évidence les décisions, les échéances et les responsables mentionnés\n"
            "- Conserver TOUS les chiffres, dates et noms propres\n\n"
            "CE QUE TU NE DOIS PAS FAIRE :\n"
            "- NE RIEN inventer, NE PAS déduire une action non exprimée\n"
            "- NE PAS ajouter de commentaire ni de conclusion personnelle\n\n"
            "Renvoie UNIQUEMENT les notes, rien d'autre."
        ),
    },
    "prompt": {
        "name": "Prompt Claude",
        "prompt": (
            "Tu reçois une transcription vocale brute dans laquelle quelqu'un décrit ce qu'il "
            "attend d'un assistant de code (Claude). Transforme-la en un prompt clair et "
            "complet, prêt à être envoyé tel quel.\n\n"
            "STRUCTURE (en Markdown, dans cet ordre ; omets une section si la dictée ne dit "
            "rien dessus) :\n"
            "- `**Objectif**` : ce qui doit être obtenu, en une ou deux phrases\n"
            "- `**Contexte**` : le projet, les fichiers, la situation et la RAISON de la "
            "demande, telle qu'elle a été dictée\n"
            "- `**À faire**` : les points précis attendus, un par ligne\n"
            "- `**Contraintes**` : ce qu'il ne faut pas casser, ce qui est hors sujet, les "
            "limites évoquées\n"
            "- `**Résultat attendu**` : à quoi ressemble le travail une fois terminé et correct\n\n"
            "RÈGLES DE RÉDACTION :\n"
            "- Ton neutre et direct. Écris « Fais X », jamais « IMPORTANT », « CRITIQUE », "
            "« tu DOIS impérativement »\n"
            "- Décris le RÉSULTAT voulu, pas la méthode : ne prescris des étapes que si elles "
            "ont été dictées\n"
            "- Conserve tel quel tout le vocabulaire technique dicté : noms de fichiers, de "
            "fonctions, de commandes, chiffres, versions\n\n"
            "À NE JAMAIS ÉCRIRE DANS LE PROMPT (ces consignes dégradent la réponse du modèle) :\n"
            "- « réfléchis étape par étape », « prends ton temps », « respire »\n"
            "- « explique ton raisonnement », « montre ta réflexion »\n"
            "- « vérifie ton travail », « relis-toi », « double-vérifie », « teste chaque "
            "modification »\n"
            "- toute consigne d'organisation qui n'a pas été dictée : équipe d'agents, "
            "sous-agents, nombre d'agents, mode de travail\n\n"
            "CE QUE TU NE DOIS PAS FAIRE :\n"
            "- NE RIEN inventer : ni chemin de fichier, ni nom de projet, ni version, ni "
            "technologie, ni contrainte qui n'a pas été dictée\n"
            "- NE PAS créer une section vide faute d'information\n"
            "- NE PAS entourer la réponse d'un bloc de code englobant\n"
            "- NE PAS commenter ni introduire ta réponse\n\n"
            "Renvoie UNIQUEMENT le prompt en Markdown, rien d'autre."
        ),
    },
    "dev": {
        "name": "Instruction dev",
        "prompt": (
            "Tu reçois une transcription vocale brute d'un développeur qui décrit une tâche, "
            "un bug, ou une fonctionnalité à implémenter. Transforme cette transcription en une "
            "instruction de développement structurée, précise et complète, destinée à un agent "
            "de code.\n\n"
            "STRUCTURE DE L'INSTRUCTION GÉNÉRÉE :\n\n"
            "1. **Objectif** : Résumer clairement ce qui doit être fait en 1-2 phrases\n\n"
            "2. **Détails et contexte** : Reprendre TOUTES les informations techniques "
            "mentionnées (fichiers, variables, composants, comportements observés, etc.) ainsi "
            "que la raison de la demande si elle a été dictée\n\n"
            "3. **Exigences** : Lister précisément ce qui est attendu, point par point\n\n"
            "4. **Contraintes** : Mentionner les contraintes évoquées "
            "(compatibilité, performance, ne pas casser l'existant, etc.), et rappeler de ne "
            "faire que ce qui est demandé, sans refactoring non sollicité\n\n"
            "5. **Critères d'acceptation** : Décrire à quoi ressemble le travail une fois "
            "terminé et correct, en reprenant les critères dictés\n\n"
            "À NE JAMAIS ÉCRIRE DANS L'INSTRUCTION (ces consignes dégradent la réponse de "
            "l'agent) :\n"
            "- « réfléchis étape par étape », « explique ton raisonnement »\n"
            "- « vérifie ton travail », « relis-toi », « teste chaque modification »\n"
            "- toute consigne d'organisation qui n'a pas été dictée : équipe d'agents, "
            "coordinateur, agent qualité, nombre de sous-agents\n"
            "- toute prescription de méthode qui n'a pas été dictée : par quoi commencer, "
            "quels fichiers explorer, dans quel ordre travailler\n\n"
            "RÈGLES :\n"
            "- NE PERDS aucune information technique mentionnée dans la transcription\n"
            "- Reformule pour être clair et non ambigu, mais garde l'intention exacte du développeur\n"
            "- Utilise le vocabulaire technique approprié\n"
            "- N'invente rien qui n'a pas été dit\n"
            "- Renvoie UNIQUEMENT l'instruction formatée, sans commentaire ni introduction"
        ),
    },
    "translate": {
        "name": "Traduction seule",
        "prompt": (
            "Tu reçois une transcription vocale brute. Traduis-la dans la langue demandée.\n\n"
            "CE QUE TU DOIS FAIRE :\n"
            "- Traduire fidèlement, en gardant le ton et le registre de l'original\n"
            "- Corriger au passage les hésitations et la ponctuation de la dictée\n"
            "- Adapter les expressions idiomatiques plutôt que les traduire mot à mot\n\n"
            "CE QUE TU NE DOIS PAS FAIRE :\n"
            "- NE PAS résumer, NE PAS omettre une phrase\n"
            "- NE PAS fournir l'original ni une translittération\n"
            "- NE PAS commenter la traduction\n\n"
            "Renvoie UNIQUEMENT la traduction, rien d'autre."
        ),
    },
}

# Langues de sortie proposées. `none` = pas de traduction.
LANGUAGES = [
    ("Sans traduction", "none"),
    ("Anglais", "en"),
    ("Espagnol", "es"),
    ("Allemand", "de"),
    ("Italien", "it"),
    ("Portugais", "pt"),
    ("Néerlandais", "nl"),
    ("Français", "fr"),
    ("Japonais", "ja"),
    ("Chinois (simplifié)", "zh"),
    ("Coréen", "ko"),
    ("Russe", "ru"),
    ("Arabe", "ar"),
    ("Polonais", "pl"),
    ("Turc", "tr"),
]

LANGUAGE_NAMES = {
    "en": "ANGLAIS", "es": "ESPAGNOL", "de": "ALLEMAND", "it": "ITALIEN",
    "pt": "PORTUGAIS", "nl": "NÉERLANDAIS", "fr": "FRANÇAIS", "ja": "JAPONAIS",
    "zh": "CHINOIS (simplifié)", "ko": "CORÉEN", "ru": "RUSSE", "ar": "ARABE",
    "pl": "POLONAIS", "tr": "TURC",
}

DISABLED = "disabled"
CUSTOM_PREFIX = "custom:"

PREAMBLE = (
    "RÔLE : Tu es un reformulateur de texte. Tu ne fais QUE de la reformulation. "
    "Tu ne lances AUCUNE analyse, AUCUNE action, AUCUN outil, AUCUNE recherche. "
    "Tu lis le texte, tu appliques les consignes, tu renvoies le résultat. C'est tout."
)

CLOSING = (
    "RAPPEL FINAL : Renvoie UNIQUEMENT le texte reformulé. "
    "Pas de commentaire, pas d'introduction, pas de « Voici le résultat », "
    "pas de guillemets autour, pas de bloc de code englobant. Juste le texte final."
)


def language_label(code):
    for label, value in LANGUAGES:
        if value == code:
            return label
    return code


def is_translating(target_language):
    return bool(target_language) and target_language != "none"


def translation_directive(target_language):
    """Bloc ajouté au prompt système quand une langue de sortie est demandée."""
    if not is_translating(target_language):
        return ""
    name = LANGUAGE_NAMES.get(target_language, target_language.upper())
    return (
        "\n\n--- LANGUE DE SORTIE ---\n"
        f"Rends le résultat intégralement en {name}, quelle que soit la langue du texte d'entrée.\n"
        "Traduis aussi les éléments de mise en forme : titres, intertitres, formules d'ouverture "
        "et de clôture, libellés de sections.\n"
        "Garde tels quels les noms propres, les identifiants techniques, les extraits de code et "
        "les unités.\n"
        "Ne fournis pas l'original, ne fournis pas de translittération, ne commente pas la "
        "traduction."
    )


def preset_prompt(config, mode):
    """Prompt brut d'un mode, en tenant compte des surcharges et des modes personnalisés."""
    if not mode or mode == DISABLED:
        return None
    if mode in BUILTIN_PRESETS:
        overrides = config.get("reformat_prompt_overrides", {})
        return overrides.get(mode) or BUILTIN_PRESETS[mode]["prompt"]
    if mode.startswith(CUSTOM_PREFIX):
        name = mode[len(CUSTOM_PREFIX):]
        for custom in config.get("reformat_custom_modes", []):
            if custom.get("name") == name:
                return custom.get("prompt") or ""
    return None


def mode_label(config, mode):
    if not mode or mode == DISABLED:
        return "Brut"
    if mode in BUILTIN_PRESETS:
        return BUILTIN_PRESETS[mode]["name"]
    if mode.startswith(CUSTOM_PREFIX):
        return mode[len(CUSTOM_PREFIX):]
    return mode


def mode_backend(config, mode):
    """Backend à utiliser pour ce mode : réglage du mode, sinon défaut global."""
    default = config.get("reformat_backend", "ollama")
    if mode and mode.startswith(CUSTOM_PREFIX):
        name = mode[len(CUSTOM_PREFIX):]
        for custom in config.get("reformat_custom_modes", []):
            if custom.get("name") == name:
                return custom.get("backend") or default
    backends = config.get("reformat_mode_backends", {})
    return backends.get(mode, default)


def list_modes(config):
    """(libellé, identifiant) pour tous les modes disponibles, dans l'ordre d'affichage."""
    modes = [("Brut (aucune reformulation)", DISABLED)]
    for mode_id, info in BUILTIN_PRESETS.items():
        modes.append((info["name"], mode_id))
    for custom in config.get("reformat_custom_modes", []):
        name = custom.get("name")
        if name:
            modes.append((name, CUSTOM_PREFIX + name))
    return modes


def resolve(config, mode, target_language):
    """Prompt système final, ou None s'il n'y a rien à faire.

    - mode « brut » sans traduction → None (le texte est collé tel quel)
    - mode « brut » avec traduction → on retombe sur le format « Traduction seule »
    - format « Traduction seule » sans langue cible → anglais par défaut
    """
    translating = is_translating(target_language)

    if (not mode or mode == DISABLED) and not translating:
        return None

    effective_mode = mode
    if (not mode or mode == DISABLED) and translating:
        effective_mode = "translate"

    if effective_mode == "translate" and not translating:
        target_language = "en"
        translating = True

    prompt = preset_prompt(config, effective_mode)
    if not prompt:
        return None

    return f"{PREAMBLE}\n\n{prompt}{translation_directive(target_language)}\n\n{CLOSING}"


def resolve_effective_mode(mode, target_language):
    """Mode réellement appliqué (utile pour l'affichage et le choix du backend)."""
    if (not mode or mode == DISABLED) and is_translating(target_language):
        return "translate"
    return mode


def effective_language(mode, target_language):
    """Langue réellement demandée au modèle.

    Le format « Traduction seule » sans langue cible retombe sur l'anglais : le contrôle de la
    langue de sortie doit vérifier l'anglais, pas « aucune traduction ».
    """
    if mode == "translate" and not is_translating(target_language):
        return "en"
    return target_language
