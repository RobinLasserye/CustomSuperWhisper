# SuperWhisper Custom — formats « Message soigné » et « Prompt Claude »

Date : 2026-08-07
Statut : en attente de validation du PO (Robin)

## Problèmes traités

1. **`message` aère mal.** Le format conserve tout (c'est sa consigne), mais il laisse des pavés :
   les phrases et les idées ne sont ni remises à la ligne ni espacées, donc le résultat est fidèle
   sans être lisible.
2. **`compact` est sec.** Il synthétise correctement mais rend un texte plat, qui se lit comme une
   note de service et pas comme un message qu'on a envie de recevoir — au point qu'il faut parfois
   s'en excuser auprès du destinataire.
3. **Aucun format ne produit un prompt.** Dicter une demande de développement donne, au mieux,
   `dev`, dont les consignes sont datées (voir plus bas).
4. **`dev` embarque trois anti-patterns désormais documentés** par Anthropic : demander une équipe
   d'agents, faire relire son propre code, faire vérifier chaque modification.

## Objectifs

- Un format de message **lisible, compact et chaleureux**, entre `message` (rien coupé, mal aéré)
  et `compact` (bien coupé, froid).
- Un format qui transforme une dictée en **prompt optimisé pour Claude**, valable sur toute la
  génération 5.
- Ne rien casser : les formats et les prompts personnalisés existants restent valides.

## Non-objectifs

- Pas de refonte de `message` ni de `compact` : ils gardent leur comportement exact.
- Pas de deuxième variante de prompt par modèle (voir « Une seule version », plus bas).
- Pas de changement du format par défaut : `reformat_mode` reste `disabled` (brut).

## Décisions du PO

| Question | Choix |
|---|---|
| Périmètre | Nouveaux formats **à côté** de `message` et `compact` |
| Smileys | Emoji **parcimonieux autorisé**, borné par une liste fermée |
| Compression | **Adaptative** selon la longueur de la dictée |
| `dev` | **Modernisé** dans la foulée |

## Format 1 — « Message soigné » (id `readable`)

### Règles

**Longueur.** Sous deux ou trois phrases (~40 mots), aucune idée n'est coupée : nettoyage et
aération seulement. Au-delà, resserrage à l'essentiel — les digressions et les redites de contexte
sautent, les idées restent.

Le seuil est exprimé en *phrases* autant qu'en mots : un modèle 8B compte mal les mots mais
reconnaît bien « deux ou trois phrases ».

**Mise en page.** Un bloc par idée, deux à quatre lignes par bloc, ligne vide entre les blocs,
phrases courtes. Jamais de puces, de titre ni de gras — c'est un message, pas un compte-rendu
(`notes` existe déjà pour ça).

**Ton.** Chaleureux et naturel. Interpellation d'ouverture seulement si le message s'adresse
visiblement à quelqu'un et n'en a pas déjà une.

**Emoji.** Au plus un, jamais en début de ligne, jamais deux à la suite, choisi dans une liste
fermée : 🙂 😊 😄 😅 😉 👍 🙏 😕 🤔.

Le déclenchement repose sur un **test lexical** — la présence d'un mot d'émotion dans la dictée
(merci, désolé, bravo, hâte, dommage, tant pis, ça m'arrangerait…) — et non sur un jugement
sémantique « le message est-il factuel ? ». Un 8B suit bien mieux un critère lexical.

Les smileys **dictés** (« xD », « mdr », « haha ») sont conservés et normalisés dans tous les cas ;
ils ne comptent pas dans le quota d'un emoji ajouté.

### Limite mesurée : l'abstention n'est pas atteignable sur qwen3:8b

Trois formulations successives ont été essayées sur le modèle par défaut, chacune sur plusieurs
dictées purement factuelles (rendez-vous et adresse, numéro de suivi de colis) :

1. « Aucun emoji sur un message purement factuel » → **0/4** conforme ;
2. la même règle placée en tête, avec énumération des cas factuels → **0/2** ;
3. le test lexical ci-dessus, avec interdiction explicite des formules de politesse ajoutées
   → **0/4** sur les factuels, **4/4** sur les dictées émotionnelles.

**Le modèle place un emoji dans tous les cas.** C'est un plafond du modèle, pas une faiblesse de
formulation. Ce qui fonctionne, en revanche : le nombre (jamais plus d'un), la position (jamais en
début de ligne), la liste fermée (respectée sur tous les essais), et surtout la **pertinence du
choix** quand il y a de l'émotion — 😕 sur une annulation, 😊 sur un remerciement.

La consigne lexicale est donc conservée pour cette pertinence, et le dégât résiduel est borné en
retirant 🎉 et 🙌 de la liste : ce sont les seuls qui détonnaient vraiment (un 🎉 était apparu sur un
rendez-vous administratif). Un 🙂 en fin de message factuel reste inoffensif.

**Si l'abstention stricte devient nécessaire**, la seule voie fiable est un post-traitement
déterministe côté code — retirer les emojis de la sortie quand la dictée source ne contient aucun
mot d'émotion. Hors périmètre de ce lot : cela introduirait un traitement spécifique à un format
dans un pipeline aujourd'hui générique (`sw/backends.py`, `clean_output`).

**Invariants du projet.** Chiffres, dates, noms propres et négations intacts ; rien d'inventé ;
aucun commentaire ni guillemet englobant.

### Prompt

```
Tu reçois une transcription vocale brute. Transforme-la en message prêt à envoyer,
agréable à lire et chaleureux, sans le dénaturer.

LONGUEUR :
- Si la dictée tient en deux ou trois phrases (environ 40 mots ou moins), ne coupe AUCUNE
  idée : contente-toi de nettoyer et d'aérer
- Si elle est plus longue, resserre à l'essentiel : garde chaque idée qui apporte quelque
  chose, supprime les digressions et les redites de contexte

MISE EN PAGE :
- Un bloc de texte par idée, deux à quatre lignes maximum par bloc
- Une ligne vide entre deux blocs
- Des phrases courtes, faciles à lire sur un téléphone
- JAMAIS de liste à puces, de titre ni de mise en gras

TON :
- Chaleureux et naturel, comme un message à quelqu'un qu'on apprécie
- Une interpellation d'ouverture (« Salut ! », « Hello ») uniquement si le message s'adresse
  visiblement à quelqu'un et n'en a pas déjà une
- Ponctuation vivante : le point d'exclamation est le bienvenu quand le ton s'y prête

EMOJI — applique ce test AVANT de rédiger :
- Conserve et écris proprement les smileys dictés (« xD », « mdr », « haha »)
- Cherche dans la dictée un mot d'émotion : merci, désolé, pardon, bravo, hâte, content,
  ravi, super, génial, dommage, tant pis, cool, j'espère, ça m'arrangerait
- S'il n'y en a AUCUN, le message est factuel : n'écris aucun emoji, et n'ajoute aucune
  formule de politesse finale du type « À très vite »
- S'il y en a au moins un, tu peux ajouter UN SEUL emoji, à la fin de la phrase concernée
- Choisis-le dans cette liste et nulle part ailleurs : 🙂 😊 😄 😅 😉 👍 🙏 😕 🤔
- Jamais en début de ligne, jamais deux à la suite

CE QUE TU NE DOIS PAS FAIRE :
- NE PAS inventer : aucun chiffre, aucune date, aucun nom, aucune idée qui n'a pas été dictée
- NE PAS toucher aux chiffres, dates, noms propres et négations : ils restent exacts
- NE PAS ajouter de commentaire, de guillemets, ni de bloc de code

Renvoie UNIQUEMENT le message, rien d'autre.
```

## Format 2 — « Prompt Claude » (id `prompt`)

### Une seule version, pas une par modèle

Les guides de prompting d'Anthropic sont désormais **par modèle**, et Fable 5 et Opus 5 divergent
sur quatre points :

| | Opus 5 | Fable 5 |
|---|---|---|
| « vérifie ton travail » | à **supprimer** (sur-vérification) | à rendre **explicite** sur les runs longs |
| sous-agents | **plafonner** la délégation | **encourager**, en asynchrone |
| raisonnement visible | neutre | **interdit** — refus `reasoning_extraction` |
| prompt très prescriptif | toléré | **dégrade** la qualité |

Ces divergences portent sur le **prompt système d'un harness** — cadence de vérification,
orchestration d'agents — pas sur la formulation d'une tâche. L'intersection des deux guides est non
vide : n'imposer ni vérification ni délégation, ne jamais demander de raisonnement visible, décrire
le résultat plutôt que la méthode. Un seul format couvre donc Fable 5, Opus 5 et Sonnet 5.

Sources :
[Prompting Claude Opus 5](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/prompting-claude-opus-5),
[Prompting Claude Fable 5](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/prompting-claude-fable-5),
[Prompting best practices](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices).

### Structure produite

Markdown, sections omises quand la dictée ne les alimente pas :

```
**Objectif**          une ou deux phrases
**Contexte**          projet, fichiers, situation, et la RAISON de la demande
**À faire**           les points précis attendus
**Contraintes**       ce qu'il ne faut pas casser, ce qui est hors sujet
**Résultat attendu**  à quoi ressemble le travail terminé et correct
```

`**Résultat attendu**` remplace l'injonction « vérifie ton travail » : énoncer les critères de fin
est utile aux deux modèles et n'est pas une consigne de vérification.

`**Contexte**` porte la *raison* de la demande, pas seulement son objet — c'est une recommandation
explicite du guide Fable 5 (« Give the reason, not only the request »).

### Prompt

```
Tu reçois une transcription vocale brute dans laquelle quelqu'un décrit ce qu'il attend d'un
assistant de code (Claude). Transforme-la en un prompt clair et complet, prêt à être envoyé
tel quel.

STRUCTURE (en Markdown, dans cet ordre ; omets une section si la dictée ne dit rien dessus) :
- `**Objectif**` : ce qui doit être obtenu, en une ou deux phrases
- `**Contexte**` : le projet, les fichiers, la situation et la RAISON de la demande, telle
  qu'elle a été dictée
- `**À faire**` : les points précis attendus, un par ligne
- `**Contraintes**` : ce qu'il ne faut pas casser, ce qui est hors sujet, les limites évoquées
- `**Résultat attendu**` : à quoi ressemble le travail une fois terminé et correct

RÈGLES DE RÉDACTION :
- Ton neutre et direct. Écris « Fais X », jamais « IMPORTANT », « CRITIQUE », « tu DOIS
  impérativement »
- Décris le RÉSULTAT voulu, pas la méthode : ne prescris des étapes que si elles ont été dictées
- Conserve tel quel tout le vocabulaire technique dicté : noms de fichiers, de fonctions, de
  commandes, chiffres, versions

À NE JAMAIS ÉCRIRE DANS LE PROMPT (ces consignes dégradent la réponse du modèle) :
- « réfléchis étape par étape », « prends ton temps », « respire »
- « explique ton raisonnement », « montre ta réflexion »
- « vérifie ton travail », « relis-toi », « double-vérifie », « teste chaque modification »
- toute consigne d'organisation qui n'a pas été dictée : équipe d'agents, sous-agents, nombre
  d'agents, mode de travail

CE QUE TU NE DOIS PAS FAIRE :
- NE RIEN inventer : ni chemin de fichier, ni nom de projet, ni version, ni technologie, ni
  contrainte qui n'a pas été dictée
- NE PAS créer une section vide faute d'information
- NE PAS entourer la réponse d'un bloc de code englobant
- NE PAS commenter ni introduire ta réponse

Renvoie UNIQUEMENT le prompt en Markdown, rien d'autre.
```

## Format 3 — « Instruction dev » (`dev`) modernisé

`dev` garde son identité : une instruction de développement **longue et très structurée**, avec
sections numérotées et vocabulaire technique. `prompt` reste le format court et générique, utilisable
aussi hors développement.

Retiré du prompt actuel (`sw/presets.py:146`) :

- la section « Stratégie d'exécution » entière — équipe d'agents, coordinateur, agent qualité,
  skill de design : orchestration imposée, à proscrire sur Opus 5 comme sur Fable 5 ;
- « Relire son propre code après implémentation et vérifier l'absence de régression » ;
- « Vérifier chaque modification, tester les cas limites » ;
- « Explorer d'abord le code existant en profondeur avant de modifier quoi que ce soit » —
  prescription de méthode, que les deux guides recommandent de laisser au modèle.

Conservé, et renforcé : « Ne faire que ce qui est demandé, pas de refactoring non sollicité ».
C'est la discipline de scope, explicitement recommandée pour les deux modèles.

Ajouté : une section **Critères d'acceptation**, qui remplace les consignes de vérification
supprimées par leur équivalent utile.

L'identifiant `dev` ne change pas : un prompt personnalisé enregistré sous cette clé continue de
gagner sur le prompt intégré (`presets.preset_prompt`).

## Intégration

Deux entrées ajoutées à `BUILTIN_PRESETS` (`sw/presets.py:11`) : `readable` juste après `compact`,
`prompt` juste avant `dev`. `list_modes`, le picker, le menu du tray et l'onglet des réglages se
dérivent tous de ce dictionnaire — aucun autre câblage n'est nécessaire.

Les identifiants `readable` et `prompt` sont neufs : aucune clé existante de
`reformat_prompt_overrides` ni de `reformat_mode_backends` n'est touchée.

## Tests

- `tests/test_presets.py` : `readable` et `prompt` présents dans `BUILTIN_PRESETS`, avec un nom et
  un prompt non vide ; `resolve()` les enveloppe bien entre `PREAMBLE` et `CLOSING` ; la directive
  de traduction s'ajoute correctement.
- `tools/eval_models.py` : deux cas ajoutés au test de pièges.
  - Un cas `readable` sur une dictée courte : toutes les informations survivent, aucun bavardage,
    au plus un emoji.
  - Un cas `prompt` sur une dictée technique : les identifiants dictés survivent, et aucune des
    formules interdites (« étape par étape », « vérifie ton travail », « sous-agents »… ) n'apparaît
    dans la sortie.

## Validation

`tools/eval_models.py` est relancé pour de vrai sur `qwen3:8b` (le modèle par défaut) après
implémentation, et les sorties sont montrées au PO. Un tirage isolé varie de ±1 point sur ce test :
les cas ajoutés servent de garde-fou de non-régression, pas de note absolue.
