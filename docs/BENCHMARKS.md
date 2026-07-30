# Benchmarks — SuperWhisper Custom

Mesures du 30 juillet 2026, sur une **RTX 5090** (32 Go), Fedora 43, CUDA 12, ctranslate2 4.7.1,
faster-whisper 1.2.1, Ollama 0.20.7, Python 3.14.

Tout est rejouable : les scripts vivent dans `tools/`, les mesures brutes sont reproduites ici.

## Sommaire

1. [Transcription : modèle × précision](#1-transcription--modèle--précision)
2. [L'effet du biais vocabulaire sur « Claude »](#2-leffet-du-biais-vocabulaire-sur--claude-)
3. [Reformulation : latence et VRAM](#3-reformulation--latence-et-vram)
4. [Reformulation : fidélité](#4-reformulation--fidélité)
5. [Latence bout-en-bout selon la longueur de la dictée](#5-latence-bout-en-bout-selon-la-longueur-de-la-dictée)
6. [Configurations conseillées par carte graphique](#6-configurations-conseillées-par-carte-graphique)

---

## 1. Transcription : modèle × précision

30 clips français (166 s d'audio), `beam_size=5`, `vad_filter=True`, chaque combinaison dans un
sous-processus isolé pour que la mesure de VRAM ne soit pas polluée par la précédente.

| Modèle | Précision | VRAM | Chargement | Vitesse | WER sans biais | WER avec biais |
|---|---|---:|---:|---:|---:|---:|
| tiny | float16 | 0,66 Go | 0,4 s | 32× | 0,965 | 0,770 |
| base | float16 | 0,75 Go | 0,4 s | 23× | 0,867 | 0,714 |
| small | int8 | 0,84 Go | 0,7 s | 26× | 0,573 | 0,509 |
| small | float16 | 1,19 Go | 0,5 s | 35× | 0,646 | 0,537 |
| distil-large-v3 | int8 | 1,50 Go | 1,3 s | 61× | 0,927 | 0,913 |
| **large-v3-turbo** | **int8** | **1,59 Go** | 1,5 s | **55×** | 0,354 | **0,268** |
| large-v3 | int8 | 2,34 Go | 2,3 s | 23× | 0,361 | **0,243** |
| medium | float16 | 2,41 Go | 0,8 s | 22× | 0,470 | 0,338 |
| distil-large-v3 | float16 | 2,44 Go | 0,9 s | 59× | 0,928 | 0,917 |
| large-v3-turbo | float16 | 2,56 Go | 1,0 s | 55× | 0,440 | 0,298 |
| large-v3 | float16 | 4,19 Go | 1,3 s | 24× | 0,370 | 0,251 |

« Vitesse » = facteur temps réel (55× signifie qu'une minute d'audio est transcrite en 1,1 s).

**Lecture importante.** Les clips sont synthétisés avec `espeak-ng`, ce qui donne une vérité
terrain exacte mais une voix hors distribution : les WER absolus sont bien plus mauvais que sur une
vraie voix. Ce qui transfère, c'est le **classement** des modèles et l'**écart entre les deux
dernières colonnes**.

Trois conclusions actionnables :

- **`distil-large-v3` est inutilisable en français** (WER 0,91 contre 0,25). C'est un modèle
  anglais uniquement — il est étiqueté comme tel dans l'interface.
- **`large-v3-turbo` en int8 est le meilleur rapport qualité/VRAM** : 1,6 Go, 55× temps réel, et
  un WER (0,268) à deux points de `large-v3` en float16 (0,251) qui coûte 2,6× plus de VRAM et
  transcrit 2,3× moins vite.
- **int8 n'est pas un compromis dégradant** ici : `large-v3` en int8 obtient le meilleur WER de
  tout le tableau (0,243) pour 2,3 Go.

## 2. L'effet du biais vocabulaire sur « Claude »

Neuf clips contiennent « Claude », « Claude Code » ou « Claude Desktop ». Le passage « biaisé »
injecte le même vocabulaire dans `hotwords` **et** `initial_prompt` (vérifié dans
`faster_whisper/transcribe.py:1542` : les deux atterrissent dans le même bloc `sot_prev` et se
cumulent).

| Modèle / précision | « Claude » correct, sans biais | avec biais | « cloud » à tort, sans → avec |
|---|---:|---:|---:|
| large-v3 / float16 | 6/9 | **9/9** | 2 → **0** |
| large-v3 / int8 | 6/9 | **9/9** | 2 → 0 |
| large-v3-turbo / int8 | 5/9 | **9/9** | 2 → 0 |
| medium / float16 | 6/9 | **9/9** | 1 → 0 |
| small / int8 | 5/9 | **9/9** | 2 → 0 |
| tiny / float16 | 1/9 | **9/9** | 4 → 0 |
| distil-large-v3 / float16 | 5/9 | 4/9 | 2 → 2 |

Le biais **n'a dégradé le WER global d'aucun modèle multilingue** : il l'améliore partout, de
0,04 à 0,20 point. Il est donc activé par défaut. Seul `distil-large-v3` régresse, ce qui est
cohérent avec son incapacité en français.

Vérification bout-en-bout sur la phrase réelle, avec `large-v3` en float16 :

```
Dicté     : Je vais demander à Claude Code de corriger le bug puis je pousse la branche sur GitHub.
Transcrit : Je vais demander à Claude Code de corriger le bug puis je pousse la manche sur GitHub.
```

« Claude Code » et « GitHub » passent (« manche » est un artefact de la voix synthétique).

## 3. Reformulation : latence et VRAM

24 appels par modèle sur 12 combinaisons format × langue, `temperature 0.2`, `num_ctx 8192`.
La VRAM est le delta mesuré sur la carte au chargement du modèle.

| Modèle | VRAM (ctx 8192) | VRAM (ctx 2048) | Latence médiane | p90 | Chargement à froid |
|---|---:|---:|---:|---:|---:|
| qwen3:1.7b | 2,68 Go | **1,94 Go** | **0,23 s** | 0,89 s | 0,9 s |
| granite4:micro | 3,30 Go | 2,40 Go | 0,42 s | 0,95 s | 1,0 s |
| qwen3.5:2b | 4,29 Go | 4,01 Go | 0,53 s | 1,70 s | 1,4 s |
| gemma3:4b-it-qat | 6,00 Go | 5,90 Go | 0,48 s | 1,91 s | 1,7 s |
| qwen3.5:4b | 6,20 Go | 5,84 Go | 0,67 s | 1,83 s | 1,8 s |
| **qwen3:8b** | 6,50 Go | 5,57 Go | 0,60 s | 2,42 s | 1,7 s |

Réduire le contexte de 8192 à 2048 économise peu (0,1 à 0,9 Go) : la place est prise par les
poids, pas par le cache d'attention. À n'utiliser que si les derniers 500 Mo manquent.

**Modèle écarté : `qwen3:4b`.** Son raisonnement fuit dans la réponse malgré `think: false` — les
appels prennent 9 à 28 s et le texte livré commence par « Okay, let's tackle this transcription
cleaning task ». C'est pourquoi l'application n'envoie `think: false` que si `/api/show` annonce la
capacité `thinking`, et nettoie malgré tout les blocs `<think>` en sortie.

## 4. Reformulation : fidélité

Pour un usage « je dicte et j'envoie tel quel », la fidélité passe avant l'élégance : une
information perdue se voit, une information **inventée** ne se voit pas.

### 4.1 Jugement à l'aveugle

12 dictées × 4 modèles anonymisés, notées par 6 jurés indépendants (2 par critère) qui ne
connaissaient pas le nom des modèles.

| Modèle | Fidélité /5 | Respect de la consigne /5 | Qualité de langue /5 |
|---|---:|---:|---:|
| qwen3:8b | **4,52** | 3,73 | 3,10 |
| qwen3.5:4b | 4,04 | **3,88** | **4,00** |
| gemma3:4b-it-qat | 2,82 | 3,59 | 3,85 |
| granite4:micro | 1,61 | 2,21 | 2,10 |

Fautes citées, en clair :

- **granite4:micro fabrique du faux.** « je pense que je vais pas pouvoir venir » devient « j'ai
  pas vraiment envie de venir » (impossibilité → mauvaise volonté), « jeudi » devient « le
  jeûne », « moi je m'occupe du front » devient « vous (front) », « il manque huit mille euros sur
  le lot deux » devient « sur l'ensemble du projet ». Déconseillé quelle que soit son empreinte.
- **gemma3:4b-it-qat invente des détails techniques** : le raccourci `Ctrl+Alt+E` répété quatre
  fois alors que la dictée disait « ctrl alt espace », « int8 » devenu « with 8GB », un numéro de
  facture fabriqué. En revanche c'est le seul japonais vraiment idiomatique du panel.
- **qwen3.5:4b** a le meilleur français (formules épistolaires, typographie, « 8 000 € »,
  « km/h × 100 ») mais réécrit un chiffre dicté à l'oral (« un giga six » → « 6 GB »), fausse un
  terme métier (« recette » → « réception du projet ») et a rendu un ticket **entièrement en
  français** alors que le japonais était demandé.
- **qwen3:8b** est le seul dont aucun juré ne cite une invention de chiffre, de nom ou de
  raccourci. Son défaut est inverse : il transforme trop peu (des « euh » subsistent, ponctuation
  parfois absente).

### 4.2 Test de pièges automatisé

Les fautes ci-dessus sont devenues un test reproductible : `tools/eval_models.py` rejoue 9 dictées
piégées et vérifie automatiquement la présence de ce qui doit survivre et l'absence de ce qui
trahit une invention. C'est l'outil à relancer avant d'adopter un nouveau modèle.

| Modèle | Score | Échecs restants |
|---|---:|---|
| **qwen3:8b** | **7/9** | « un giga six » perdu à la traduction (2 cas) |
| qwen3.5:2b | 5/9 | perd `little endian` et `SteamVR` ; ne traduit pas en japonais |
| qwen3:1.7b | 5/9 | perd les montants 340/280 ; perd `08 02` |
| qwen3.5:4b | 4/9 | invente `[votre nom]` ; « recette » → « réception » ; japonais rendu en français |
| gemma3:4b-it-qat | 4/9 | invente `Ctrl+Alt+E` ; perd `int8` et `1,6 Go` |
| granite4:micro | 2/9 | perd « jeudi »/« vendredi », invente « rendez-vous », perd « lot deux » |

```
$ python tools/eval_models.py qwen3:8b
=== qwen3:8b
    modalisation                 OK    1.89s
    chiffres_facture             OK    0.57s
    raccourci                    OK    1.37s
    specs_orales                 ÉCHEC  0.50s  → manque : un giga six / 1,6
    responsabilites              OK    0.60s
    identifiants                 OK    0.75s
    traduction_japonais          OK    1.61s
    traduction_anglais_chiffres  ÉCHEC  0.25s  → manque : 1.6 / one gigabyte six
    pas_de_bavardage             OK    0.21s
    → 7/9
```

**Conséquence retenue : `qwen3:8b` est le modèle par défaut.** Le contrôle de langue de sortie
avec seconde tentative (`sw/backends.reformat`) existe précisément à cause de l'échec japonais
observé sur les autres modèles.

### 4.3 Fidélité sur longue dictée

Dictée synthétique à contenu unique (aucune déduplication légitime possible), 60 « lignes », soit
1664 mots — environ 4 min 30 de parole. Vérification automatique de la survie des 60 montants et
des 60 sujets.

| Modèle | 281 mots | 832 mots | 1664 mots | Montants conservés | Sujets conservés |
|---|---:|---:|---:|---:|---:|
| **qwen3:8b** | 3,13 s | 7,47 s | 14,27 s | **60/60** | **60/60** |
| qwen3.5:4b | 2,99 s | 4,90 s | 11,89 s | 60/60 | 60/60 |
| granite4:micro | 4,90 s | 2,56 s | 5,34 s | 60/60 | 60/60 |
| gemma3:4b-it-qat | 3,23 s | 3,56 s | 5,83 s | 60/60 | 60/60 |

Aucun modèle ne tronque une longue dictée : la fidélité se joue sur les détails (§4.1, §4.2), pas
sur le volume. `qwen3:8b` est le plus lent des quatre parce qu'il est aussi le seul à réécrire les
1664 mots sans en condenser un seul (ratio de 1,00 ; les autres rendent 47 à 88 % du volume).

> Deux pièges de mesure rencontrés pendant ce benchmark, notés ici pour ne pas y retomber :
> une première version du test répétait les mêmes phrases, et les modèles avaient donc **raison**
> de dédupliquer ; une seconde version cherchait « 1025 » dans une sortie où qwen3.5:4b avait
> écrit « 1 025 » avec l'espace des milliers, conforme à la typographie française. Les deux
> « pertes d'information » étaient imaginaires.

## 5. Latence bout-en-bout selon la longueur de la dictée

Les deux étages ont été mesurés séparément, chacun sur son axe naturel : la transcription dépend de
la **durée de l'audio**, la reformulation du **nombre de mots**. Les additionner donne le temps
perçu entre le relâchement du raccourci et le texte collé.

**Étage 1 — transcription** (mesuré) :

| Dictée | Audio | `large-v3` float16 | `large-v3-turbo` int8 |
|---|---:|---:|---:|
| 1 ligne (18 mots) | 4,5 s | 0,22 s | 0,09 s |
| 3 lignes (53 mots) | 12,8 s | 0,52 s | 0,16 s |
| 10 lignes (166 mots) | 44,5 s | 1,97 s | 0,45 s |
| 30 lignes (498 mots) | 2 min 12 | 11,43 s | 1,24 s |
| 60 lignes (996 mots) | 4 min 24 | 14,69 s | 2,31 s |

**Étage 2 — reformulation `qwen3:8b`** (mesuré, contenu unique) : 0,19 s pour 18 mots, 0,39 s pour
53 mots, 3,13 s pour 281 mots, 7,47 s pour 832 mots, 14,27 s pour 1664 mots.

**Total, configuration par défaut** (`large-v3` float16 + `qwen3:8b`) : environ **0,4 s** pour une
phrase, **1 s** pour un paragraphe court, **5 s** pour 280 mots, **19 s** pour 500 mots et **29 s**
pour une dictée de quatre minutes. Avec `large-v3-turbo` en int8 à la place, le même palier de
quatre minutes tombe à **17 s** : la transcription passe de 14,7 s à 2,3 s.

La transcription domine sur les dictées longues avec `large-v3`, ce qui rend `large-v3-turbo` en
int8 intéressant même sur une grosse carte.

## 6. Configurations conseillées par carte graphique

Ces paliers sont ceux que l'onglet **Modèles** applique en un clic, après détection de la VRAM.

| VRAM | Transcription | Reformulation | Contexte | Total | Cohabitation |
|---|---|---|---:|---:|---|
| 4 Go | large-v3-turbo int8 | qwen3:1.7b | 2048 | 3,5 Go | non — décharger Whisper après 5 min |
| 6 Go | large-v3-turbo int8 | qwen3:1.7b | 8192 | 4,3 Go | oui |
| 8 Go | large-v3-turbo int8 | qwen3.5:2b | 8192 | 5,9 Go | oui |
| 12 Go | large-v3 int8 | qwen3:8b | 8192 | 8,8 Go | oui |
| 16 Go et + | large-v3 float16 | qwen3:8b | 8192 | 10,7 Go | oui |

Deux réglages servent les petites cartes :

- **« Décharger Whisper après N minutes »** libère la VRAM entre deux dictées, au prix d'un
  rechargement de 1 à 2 s à la dictée suivante.
- **`keep_alive`** côté Ollama fait la même chose pour le modèle de reformulation. `0` le décharge
  immédiatement, `30m` (défaut) le garde chaud une demi-heure.

Sous 6 Go, le modèle de reformulation disponible (`qwen3:1.7b`) perd des chiffres sur les dictées
longues : il est honnête de s'en servir pour des messages courts et de désactiver la reformulation
pour les dictées techniques.

## Reproduire

```bash
python tools/eval_models.py                  # test de pièges sur tous les modèles installés
python tools/e2e_check.py                    # chaîne complète : audio → texte → reformulation
python tools/e2e_check.py --model large-v3-turbo --compute int8 --llm qwen3:1.7b
python -m pytest tests/ -q                   # 119 tests (logique pure + sélecteur hors écran)
```

Les scripts de mesure de VRAM, de WER et de scaling utilisés pour produire ce document sont
volontairement hors du dépôt (ils téléchargent plusieurs gigaoctets de modèles) ; leur méthode est
décrite en tête de chaque tableau et `tools/eval_models.py` en reprend la partie reproductible.
