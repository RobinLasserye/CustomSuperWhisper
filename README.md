# SuperWhisper Custom

Dictée vocale locale pour Linux (KDE/Wayland) et Windows : un raccourci, on parle, le texte est
transcrit, mis en forme et collé dans l'application active. **Rien ne sort de la machine** —
Whisper tourne en local sur le GPU, la reformulation aussi, via Ollama.

```
Ctrl + Alt + Espace         dicter, puis appliquer le format par défaut
Ctrl + Alt + Maj + Espace   dicter, puis choisir le format et la langue dans un sélecteur
```

## Ce que ça fait

- **Transcription** locale par faster-whisper, sur GPU NVIDIA.
- **Vocabulaire dirigé** : les noms propres qu'on utilise vraiment (Claude Code, Ollama, PipeWire…)
  sont injectés dans le décodage. Mesuré : « Claude » passe de 6 clips sur 9 à **9 sur 9**, et les
  confusions « cloud » tombent à zéro ([benchmarks](docs/BENCHMARKS.md#2-leffet-du-biais-vocabulaire-sur--claude-)).
- **Corrections après coup** : un dictionnaire éditable (`motif => remplacement`, regex acceptées)
  rattrape ce que le décodage laisse passer.
- **Filtre d'hallucinations** : Whisper ayant appris sur des sous-titres, il produit parfois
  « Sous-titrage ST' 501 », « Merci d'avoir regardé cette vidéo » ou un crédit Amara.org sur du
  silence. Ces phrases sont retirées, sans toucher au reste du segment.
- **Neuf formats de sortie** : message, compact, WhatsApp/Messenger, mail formel, Slack, ticket
  GitHub, notes, instruction dev, traduction seule. Tous éditables, duplicables, et on peut en
  créer d'autres.
- **Traduction universelle** : n'importe quel format peut sortir dans n'importe quelle langue
  (15 proposées), dans le même appel — un mail formel en anglais, un ticket GitHub en japonais.
  La langue de sortie est vérifiée, avec une seconde tentative si le modèle l'a ignorée.
- **Gestionnaire de modèles** : la VRAM de la carte est détectée, une configuration est conseillée
  avec ses chiffres mesurés (vitesse, VRAM, qualité), et les modèles se téléchargent en un clic.

## Installation

```bash
git clone git@github.com:RobinLasserye/CustomSuperWhisper.git
cd CustomSuperWhisper
python install.py                # venv, dépendances, modèle Whisper, autostart
```

Pour la reformulation locale, il faut [Ollama](https://ollama.com) et un modèle :

```bash
ollama pull qwen3:8b             # défaut conseillé (6,5 Go de VRAM)
ollama pull qwen3:1.7b           # pour une carte de 4 à 6 Go
```

Ou bien depuis l'onglet **Modèles** des réglages, qui affiche les mesures et télécharge à votre
place.

## Réglages

Double-clic sur l'icône de la barre système (ou clic droit → Paramètres). Six onglets :

| Onglet | Contenu |
|---|---|
| **Transcription** | modèle, précision, langue dictée, GPU, microphone, déchargement après inactivité |
| **Vocabulaire** | biais de reconnaissance, dictionnaire de corrections, règle « cloud », champ d'essai en direct |
| **Nettoyage** | motifs d'hallucination, formules ambiguës et leurs seuils, effondrement des répétitions |
| **Reformulation** | backend, hôte et modèle Ollama, contexte, format et langue par défaut, édition des consignes |
| **Modèles** | VRAM détectée, configuration conseillée, tableaux de performance, téléchargements |
| **Général** | collage automatique, contenu du sélecteur |

Le clic droit sur l'icône permet aussi de changer **le format par défaut** et **la langue de
sortie** sans ouvrir les réglages.

## Choisir ses modèles selon la carte graphique

Ces paliers sont ceux que l'onglet **Modèles** applique en un clic. Détail et méthode dans
[docs/BENCHMARKS.md](docs/BENCHMARKS.md).

| VRAM | Transcription | Reformulation | Total | Remarque |
|---|---|---|---:|---|
| 4 Go | large-v3-turbo int8 | qwen3:1.7b (ctx 2048) | 3,5 Go | décharger Whisper entre deux dictées |
| 6 Go | large-v3-turbo int8 | qwen3:1.7b | 4,3 Go | messages courts de préférence |
| 8 Go | large-v3-turbo int8 | qwen3.5:2b | 5,9 Go | bon compromis général |
| 12 Go | large-v3 int8 | qwen3:8b | 8,8 Go | meilleur taux d'erreur mesuré |
| 16 Go et + | large-v3 float16 | qwen3:8b | 10,7 Go | qualité maximale |

Deux choses qui surprennent et que les mesures montrent :

- **`large-v3-turbo` en int8** ne coûte que 1,6 Go, transcrit 55× plus vite que le temps réel, et
  son taux d'erreur (0,268) est à deux points de `large-v3` en float16 qui coûte 2,6× plus.
- **`distil-large-v3` est un modèle anglais** : 0,91 de taux d'erreur en français contre 0,25. Il
  est signalé comme tel dans l'interface.

## Choisir son modèle de reformulation

Pour un texte qu'on envoie tel quel, ce qui compte n'est pas l'élégance mais la fidélité : une
information perdue se voit, une information **inventée** ne se voit pas. Les modèles ont donc été
jugés à l'aveugle par six jurés, puis soumis à un test de pièges automatisé.

| Modèle | Pièges | VRAM | Latence | Verdict |
|---|---:|---:|---:|---|
| **qwen3:8b** | **7/9** | 6,5 Go | 0,60 s | défaut : aucune invention relevée, seul à traduire correctement |
| qwen3.5:2b | 5/9 | 4,3 Go | 0,53 s | correct pour 8 Go, perd des identifiants techniques |
| qwen3:1.7b | 5/9 | 2,7 Go | 0,23 s | poids plume, à réserver aux messages courts |
| qwen3.5:4b | 4/9 | 6,2 Go | 0,67 s | meilleur français, mais réécrit des chiffres dictés |
| gemma3:4b-it-qat | 4/9 | 6,0 Go | 0,48 s | invente des détails (un raccourci clavier jamais dicté) |
| granite4:micro | 2/9 | 3,3 Go | 0,42 s | **à éviter** : « jeudi » → « le jeûne », inversions de sens |

Pour évaluer un modèle qui n'est pas dans cette liste :

```bash
python tools/eval_models.py mon-modele:tag
```

## Développement

```
superwhisper.py         point d'entrée : contrôleur, raccourcis, orchestration
sw/runtime.py           plateforme, chemins, amorçage CUDA (ré-exec)
sw/config.py            schéma, chargement, migration
sw/vocabulary.py        biais Whisper et moteur de corrections
sw/artifacts.py         filtre des hallucinations
sw/presets.py           formats et traduction
sw/backends.py          Ollama, Claude Code, contrôle de la langue de sortie
sw/langcheck.py         heuristique « ce texte est-il dans la bonne langue ? »
sw/models_catalog.py    mesures, recommandations VRAM, téléchargements
sw/transcriber.py       faster-whisper
sw/audio.py             enregistrement
sw/clipboard.py         presse-papier et collage
sw/instance.py          instance unique
sw/hardware.py          GPU et entrées audio
sw/ui/                  overlay, réglages, sélecteur, onglet modèles
tools/eval_models.py    test de pièges (fidélité d'un modèle)
tools/e2e_check.py      chaîne complète audio → texte → reformulation
tests/                  119 tests pytest (logique pure + sélecteur hors écran)
```

```bash
.venv/bin/pip install pytest     # une seule fois
.venv/bin/python -m pytest tests/ -q   # logique pure, aucun GPU requis
.venv/bin/python tools/e2e_check.py    # chaîne complète (GPU + Ollama requis)
```

## Dépannage

**« Library libcublas.so.12 is not found »** — ne devrait plus arriver : `sw/runtime.py` se
ré-exécute avec le bon `LD_LIBRARY_PATH` (le loader dynamique ne relit pas cette variable après le
démarrage du processus, patcher `os.environ` ne suffit donc pas). Si le message persiste,
`nvidia-cublas-cu12` et `nvidia-cudnn-cu12` sont probablement absents du venv.

**Enregistrement vide, pas d'animation** — le microphone doit rester sur `default`. Les index
numériques de PortAudio changent au redémarrage et finissent par désigner un périphérique matériel
qui refuse le 16 kHz (`PaErrorCode -9997`).

**Le texte n'est pas collé après le sélecteur** — le sélecteur prend le focus le temps du choix. Si
le gestionnaire de fenêtres ne le rend pas, décochez « Coller aussi quand le sélecteur a été
utilisé » dans l'onglet Général : le texte reste dans le presse-papier.

**Reformulation qui échoue** — le texte brut est copié **avant** l'appel au modèle : en cas
d'échec, rien n'est perdu et l'overlay affiche la cause (Ollama éteint, modèle absent, délai
dépassé).

**Sur Wayland**, l'overlay utilise le scripting KWin pour rester au-dessus des autres fenêtres, et
`wl-copy` pour le presse-papier (`QClipboard` n'est pas fiable sans focus).
