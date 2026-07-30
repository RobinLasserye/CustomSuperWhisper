# SuperWhisper Custom — reformulation locale, biais vocabulaire et traduction

Date : 2026-07-30
Statut : validé par le PO (Robin), en implémentation

## Problèmes traités

1. **« Claude » transcrit « cloud »** — les noms propres techniques sont mal reconnus, ce qui
   oblige à corriger à la main plusieurs fois par jour.
2. **La reformulation passe par l'API Claude** (`claude -p` en subprocess) : dépendance réseau,
   latence de plusieurs secondes, et le texte dicté sort de la machine.
3. **Un seul format à la fois**, choisi à l'avance dans les réglages : impossible de décider
   « ça, finalement, en mail formel » après avoir parlé.
4. **Pas de traduction** : il faut repasser par un autre outil.

## Objectifs

- Reconnaissance fiable du vocabulaire technique, sans casser le reste de la transcription.
- Reformulation 100 % locale, légère, rapide, et surtout fidèle (rien perdu, rien inventé).
- Choix du format après avoir parlé, sans ralentir le chemin rapide.
- Traduction disponible sur **tous** les formats, pas comme un mode à part.
- Tenir sur une petite carte graphique : la VRAM nécessaire doit être mesurée, pas devinée.

## Non-objectifs

- Pas de serveur, pas de daemon supplémentaire : Ollama est déjà là.
- Pas de refonte de l'enregistrement audio ni de l'overlay de spectre (ça marche).
- Pas de portage Windows testé dans ce lot (le code reste cross-platform, non vérifié).

## Architecture

Le fichier unique de 1362 lignes devient un paquet, le point d'entrée garde son nom (l'autostart
et `install.py` ne changent pas) :

| Module | Responsabilité |
|---|---|
| `superwhisper.py` | point d'entrée, contrôleur, raccourcis, orchestration des threads |
| `sw/runtime.py` | plateforme, chemins de config, découverte des libs CUDA + **ré-exec** |
| `sw/config.py` | schéma par défaut, chargement/sauvegarde, **migration** des clés `claude_*` |
| `sw/vocabulary.py` | construction `hotwords`/`initial_prompt`, moteur de corrections |
| `sw/presets.py` | formats livrés, langues, assemblage du prompt système |
| `sw/backends.py` | `OllamaBackend`, `ClaudeCliBackend`, nettoyage de sortie |
| `sw/transcriber.py` | faster-whisper, chargement/déchargement |
| `sw/audio.py` | enregistrement sounddevice |
| `sw/clipboard.py` | presse-papier et collage automatique |
| `sw/instance.py` | instance unique et IPC (SIGUSR1 / event nommé) |
| `sw/ui/overlay.py` | overlay et spectre |
| `sw/ui/settings.py` | fenêtre de réglages (onglets) |
| `sw/ui/picker.py` | sélecteur de format et de langue |

### Correctif CUDA (découvert pendant le benchmark)

Le bloc en tête de `superwhisper.py` qui pousse `LD_LIBRARY_PATH` dans `os.environ` **n'a aucun
effet** : le loader dynamique ne relit pas la variable après le démarrage du processus. L'app ne
fonctionne aujourd'hui que parce que le `.desktop` d'autostart pose la variable ; lancée depuis le
menu KDE (`~/.local/share/applications/superwhisper-custom.desktop`, qui ne la pose pas), la
transcription échoue sur `Library libcublas.so.12 is not found`.

Correctif : `sw/runtime.ensure_cuda_libs()` se **ré-exécute** (`os.execve`) avec la variable
correctement positionnée, une seule fois (garde `SW_CUDA_REEXEC`). Sous Windows,
`os.add_dll_directory()` (le patch de `PATH` ne suffit pas depuis Python 3.8).

## 1. Reconnaissance du vocabulaire — trois couches indépendantes

**Couche A — biais Whisper.** Une liste de termes (`vocabulary`) alimente `hotwords` *et*
`initial_prompt`. Vérifié dans `faster_whisper/transcribe.py:1542` : les deux atterrissent dans le
même bloc `sot_prev` et se cumulent (les hotwords ne sont ignorés que si `prefix` est utilisé, ce
que l'app ne fait pas). Activable/désactivable ; l'effet réel sur le WER est mesuré au benchmark.

**Couche B — dictionnaire de corrections.** Liste ordonnée de règles appliquées après
transcription, éditées en texte (`motif => remplacement`, préfixe `re:` pour du regex). Motifs
littéraux compilés avec des gardes de mot (`(?<!\w)…(?!\w)`) et espaces souples, insensibles à la
casse. Ordre significatif : les règles contextuelles passent avant les règles nues.

**Couche C — règle « cloud ».** Robin dit « cloud » au sens hébergement extrêmement rarement :
`cloud → Claude` est donc **active par défaut**, mais neutralisée sur les zones couvertes par une
liste d'exceptions (`dans le cloud`, `cloud public`, `cloud AWS`, `hébergement cloud`, …). Les
exceptions sont évaluées **après** les règles contextuelles, donc `cloud code` est déjà devenu
`Claude Code` quand l'exception `le cloud` est testée.

Les réglages contiennent un champ de test : on tape une phrase, on voit le résultat corrigé.

## 2. Reformulation locale

`OllamaBackend` sur `127.0.0.1:11434`, en `urllib` de la bibliothèque standard (aucune dépendance
ajoutée) :

- `POST /api/chat`, `stream:false`, `temperature 0.2`, `top_p 0.9`, `num_ctx 8192`
- `keep_alive` configurable (défaut `30m`) pour éviter le rechargement à chaque dictée
- `think:false` **uniquement** si `/api/show` annonce la capacité `thinking` (sinon Ollama refuse)
- timeout configurable (défaut 60 s), erreurs traduites en messages lisibles dans l'overlay

`ClaudeCliBackend` conserve le comportement actuel et reste sélectionnable **par mode**, backend
par défaut `ollama`. Ollama éteint ⇒ texte brut collé + overlay orange nommant la cause.

Nettoyage de sortie testé : blocs `<think>`, fence markdown enveloppant la totalité, guillemets
encadrants, ligne de préambule (« Voici le message nettoyé : »).

## 3. Formats livrés

`message`, `compact`, `dev` gardent leurs identifiants — les prompts personnalisés déjà présents
dans la config de Robin sont donc conservés tels quels. Nouveaux : `mail` (mail formel),
`chat` (WhatsApp/Messenger), `github` (ticket Markdown), `slack`, `notes` (compte-rendu),
`translate` (traduction seule). Les formats personnalisés existants restent créables, éditables,
supprimables, et gagnent un backend par mode.

## 4. Traduction universelle

La langue cible est **orthogonale** au format : n'importe quel format peut sortir en n'importe
quelle langue. Elle est appliquée dans le **même appel** LLM (un seul aller-retour, la mise en
forme n'est pas cassée par une seconde passe) via un bloc de directive ajouté au prompt système.
Réglable comme défaut global, par mode, et à la volée dans le sélecteur. `translate` sans langue
cible retombe sur l'anglais.

## 5. Sélecteur de format (hybride)

- `Ctrl+Alt+Espace` : inchangé — mode et langue par défaut, aucune question posée
- `Ctrl+Alt+Maj+Espace` : le sélecteur s'ouvre **après** la transcription, avec l'aperçu du texte
  brut, les formats en `1`-`9`, la langue cible sur une ligne dédiée, `Entrée` applique,
  `Échap` colle le texte brut
- Tray : sous-menus « Format par défaut » et « Langue de sortie » en cases radio

Le sélecteur est une vraie fenêtre qui prend le focus : lire des touches sans voler le focus
injecterait les chiffres dans l'application cible. Le collage automatique attend sa fermeture
(délai porté à 0,35 s) et reste désactivable (`auto_paste_after_picker`) ; le texte est de toute
façon toujours dans le presse-papier.

## 6. Économie de VRAM (objectif petite carte)

- `ollama_keep_alive` court libère le LLM entre deux dictées
- `whisper_idle_unload_min` (0 = jamais) décharge le modèle Whisper après inactivité
- La documentation livre une matrice de configurations mesurées par palier de VRAM

## 7. Benchmarks livrés (`docs/BENCHMARKS.md`)

1. **Transcription** : modèles × précision → latence, RTF, VRAM mesurée en sous-processus isolé,
   WER sur 30 clips à vérité terrain, et gain du biais vocabulaire sur « Claude ».
2. **Reformulation et traduction** : modèles candidats × formats × langues, latence, VRAM, avec
   jugement à l'aveugle multi-critères (fidélité / respect du format / qualité de langue).
3. **Combiné, par taille de texte** : latence bout-en-bout selon le nombre de lignes dictées.
4. **Matrice VRAM** : configuration conseillée pour 4, 6, 8, 12 et 32 Go.

Limite assumée : les clips de test sont synthétisés (`espeak-ng`), ce qui donne une vérité terrain
exacte et un classement fiable entre modèles, mais pas un WER absolu représentatif d'une vraie
voix. La validation finale du correctif « Claude » se fait sur un enregistrement réel de Robin.

## 8. Tests

`pytest` sur la logique pure : moteur de corrections (dont chaque cas « vrai cloud »),
construction du biais, résolution des formats et des overrides, assemblage du prompt de
traduction, migration de la config réelle de Robin, nettoyage de sortie LLM, payload et mapping
d'erreurs Ollama (`urlopen` simulé). Un test d'intégration bout-en-bout (WAV → transcription →
corrections → Ollama) est marqué à part car il exige GPU et Ollama.

## 9. Déploiement

Arrêt de l'instance avant toute écriture de config (l'app réécrit le fichier), relance hors shell
via `systemd-run --user` (le `pgrep -f superwhisper.py` s'auto-matche), vérification de l'autostart,
retrait du `CUDA_VISIBLE_DEVICES=1` codé en dur qui contredit la config, ajout de
`LD_LIBRARY_PATH` devenu inutile mais conservé sans risque, puis commit et push sur `origin`.
