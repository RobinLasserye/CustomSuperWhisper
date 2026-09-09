# Historique local d’évaluation

Dans **Paramètres → Général → Historique local d’évaluation**, activer la conservation
puis, si souhaité, l’audio. Ces options sont désactivées par défaut pour les nouvelles installations.
Désactiver la collecte conserve les données déjà présentes. Aucun entraînement, optimiseur DSPy,
export ou envoi réseau n’est déclenché par cette fonctionnalité.

**Exécutions locales** affiche les 200 dernières exécutions ; **Charger davantage** en affiche
200 de plus. Les autres restent sur disque. **Bilan** calcule nombre de runs, avertissements,
évaluations manuelles, taille de la base et de l’audio, médianes et P95 des durées disponibles.
Les chiffres agrègent les modèles et pipelines : ce n’est pas une comparaison contrôlée.
La vue affiche les mesures et paramètres détaillés de chaque exécution et appel.

## Où vont les données ?

- Linux : `$XDG_STATE_HOME/superwhisper-custom/history.sqlite3`, sinon
  `~/.local/state/superwhisper-custom/history.sqlite3`.
- Windows : `%LOCALAPPDATA%/superwhisper-custom/history.sqlite3`.
- Audio compressé dans une table de la même base, associé à l’identifiant de l’exécution.
- Le stockage refuse un emplacement situé dans un dépôt Git. Le dépôt ignore aussi les bases,
  fichiers audio et dossiers privés usuels. Seuls code, documentation et tests synthétiques sont
  destinés à GitHub. Ne pas déplacer manuellement une base dans un autre dossier synchronisé.
- Linux : dossier privé (0700) et base (0600). Ce n’est **pas du chiffrement** ; les sauvegardes
  système et les administrateurs de la machine restent hors du contrôle de l’application.
- Suppression par exécution ou de toute la base via la vue (confirmation). `secure_delete`
  efface les contenus supprimés des pages SQLite ; il ne garantit pas l’effacement physique de
  copies sur SSD, instantanés ou sauvegardes. L’espace SQLite peut être réutilisé sans réduction
  immédiate de la taille du fichier.

## Données enregistrées

- Identifiant, date UTC, pipeline, format, langue, backend et modèle.
- Texte source du pipeline, résultat, avertissements, étapes, décisions/arêtes, tentatives,
  consigne exacte à chaque appel, temps de chaque étape.
- Sortie Whisper avant nettoyage, segments (texte, début/fin, log-probabilité moyenne,
  probabilité de non-parole et ratio de compression), langue détectée et probabilité,
  durée après VAD, artefacts retirés : lorsqu’ils sont fournis par le transcripteur.
- Durée audio, durée de transcription incluant le chargement éventuel, nettoyage,
  pipeline texte incluant la compilation du graphe et arrêt micro → résultat. Cette dernière
  inclut l’attente humaine du sélecteur quand il est utilisé ; elle exclut le collage réel.
- Compteurs Ollama disponibles : tokens en entrée/sortie, temps total serveur, chargement,
  évaluation du prompt, génération et tokens/seconde. Chaque tentative a ses propres compteurs.
  Une mesure absente reste absente. Les durées serveur en nanosecondes deviennent des millisecondes.
- Temps CPU du thread client Python : **pas** le temps CPU/GPU du serveur Ollama, ni une mesure
  d’énergie. Pas de mesure de VRAM, d’utilisation GPU, de FLOPs ou de consommation électrique.
- Paramètres d’inférence et instantané des réglages de transcription (vocabulaire, corrections,
  filtres et seuils compris), versions des bibliothèques, Python et OS.
  Aucun secret d’environnement, clé API ou configuration complète n’est collecté.
- Évaluation manuelle, sortie finale attendue et transcription mot à mot de référence, facultatives. Une langue plausible ou l’absence d’avertissement
  ne prouve pas que le sens est préservé. Les probabilités Whisper ne sont pas des scores de qualité.

Le pipeline historique et le backend Claude conservent leurs destinations existantes :
l’historique local ne change pas la destination des requêtes du modèle. Le chemin Ollama
LangGraph conserve ses restrictions locales et la désactivation du tracing.

## Audio : deux compromis explicites

- **Original exact** : mono float32 little-endian à la fréquence de capture (16 kHz), compressé
  avec zlib. Les bits des échantillons sont préservés. Ce n’est pas FLAC, et ce n’est pas la
  compression audio la plus compacte. Taille brute : 230,4 Mo/h ; gain dépendant du signal.
- **Opus voix 32 kb/s** : Ogg/Opus via PyAV (déjà utilisé par faster-whisper), avec perte,
  environ 14,4 Mo/h plus conteneur à débit moyen de 32 kb/s. Le débit peut varier. Le codec et
  les tailles effectives sont enregistrés. Les tests vérifient encodage, décodage et durée sur
  audio synthétique ; ils ne démontrent pas l’équivalence des transcriptions sur voix réelle.

**Écouter l’audio** / **Arrêter l’écoute** permettent de réécouter une dictée. Éviter de lancer
une nouvelle dictée pendant la lecture, pour ne pas réenregistrer le haut-parleur.
Pour évaluer l’impact d’Opus sur Whisper, comparer les mêmes enregistrements avant/après
compression sur un ensemble de références corrigées, distinct du jeu utilisé pour optimiser.
L’option audio est séparée ; changer le codec ne reconvertit pas les anciennes données.

La collecte reste illimitée jusqu’à suppression : surveiller la taille dans **Bilan**.
Aucune donnée historique de l’ancienne session en mémoire n’est récupérable après fermeture.

## Vérification et analyse locale

```bash
QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest -q
.venv/bin/python -m sw.history
```

La seconde commande produit uniquement des statistiques agrégées, sans transcriptions ni audio.
Les tests de stockage utilisent des données synthétiques et des répertoires temporaires ; ils
n’accèdent pas à l’historique personnel. Un stockage indisponible ne doit jamais empêcher de
livrer la dictée. La vue indique alors que le rapport n’a pas été enregistré.
