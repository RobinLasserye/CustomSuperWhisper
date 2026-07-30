"""Backends de reformulation : Ollama en local, Claude Code en option.

`ReformatError` porte un message déjà rédigé pour l'overlay : l'appelant n'a pas à traduire des
codes d'erreur. Aucune dépendance en dehors de la bibliothèque standard.
"""

import json
import re
import subprocess
import urllib.error
import urllib.request

from . import langcheck, models_catalog
from .runtime import CLAUDE_BIN, log

# Capacités partagées entre instances : un nouveau backend est construit à chaque dictée, sans ce
# cache /api/show serait interrogé chaque fois.
_CAPABILITIES_CACHE = {}

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE = re.compile(r"^```[a-zA-Z0-9_+-]*\s*\n(.*)\n```$", re.DOTALL)
_FENCE_LINE = re.compile(r"^```", re.MULTILINE)
# Volontairement étroit : « Voici les chiffres du trimestre : » est un contenu légitime, seule
# une annonce du résultat lui-même doit disparaître.
_PREAMBLE_LINE = re.compile(
    r"^(voici|voilà|here is|here's)\s+(le|la|les|the|your|ton|votre)?\s*"
    r"(texte|message|résultat|resultat|result|ticket|mail|e-mail|email|note|notes|traduction|"
    r"translation|version)[^\n]{0,40}:\s*\n+",
    re.IGNORECASE)
_QUOTE_PAIRS = (("«", "»"), ('"', '"'), ("“", "”"), ("'", "'"), ("‘", "’"))


class ReformatError(Exception):
    """Erreur de reformulation, avec un message affichable tel quel."""


def _normalize_host(host):
    """« 127.0.0.1:11434 » saisi sans schéma reste utilisable."""
    host = (host or "").strip().rstrip("/")
    if not host:
        return "http://127.0.0.1:11434"
    if "://" not in host:
        host = "http://" + host
    return host


def clean_output(text):
    """Retire ce que les modèles ajoutent malgré les consignes."""
    if not text:
        return ""

    # Raisonnement laissé dans la réponse
    text = _THINK_BLOCK.sub("", text)
    if "</think>" in text.lower():
        index = text.lower().rfind("</think>")
        text = text[index + len("</think>"):]
    if "<think>" in text.lower():
        text = text[:text.lower().index("<think>")]

    text = text.strip()

    # Bloc de code enveloppant la totalité de la réponse. On ne le retire que s'il n'y a
    # exactement que ces deux délimiteurs : un ticket qui commence et finit par un vrai bloc de
    # code en contient davantage, et le découper le mutilerait.
    if len(_FENCE_LINE.findall(text)) == 2:
        match = _FENCE.match(text)
        if match:
            text = match.group(1).strip()

    # Ligne de préambule (« Voici le message nettoyé : »)
    text = _PREAMBLE_LINE.sub("", text, count=1).strip()

    # Guillemets encadrant l'intégralité du texte
    for opening, closing in _QUOTE_PAIRS:
        if len(text) > 2 and text.startswith(opening) and text.endswith(closing):
            inner = text[len(opening):-len(closing)]
            if opening not in inner and closing not in inner:
                text = inner.strip()
                break

    return text.strip()


# ─── Ollama ───────────────────────────────────────────────────────────────────

class OllamaBackend:
    """Client minimal de l'API Ollama locale."""

    name = "ollama"
    label = "Ollama (local)"

    def __init__(self, host="http://127.0.0.1:11434", model="qwen3.5:4b", keep_alive="30m",
                 timeout=60, num_ctx=8192, temperature=0.2):
        self.host = _normalize_host(host)
        self.model = model
        self.keep_alive = keep_alive
        self.timeout = timeout
        self.num_ctx = num_ctx
        self.temperature = temperature
        self._capabilities = {}

    @classmethod
    def from_config(cls, config):
        return cls(host=config.get("ollama_host"),
                   model=config.get("ollama_model"),
                   keep_alive=config.get("ollama_keep_alive", "30m"),
                   timeout=config.get("ollama_timeout_s", 60),
                   num_ctx=config.get("ollama_num_ctx", 8192),
                   temperature=config.get("ollama_temperature", 0.2))

    # — HTTP —

    def _post(self, path, payload, timeout=None):
        request = urllib.request.Request(
            self.host + path, data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(request, timeout=timeout or self.timeout) as response:
            return json.loads(response.read().decode())

    def _get(self, path, timeout=5):
        with urllib.request.urlopen(self.host + path, timeout=timeout) as response:
            return json.loads(response.read().decode())

    # — État —

    # Les sondes appelées depuis l'interface ont un délai court : la fenêtre de réglages ne doit
    # pas geler plusieurs secondes quand l'hôte configuré ne répond pas.
    UI_TIMEOUT = 1.5

    def list_models(self):
        try:
            data = self._get("/api/tags", timeout=self.UI_TIMEOUT)
            return sorted(str(entry["name"]) for entry in data["models"])
        except Exception:
            return []                     # forme inattendue comprise : la fenêtre doit s'ouvrir

    def is_running(self):
        try:
            self._get("/api/tags", timeout=self.UI_TIMEOUT)
            return True
        except Exception:
            return False

    def capabilities(self, model=None):
        """Capacités annoncées par Ollama, mises en cache pour tout le processus."""
        model = model or self.model
        for cache in (self._capabilities, _CAPABILITIES_CACHE):
            if model in cache:
                return cache[model]
        try:
            data = self._post("/api/show", {"model": model}, timeout=10)
            capabilities = list(data.get("capabilities") or [])
        except Exception as exc:
            # Sans réponse, on retombe sur ce que le catalogue sait du modèle : conclure « pas de
            # raisonnement » laisserait le mode réflexion actif, avec des réponses de 30 s dont le
            # raisonnement fuit dans le texte livré.
            known = models_catalog.LLM_MODELS.get(model, {})
            capabilities = ["thinking"] if known.get("thinking") else []
            log(f"Ollama : capacités de {model} indisponibles ({exc}) — "
                f"repli sur le catalogue ({capabilities or 'aucune'})")
            self._capabilities[model] = capabilities
            return capabilities
        self._capabilities[model] = capabilities
        _CAPABILITIES_CACHE[model] = capabilities
        return capabilities

    def supports_thinking(self, model=None):
        return "thinking" in self.capabilities(model)

    def warm_up(self):
        """Charge le modèle en VRAM sans produire de texte, pour que la première dictée soit
        aussi rapide que les suivantes."""
        try:
            self._post("/api/chat", {
                "model": self.model, "messages": [{"role": "user", "content": "ok"}],
                "stream": False, "keep_alive": self.keep_alive,
                "options": {"num_predict": 1, "num_ctx": self.num_ctx},
            }, timeout=180)
            return True
        except Exception as exc:
            log(f"Ollama : préchargement impossible ({exc})")
            return False

    # — Reformulation —

    def reformat(self, text, system_prompt):
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system_prompt},
                         {"role": "user", "content": text}],
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {"temperature": self.temperature, "top_p": 0.9, "num_ctx": self.num_ctx},
        }
        if self.supports_thinking():
            payload["think"] = False           # refusé par Ollama si le modèle ne sait pas penser

        try:
            data = self._post("/api/chat", payload)
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = json.loads(exc.read().decode()).get("error", "")
            except Exception:
                pass
            if "not found" in detail.lower() or exc.code == 404:
                raise ReformatError(
                    f"Modèle « {self.model} » absent d'Ollama — texte brut collé") from exc
            raise ReformatError(f"Ollama a répondu {exc.code} — texte brut collé") from exc
        except urllib.error.URLError as exc:
            raise ReformatError("Ollama injoignable — texte brut collé") from exc
        except TimeoutError as exc:
            raise ReformatError(
                f"Ollama : délai de {self.timeout} s dépassé — texte brut collé") from exc
        except OSError as exc:
            raise ReformatError(f"Ollama : erreur réseau ({exc}) — texte brut collé") from exc
        except ValueError as exc:
            raise ReformatError("Ollama : réponse illisible — texte brut collé") from exc

        try:
            content = clean_output((data.get("message") or {}).get("content", ""))
        except AttributeError as exc:
            raise ReformatError("Ollama : réponse de forme inattendue — texte brut collé") from exc
        if not content:
            raise ReformatError("Le modèle local a renvoyé une réponse vide — texte brut collé")
        return content


# ─── Claude Code (optionnel) ──────────────────────────────────────────────────

class ClaudeCliBackend:
    """Reformulation par le CLI Claude Code. Conservé en option : ce n'est plus le défaut."""

    name = "claude"
    label = "Claude Code (réseau)"

    def __init__(self, binary=CLAUDE_BIN, timeout=60):
        self.binary = binary
        self.timeout = timeout

    @classmethod
    def from_config(cls, config):
        return cls(timeout=config.get("claude_timeout_s", 60))

    def is_running(self):
        import os
        return self.binary == "claude" or os.path.exists(self.binary)

    def reformat(self, text, system_prompt):
        try:
            result = subprocess.run(
                [self.binary, "-p", system_prompt, "--tools", ""],
                input=text, capture_output=True, text=True, timeout=self.timeout)
        except subprocess.TimeoutExpired as exc:
            raise ReformatError(
                f"Claude : délai de {self.timeout} s dépassé — texte brut collé") from exc
        except FileNotFoundError as exc:
            raise ReformatError("Claude introuvable — texte brut collé") from exc
        except OSError as exc:
            raise ReformatError(f"Claude : erreur ({exc}) — texte brut collé") from exc

        if result.returncode != 0 or not result.stdout.strip():
            detail = result.stderr.strip()[:120] or f"code {result.returncode}"
            raise ReformatError(f"Claude a échoué ({detail}) — texte brut collé")

        content = clean_output(result.stdout)
        if not content:
            raise ReformatError("Claude a renvoyé une réponse vide — texte brut collé")
        return content


BACKENDS = {"ollama": OllamaBackend, "claude": ClaudeCliBackend}


def build_backend(config, name):
    return BACKENDS.get(name, OllamaBackend).from_config(config)


# ─── Reformulation avec contrôle de la langue de sortie ───────────────────────

RETRY_DIRECTIVE = (
    "\n\nATTENTION : la réponse précédente n'était pas dans la langue demandée. "
    "Écris TOUT le texte dans la langue de sortie exigée ci-dessus, sans exception, "
    "titres et libellés compris. N'écris rien en français."
)


def reformat(backend, text, system_prompt, target_language=None):
    """Reformule, et si une langue cible est demandée, vérifie qu'elle a été respectée.

    Retourne (texte, avertissement_ou_None). Une seule seconde tentative est faite : elle corrige
    le cas mesuré au benchmark (ticket rendu en français alors que le japonais était demandé) sans
    doubler systématiquement la latence.
    """
    result = backend.reformat(text, system_prompt)

    if not target_language or target_language == "none":
        return result, None

    verdict = langcheck.looks_like(result, target_language)
    if verdict is not False:
        return result, None

    log(f"langue de sortie non respectée ({target_language}) — seconde tentative")
    try:
        retry = backend.reformat(text, system_prompt + RETRY_DIRECTIVE)
    except ReformatError:
        return result, "Traduction incertaine — vérifie la langue"

    if langcheck.looks_like(retry, target_language) is False:
        return retry, "Traduction incertaine — vérifie la langue"
    return retry, None
