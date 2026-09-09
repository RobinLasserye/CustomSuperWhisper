"""LangChain's official Ollama integration, restricted to local inference."""
from ipaddress import ip_address
from urllib.parse import urlsplit

from .backends import OllamaBackend, ReformatError, clean_output


def validate_local(config):
    from .backends import _normalize_host
    host = _normalize_host(config.get("ollama_host"))
    parsed = urlsplit(host)
    try:
        loopback = parsed.hostname == "localhost" or ip_address(parsed.hostname).is_loopback
    except ValueError:
        loopback = False
    if parsed.scheme not in ("http", "https") or not loopback or parsed.username or parsed.password:
        raise ReformatError("LangGraph requiert un hôte Ollama local — texte brut conservé")
    if "cloud" in config.get("ollama_model", "").lower():
        raise ReformatError("Choisis un modèle Ollama local — texte brut conservé")
    return host


class LangChainOllamaBackend(OllamaBackend):
    def capabilities(self, model=None):
        # The original urllib probe inherits proxy settings. This local path must
        # not, and must reject cloud aliases before sending any dictation text.
        import httpx
        validate_local({"ollama_host": self.host, "ollama_model": model or self.model})
        response = httpx.post(self.host + "/api/show", json={"model": model or self.model},
                              timeout=self.timeout, trust_env=False, follow_redirects=False)
        response.raise_for_status()
        data = response.json()
        if data.get("remote_model") or data.get("remote_host"):
            raise ReformatError("Ce modèle Ollama utilise le cloud — texte brut conservé")
        return data.get("capabilities") or []

    def reformat(self, text, system_prompt):
        # Lazy imports allow the historical pipeline to work without these packages.
        from langchain_ollama import ChatOllama
        from langsmith import tracing_context

        self.last_metrics = {}
        validate_local({"ollama_host": self.host, "ollama_model": self.model})
        try:
            with tracing_context(enabled=False):
                model = ChatOllama(
                    base_url=self.host, model=self.model, temperature=self.temperature,
                    top_p=0.9, num_ctx=self.num_ctx, keep_alive=self.keep_alive,
                    reasoning=False if self.supports_thinking() else None,
                    client_kwargs={"timeout": self.timeout, "trust_env": False,
                                   "follow_redirects": False},
                )
                response = model.invoke([("system", system_prompt), ("human", text)],
                                        config={"callbacks": []})
                from .metrics import model_metrics
                self.last_metrics = model_metrics(response.response_metadata)
                result = clean_output(response.content)
            if not result:
                raise ReformatError("Le modèle local a renvoyé une réponse vide — texte brut conservé")
            return result
        except ReformatError:
            raise
        except Exception as exc:
            raise ReformatError("Ollama via LangChain indisponible — texte brut conservé") from exc
