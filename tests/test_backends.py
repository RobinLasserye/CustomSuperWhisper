"""Nettoyage des sorties de modèle, appel Ollama, et contrôle de la langue."""

import json
import urllib.error

import pytest

from sw import backends, langcheck


# ─── Nettoyage ────────────────────────────────────────────────────────────────

def test_retire_un_bloc_de_raisonnement():
    assert backends.clean_output("<think>Bon alors…</think>Le message final.") \
        == "Le message final."


def test_retire_un_raisonnement_non_ferme():
    assert backends.clean_output("Le message final.<think>je continue") == "Le message final."


def test_garde_le_texte_apres_le_dernier_think():
    assert backends.clean_output("<think>a</think><think>b</think>Final") == "Final"


def test_retire_une_fence_englobante():
    assert backends.clean_output("```markdown\n## Titre\nTexte\n```") == "## Titre\nTexte"


def test_conserve_une_fence_interne():
    texte = "Voir le correctif :\n```python\nprint(1)\n```\nÀ tester."
    assert backends.clean_output(texte) == texte


def test_retire_une_ligne_de_preambule():
    assert backends.clean_output("Voici le message nettoyé :\nSalut, ça va ?") == "Salut, ça va ?"


def test_ne_retire_pas_une_premiere_ligne_utile():
    texte = "Objet : facture de juin\n\nMadame, Monsieur,"
    assert backends.clean_output(texte) == texte


def test_retire_les_guillemets_encadrants():
    assert backends.clean_output("« Salut, ça va ? »") == "Salut, ça va ?"
    assert backends.clean_output('"Salut"') == "Salut"


def test_conserve_les_guillemets_internes():
    texte = 'Il a dit « oui » puis « non »'
    assert backends.clean_output(texte) == texte


def test_chaine_vide():
    assert backends.clean_output("") == ""
    assert backends.clean_output(None) == ""


# ─── Ollama : requête et erreurs ──────────────────────────────────────────────

class FakeResponse:
    def __init__(self, payload):
        self._payload = json.dumps(payload).encode()

    def read(self):
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


@pytest.fixture
def ollama(monkeypatch):
    """Backend Ollama dont le transport est simulé ; `calls` reçoit les requêtes envoyées."""
    backend = backends.OllamaBackend(model="modele-test", keep_alive="5m", timeout=12,
                                     num_ctx=4096, temperature=0.3)
    calls = []

    def fake_urlopen(request, timeout=None):
        body = json.loads(request.data.decode()) if request.data else None
        calls.append({"url": request.full_url, "body": body, "timeout": timeout})
        if request.full_url.endswith("/api/show"):
            return FakeResponse({"capabilities": ["completion", "thinking"]})
        return FakeResponse({"message": {"content": "  Texte reformulé.  "}})

    monkeypatch.setattr(backends.urllib.request, "urlopen", fake_urlopen)
    return backend, calls


def test_reformat_envoie_le_bon_payload(ollama):
    backend, calls = ollama
    assert backend.reformat("texte brut", "consigne") == "Texte reformulé."

    chat = [call for call in calls if call["url"].endswith("/api/chat")][0]
    body = chat["body"]
    assert body["model"] == "modele-test"
    assert body["stream"] is False
    assert body["keep_alive"] == "5m"
    assert body["options"]["num_ctx"] == 4096
    assert body["options"]["temperature"] == 0.3
    assert body["messages"][0] == {"role": "system", "content": "consigne"}
    assert body["messages"][1] == {"role": "user", "content": "texte brut"}
    assert chat["timeout"] == 12


def test_think_desactive_seulement_si_le_modele_sait_penser(ollama, monkeypatch):
    backend, calls = ollama
    backend.reformat("t", "c")
    assert [c for c in calls if c["url"].endswith("/api/chat")][0]["body"]["think"] is False

    backend._capabilities = {"modele-test": ["completion"]}
    calls.clear()
    backend.reformat("t", "c")
    assert "think" not in [c for c in calls if c["url"].endswith("/api/chat")][0]["body"]


def test_capabilities_sont_mises_en_cache(ollama):
    backend, calls = ollama
    backend.capabilities()
    backend.capabilities()
    assert len([c for c in calls if c["url"].endswith("/api/show")]) == 1


def test_ollama_eteint_donne_un_message_lisible(monkeypatch):
    backend = backends.OllamaBackend()
    backend._capabilities = {backend.model: []}

    def refuse(*_args, **_kwargs):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(backends.urllib.request, "urlopen", refuse)
    with pytest.raises(backends.ReformatError, match="injoignable"):
        backend.reformat("t", "c")


def test_modele_absent_est_signale(monkeypatch):
    backend = backends.OllamaBackend(model="fantome")
    backend._capabilities = {"fantome": []}

    class FakeHTTPError(urllib.error.HTTPError):
        def __init__(self):
            super().__init__("url", 404, "Not Found", {}, None)

        def read(self):
            return json.dumps({"error": 'model "fantome" not found'}).encode()

    def raise_http(*_args, **_kwargs):
        raise FakeHTTPError()

    monkeypatch.setattr(backends.urllib.request, "urlopen", raise_http)
    with pytest.raises(backends.ReformatError, match="fantome"):
        backend.reformat("t", "c")


def test_timeout_est_signale(monkeypatch):
    backend = backends.OllamaBackend(timeout=7)
    backend._capabilities = {backend.model: []}

    def timeout(*_args, **_kwargs):
        raise TimeoutError()

    monkeypatch.setattr(backends.urllib.request, "urlopen", timeout)
    with pytest.raises(backends.ReformatError, match="7 s"):
        backend.reformat("t", "c")


def test_reponse_vide_est_une_erreur(monkeypatch):
    backend = backends.OllamaBackend()
    backend._capabilities = {backend.model: []}
    monkeypatch.setattr(backends.urllib.request, "urlopen",
                        lambda *a, **k: FakeResponse({"message": {"content": "   "}}))
    with pytest.raises(backends.ReformatError, match="vide"):
        backend.reformat("t", "c")


# ─── Contrôle de la langue de sortie ──────────────────────────────────────────

class StubBackend:
    """Backend simulé : renvoie les réponses de la liste, une par appel."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def reformat(self, _text, system_prompt):
        self.prompts.append(system_prompt)
        return self.responses.pop(0)


def test_pas_de_verification_sans_langue_cible():
    backend = StubBackend(["Résultat"])
    assert backends.reformat(backend, "t", "c", None) == ("Résultat", None)
    assert len(backend.prompts) == 1


def test_traduction_correcte_passe_du_premier_coup():
    backend = StubBackend(["転写のショートカットが動作しません。"])
    result, warning = backends.reformat(backend, "t", "c", "ja")
    assert warning is None
    assert len(backend.prompts) == 1
    assert "転写" in result


def test_traduction_restee_en_francais_declenche_une_seconde_tentative():
    backend = StubBackend([
        "## Le raccourci ne fonctionne pas et le texte brut est collé sans message clair",
        "## ショートカットが動作せず、エラーメッセージも表示されません。",
    ])
    result, warning = backends.reformat(backend, "t", "consigne", "ja")
    assert len(backend.prompts) == 2
    assert backends.RETRY_DIRECTIVE in backend.prompts[1]
    assert warning is None
    assert "ショートカット" in result


def test_deux_echecs_de_suite_avertissent_l_utilisateur():
    francais = "## Le raccourci ne fonctionne pas et le texte est collé sans message clair"
    backend = StubBackend([francais, francais])
    result, warning = backends.reformat(backend, "t", "c", "ja")
    assert warning == "Traduction incertaine — vérifie la langue"
    assert result == francais


def test_echec_de_la_seconde_tentative_garde_le_premier_resultat():
    class Flaky(StubBackend):
        def reformat(self, text, system_prompt):
            self.prompts.append(system_prompt)
            if len(self.prompts) == 1:
                return "Texte resté en français avec beaucoup de mots pour être jugeable ici"
            raise backends.ReformatError("Ollama injoignable")

    backend = Flaky([])
    result, warning = backends.reformat(backend, "t", "c", "en")
    assert "français" in result
    assert warning == "Traduction incertaine — vérifie la langue"


# ─── Heuristique de langue ────────────────────────────────────────────────────

def test_detecte_les_ecritures_non_latines():
    assert langcheck.looks_like("これはテストです", "ja") is True
    assert langcheck.looks_like("Ceci est un test", "ja") is False
    assert langcheck.looks_like("Это тест на русском", "ru") is True


def test_compare_les_langues_latines_au_francais():
    anglais = "The invoice amount is wrong and I would like to understand the difference please"
    francais = "Le montant de la facture est faux et j'aimerais comprendre la différence"
    assert langcheck.looks_like(anglais, "en") is True
    assert langcheck.looks_like(francais, "en") is False
    assert langcheck.looks_like(francais, "fr") is True


def test_texte_trop_court_est_indetermine():
    assert langcheck.looks_like("Salut", "en") is None
    assert langcheck.looks_like("", "en") is None
