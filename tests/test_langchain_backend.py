"""Exercise real ChatOllama serialization using an HTTP mock transport."""
import json

import httpx
import pytest
from langchain_ollama import ChatOllama
from langsmith import Client

from sw import pipeline
from sw.backends import ReformatError
from sw.langchain_backend import LangChainOllamaBackend, validate_local


@pytest.mark.parametrize("host,model", [
    ("https://api.example.com", "qwen3:4b"), ("http://192.168.1.2:11434", "local"),
    ("http://user:password@localhost:11434", "local"),
    ("http://localhost:11434", "gpt-oss:120b-cloud"),
])
def test_nonlocal_settings_rejected(host, model):
    with pytest.raises(ReformatError):
        validate_local({"ollama_host": host, "ollama_model": model})


@pytest.mark.parametrize("host", ["localhost:11434", "http://127.0.0.1:11434", "http://[::1]:11434"])
def test_loopback_accepted(host):
    assert validate_local({"ollama_host": host, "ollama_model": "qwen3:4b"})


def transport_setup(monkeypatch, reply="<think>hidden</think>\nHello", show=None, status=200):
    requests = []
    kwargs_seen = []

    def handle(request):
        requests.append((request.url, json.loads(request.content)))
        if request.url.path == "/api/show":
            return httpx.Response(200, json=show if show is not None else {"capabilities": ["thinking"]})
        body = {"model": "local", "message": {"role": "assistant", "content": reply}, "done": True}
        if status != 200:
            return httpx.Response(status, json={"error": "missing"})
        return httpx.Response(200, content=json.dumps(body) + "\n")

    transport = httpx.MockTransport(handle)

    def post(url, **kwargs):
        assert kwargs.pop("trust_env") is False
        with httpx.Client(transport=transport, trust_env=False) as client:
            return client.post(url, **kwargs)

    monkeypatch.setattr(httpx, "post", post)

    def model(**kwargs):
        kwargs_seen.append(kwargs.copy())
        kwargs["sync_client_kwargs"] = {"transport": transport}
        return ChatOllama(**kwargs)

    monkeypatch.setattr("langchain_ollama.ChatOllama", model)
    return requests, kwargs_seen


def test_real_graph_and_langchain_request_with_ambient_tracing_disabled(monkeypatch):
    requests, kwargs = transport_setup(monkeypatch)
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "test-key-never-sent")
    def forbidden(*args, **kwargs):
        pytest.fail("LangSmith client constructed")
    monkeypatch.setattr(Client, "__init__", forbidden)
    cfg = {"pipeline": "langgraph", "ollama_model": "local", "ollama_timeout_s": 7,
           "ollama_num_ctx": 4096, "ollama_temperature": 0.1, "ollama_keep_alive": "5m"}
    report = pipeline.run_pipeline(cfg, "my transcript", "message", "none")
    assert report.output == "Hello" and report.warning is None
    assert len(requests) == 2
    assert all(url.host == "127.0.0.1" for url, _ in requests)
    payload = requests[-1][1]
    assert payload["messages"][1]["content"] == "my transcript"
    assert [message["role"] for message in payload["messages"]] == ["system", "user"]
    assert payload["think"] is False
    assert payload["options"]["num_ctx"] == 4096
    assert payload["options"]["temperature"] == 0.1
    assert payload["keep_alive"] == "5m"
    assert kwargs[0]["client_kwargs"]["timeout"] == 7
    assert kwargs[0]["client_kwargs"]["trust_env"] is False


def test_cloud_alias_rejected_before_transcript_is_sent(monkeypatch):
    requests, kwargs = transport_setup(monkeypatch, show={"remote_model": "remote-alias"})
    backend = LangChainOllamaBackend(model="innocent-alias")
    with pytest.raises(ReformatError):
        backend.reformat("private transcript", "private prompt")
    assert len(requests) == 1
    assert requests[0][1] == {"model": "innocent-alias"}


@pytest.mark.parametrize("reply,status", [("", 200), ("ignored", 404)])
def test_empty_and_http_error_fall_back(monkeypatch, reply, status):
    transport_setup(monkeypatch, reply=reply, status=status)
    report = pipeline.run_pipeline({"pipeline": "langgraph", "ollama_model": "local"},
                                   "raw", "message", "none")
    assert report.output == "raw" and report.warning
