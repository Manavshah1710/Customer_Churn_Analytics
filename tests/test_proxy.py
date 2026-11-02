"""
Tests for the OpenRouter failover proxy, using a fake upstream (no network).
"""

import http.client
import io
import json
import os
import threading
import urllib.error
from unittest.mock import patch

import pytest

import claude_proxy as cp

TEST_CONFIGS = [
    {"name": "A", "key": "aaaa", "model": "m"},
    {"name": "B", "key": "bbbb", "model": "m"},
]


class FakeResponse:
    """Upstream reply: returns each chunk in turn, raising any chunk that is an exception."""
    status = 200

    def __init__(self, chunks):
        self.chunks = iter(chunks)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass

    def getheaders(self):
        return [("Content-Type", "text/event-stream")]

    def read(self, size=-1):
        chunk = next(self.chunks, b"")
        if isinstance(chunk, Exception):
            raise chunk
        return chunk


def http_error(code, body):
    return urllib.error.HTTPError("http://upstream", code, "error", {}, io.BytesIO(body))


@pytest.fixture
def proxy(monkeypatch):
    """Runs the proxy on a free port. Set proxy.upstream, then call proxy.post(); proxy.calls lists keys used."""
    monkeypatch.setattr(cp, "CONFIGS", TEST_CONFIGS)
    monkeypatch.setattr(cp.FailoverHandler, "current_index", 0)
    monkeypatch.setattr(cp.time, "sleep", lambda seconds: None)

    server = cp.http.server.ThreadingHTTPServer(("127.0.0.1", 0), cp.FailoverHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    class Proxy:
        calls = []
        upstream = None

        @staticmethod
        def post():
            conn = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=10)
            conn.request("POST", "/v1/messages", body=json.dumps({"model": "x"}))
            res = conn.getresponse()
            return res.status, res.read()

    def fake_urlopen(req, timeout=None):
        key = req.get_header("Authorization").split()[-1]
        Proxy.calls.append(key)
        return Proxy.upstream(key)

    monkeypatch.setattr(cp.urllib.request, "urlopen", fake_urlopen)
    yield Proxy
    server.shutdown()
    server.server_close()


def test_failover_sticks_across_requests(proxy):
    """Once account A is rate limited, the next request must start on account B."""
    def upstream(key):
        if key == "aaaa":
            raise http_error(429, b"limited")
        return FakeResponse([b"ok"])
    proxy.upstream = upstream

    assert proxy.post() == (200, b"ok")
    assert proxy.post() == (200, b"ok")
    assert proxy.calls == ["aaaa", "bbbb", "bbbb"]


def test_no_retry_after_reply_started(proxy):
    """If upstream dies mid-reply, the client must not get a second reply joined on."""
    proxy.upstream = lambda key: FakeResponse([b"partial", TimeoutError("timed out")])

    assert proxy.post() == (200, b"partial")
    assert proxy.calls == ["aaaa"]


def test_error_mentioning_rate_is_not_failover(proxy):
    """A 400 whose text happens to contain 'rate' must reach the client unchanged."""
    def upstream(key):
        raise http_error(400, b"failed to generate")
    proxy.upstream = upstream

    assert proxy.post() == (400, b"failed to generate")
    assert proxy.calls == ["aaaa"]


def test_main_stops_without_keys(monkeypatch):
    """Starting with no keys must fail clearly instead of serving false rate limit errors."""
    monkeypatch.setattr(cp, "CONFIGS", [])
    with pytest.raises(SystemExit):
        cp.main()


def test_env_loader_only_loads_openrouter_keys(tmp_path):
    """Importing the proxy must not put ANTHROPIC_API_KEY into the test run, where it could reach the real API."""
    env_file = tmp_path / ".env"
    env_file.write_text("ANTHROPIC_API_KEY=fake\nOPENROUTER_API_KEY_1=fake-router\n")

    with patch.dict(os.environ):
        os.environ.pop("ANTHROPIC_API_KEY", None)
        os.environ.pop("OPENROUTER_API_KEY_1", None)
        cp.load_env_file(str(env_file))

        assert "ANTHROPIC_API_KEY" not in os.environ
        assert os.environ["OPENROUTER_API_KEY_1"] == "fake-router"
