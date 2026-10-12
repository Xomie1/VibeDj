"""Audius client key rotation, against a fake transport (no network needed)."""
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest

from app.audius import AudiusClient, KeyPool
from app.config import audius_credentials

KEYS = [("0xk1", "tok1"), ("0xk2", "tok2"), ("0xk3", "tok3")]


def fake_audius(status_for=lambda key: 200):
    """Transport that records (api_key, Authorization) per request and answers status_for(api_key)."""
    seen = []

    def handler(req: httpx.Request) -> httpx.Response:
        key = req.url.params.get("api_key")
        seen.append((key, req.headers.get("Authorization")))
        status = status_for(key)
        if status != 200:
            return httpx.Response(status, json={"error": "nope"})
        if req.url.path.endswith("/stream"):
            return httpx.Response(200, content=b"x" * 100)
        return httpx.Response(200, json={"data": [{"id": "t1"}]})

    return httpx.MockTransport(handler), seen


@pytest.fixture
def sleeps(monkeypatch):
    calls = []
    monkeypatch.setattr("app.audius.time.sleep", calls.append)
    return calls


def test_credentials_from_env():
    env = {
        "AUDIUS_API_KEY_2": "0xk2", "AUDIUS_BEARER_TOKEN_2": "tok2",
        "AUDIUS_API_KEY_1": "0xk1", "AUDIUS_BEARER_TOKEN_1": "tok1",
        "AUDIUS_BEARER_TOKEN": "tok0",          # old single-key setting still counts
        "AUDIUS_API_KEY_3": "  ",               # blank entries are ignored
        "AUDIUS_APP_NAME": "vibe-dj",
    }
    assert audius_credentials(env) == [("", "tok0"), ("0xk1", "tok1"), ("0xk2", "tok2")]


def test_rotates_keys_each_request(sleeps):
    transport, seen = fake_audius()
    c = AudiusClient(credentials=KEYS, transport=transport)
    for _ in range(6):
        assert c.trending(limit=1) == [{"id": "t1"}]
    assert [k for k, _ in seen] == ["0xk1", "0xk2", "0xk3"] * 2
    # each request carries the bearer token that belongs to its own key
    assert all(auth == f"Bearer tok{k[-1]}" for k, auth in seen)
    assert sleeps == []


def test_rate_limited_key_sits_out(sleeps):
    transport, seen = fake_audius(lambda key: 429 if key == "0xk1" else 200)
    c = AudiusClient(credentials=KEYS, transport=transport)
    for _ in range(4):
        c.trending(limit=1)
    keys = [k for k, _ in seen]
    assert keys[:2] == ["0xk1", "0xk2"]       # 429 on k1, same request retried on k2
    assert keys.count("0xk1") == 1            # then k1 rests while k2/k3 carry on
    assert c.pool.creds[0].resting_until > c.pool.clock()
    assert sleeps == []


def test_refused_key_sits_out(sleeps):
    transport, seen = fake_audius(lambda key: 401 if key == "0xk2" else 200)
    c = AudiusClient(credentials=KEYS, transport=transport)
    for _ in range(5):
        c.trending(limit=1)
    assert [k for k, _ in seen].count("0xk2") == 1
    assert c.pool.creds[1].resting_until - c.pool.clock() > 3000


def test_gives_up_when_every_key_is_limited(sleeps):
    transport, seen = fake_audius(lambda key: 429)
    c = AudiusClient(credentials=KEYS, transport=transport)
    with pytest.raises(httpx.HTTPStatusError):
        c.trending(limit=1)
    assert len(seen) == len(KEYS) + 2
    assert all(s <= 10 for s in sleeps)       # waits for a key to free up, but never for long


def test_no_keys_still_works(sleeps):
    transport, seen = fake_audius()
    c = AudiusClient(credentials=[], transport=transport)
    assert c.trending(limit=1) == [{"id": "t1"}]
    assert seen == [(None, None)]             # app_name only, no key, no token
    assert c.pool.configured == 0


def test_stream_url_rotates_public_key_only():
    c = AudiusClient(credentials=KEYS, app_name="vibe-dj")
    urls = [c.stream_url("abc") for _ in range(3)]
    qs = [parse_qs(urlsplit(u).query) for u in urls]
    assert [q["api_key"] for q in qs] == [["0xk1"], ["0xk2"], ["0xk3"]]
    assert all(q["app_name"] == ["vibe-dj"] for q in qs)
    assert not any("tok" in u for u in urls)   # bearer tokens never reach the browser


def test_download_head_fails_over(sleeps):
    transport, seen = fake_audius(lambda key: 429 if key == "0xk1" else 200)
    c = AudiusClient(credentials=KEYS, transport=transport)
    assert c.download_head("abc", max_bytes=50) == b"x" * 100
    assert [k for k, _ in seen] == ["0xk1", "0xk2"]


def test_pool_waits_for_soonest_key_when_all_rest():
    now = [100.0]
    pool = KeyPool(KEYS, clock=lambda: now[0])
    a, b, c = pool.creds
    pool.bench(a, 30)
    pool.bench(b, 5)
    pool.bench(c, 60)
    cred, wait = pool.pick()
    assert cred is b and wait == pytest.approx(5)
    now[0] += 6
    assert pool.pick() == (b, 0.0)            # rested long enough, back in rotation


def test_repeated_rate_limits_rest_longer():
    pool = KeyPool(KEYS, clock=lambda: 0.0)
    a = pool.creds[0]
    assert [pool.bench(a) for _ in range(3)] == [2, 4, 8]
    pool.ok(a)
    assert pool.bench(a) == 2
