"""Tests on a synthetic catalogue (no network needed).

Run:  pytest -q
"""
import os
import tempfile
from urllib.parse import parse_qs, urlsplit

import numpy as np
import pytest

from app.keys import key_compat, parse_key
from app.recommender import DEFAULT_WEIGHTS, Recommender, SessionState, bpm_fit


def make_catalogue(n_per=40, seed=0):
    """Three clear 'scenes': chill lofi, deep house, fast drum & bass."""
    rng = np.random.default_rng(seed)
    scenes = [
        ("Lo-Fi", 82, 0.25, "8A"),
        ("Deep House", 122, 0.6, "5A"),
        ("Drum & Bass", 174, 0.9, "10A"),
    ]
    tracks = []
    for s, (genre, bpm, energy, key) in enumerate(scenes):
        centre = rng.normal(size=25)
        for k in range(n_per):
            v = centre + rng.normal(scale=0.35, size=25)
            tracks.append({
                "id": f"{s}-{k}", "title": f"{genre} {k}", "artist": f"artist-{s}-{k % 8}",
                "artist_id": f"a{s}-{k % 8}", "genre": genre,
                "bpm": bpm + rng.normal(scale=2), "key": key,
                "energy": float(np.clip(energy + rng.normal(scale=0.05), 0, 1)),
                "play_count": int(rng.integers(10, 10000)),
                "vector": (v / np.linalg.norm(v)).astype(np.float32),
            })
    return tracks


def scene(t):
    return t["id"].split("-")[0]


def test_keys():
    assert parse_key("A minor") == "8A"
    assert parse_key("C") == "8B"
    assert parse_key("Bbm") == "3A"
    assert parse_key("11b") == "11B"
    assert parse_key("nonsense") is None
    assert key_compat("8A", "8A") == 1.0
    assert key_compat("8A", "9A") > key_compat("8A", "2B")
    assert key_compat("12A", "1A") == 0.85   # wraps around the wheel


def test_bpm_fit_half_double():
    assert bpm_fit(120, 120) == pytest.approx(1.0)
    assert bpm_fit(87, 174) > 0.9            # half-time counts
    assert bpm_fit(120, 140) < 0.1


def test_stays_in_the_vibe():
    tracks = make_catalogue()
    rec = Recommender(tracks)
    rng = np.random.default_rng(1)
    st = SessionState(played=[tracks[45]])   # start in deep house
    for _ in range(15):
        t, _ = rec.next_track(st, rng=rng)
        st.played.append(t)
    same = sum(scene(t) == "1" for t in st.played)
    assert same >= 14, [scene(t) for t in st.played]


def test_no_repeats_and_artist_spacing():
    tracks = make_catalogue()
    rec = Recommender(tracks)
    st = SessionState(played=[tracks[0]])
    for _ in range(20):
        t, _ = rec.next_track(st, rng=np.random.default_rng(len(st.played)))
        st.played.append(t)
    ids = [t["id"] for t in st.played]
    assert len(ids) == len(set(ids))
    back_to_back = sum(a["artist_id"] == b["artist_id"] for a, b in zip(st.played, st.played[1:]))
    assert back_to_back <= 1


def test_vibe_up_raises_energy():
    tracks = make_catalogue()
    rec = Recommender(tracks)
    base = SessionState(played=[tracks[45]], vibe="hold")
    up = SessionState(played=[tracks[45]], vibe="up")
    assert rec.target_energy(up) > rec.target_energy(base)


def test_learns_from_skips():
    """A listener who always skips low-BPM-fit picks should push the bpm weight up."""
    tracks = make_catalogue()
    rec = Recommender(tracks, lr=0.1)
    w0 = rec.weights["bpm"]
    for _ in range(200):
        rec.learn({**{c: 0.5 for c in DEFAULT_WEIGHTS if c != "bias"}, "bpm": 0.1}, liked=False)
        rec.learn({**{c: 0.5 for c in DEFAULT_WEIGHTS if c != "bias"}, "bpm": 0.95}, liked=True)
    assert rec.weights["bpm"] > w0


def test_api_end_to_end(monkeypatch):
    tmp = tempfile.mkdtemp()
    monkeypatch.setenv("VIBE_DB_PATH", os.path.join(tmp, "t.db"))
    import importlib
    import app.config as config
    importlib.reload(config)
    import app.db as dbmod
    importlib.reload(dbmod)
    d = dbmod.DB()
    for t in make_catalogue(n_per=10):
        d.upsert_track(t)
        d.save_features(t["id"], {k: t[k] for k in ("bpm", "key", "energy", "vector")} | {"brightness": .5, "percussive": .5})
    import app.main as main
    importlib.reload(main)
    from fastapi.testclient import TestClient
    c = TestClient(main.app)

    assert c.get("/api/stats").json()["analysed"] == 30
    assert c.get("/api/search", params={"q": "house"}).json()
    s = c.post("/api/sessions", json={"seed_track_id": "1-0"}).json()
    sid = s["session_id"]
    assert s["track"]["id"] == "1-0"

    nxt = c.post(f"/api/sessions/{sid}/next").json()
    assert "reason" in nxt["track"]
    stream_qs = parse_qs(urlsplit(nxt["track"]["stream_url"]).query)
    assert stream_qs["app_name"] == ["vibe-dj"]
    r = c.post(f"/api/sessions/{sid}/event", json={"track_id": nxt["track"]["id"], "outcome": "skip", "listened_sec": 5}).json()
    assert r["learned"] is False
    assert c.post(f"/api/sessions/{sid}/vibe", json={"vibe": "up"}).json()["ok"]
    nxt2 = c.post(f"/api/sessions/{sid}/next").json()
    assert nxt2["track"]["id"] != nxt["track"]["id"]
    assert c.get("/").status_code == 200
