"""Tests on a synthetic catalogue (no network needed).

Run:  pytest -q
"""
import os
import tempfile
from urllib.parse import parse_qs, urlsplit

import numpy as np
import pytest

from app.keys import key_compat, parse_key
from app.recommender import DEFAULT_WEIGHTS, Recommender, Section, SessionState, bpm_fit


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
        st.played.append(rec.next_track(st, rng=rng).track)
    same = sum(scene(t) == "1" for t in st.played)
    assert same >= 14, [scene(t) for t in st.played]


def test_no_repeats_and_artist_spacing():
    tracks = make_catalogue()
    rec = Recommender(tracks)
    st = SessionState(played=[tracks[0]])
    for _ in range(20):
        st.played.append(rec.next_track(st, rng=np.random.default_rng(len(st.played))).track)
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
    sec = s["track"]["section"]
    assert (sec["n"], sec["kind"], sec["position"], sec["label"]) == (1, "start", 1, "Warm Deep House")

    nxt = c.post(f"/api/sessions/{sid}/next").json()
    assert "reason" in nxt["track"]
    stream_qs = parse_qs(urlsplit(nxt["track"]["stream_url"]).query)
    assert stream_qs["app_name"] == ["vibe-dj"]
    r = c.post(f"/api/sessions/{sid}/event", json={"track_id": nxt["track"]["id"], "outcome": "skip", "listened_sec": 5}).json()
    assert r["learned"] is False
    assert c.post(f"/api/sessions/{sid}/vibe", json={"vibe": "up"}).json()["ok"]
    nxt2 = c.post(f"/api/sessions/{sid}/next").json()
    assert nxt2["track"]["id"] != nxt["track"]["id"]
    assert [x["track"]["section"]["position"] for x in (nxt, nxt2)] == [2, 3]   # skips count too
    assert {x["track"]["genre"] for x in (nxt, nxt2)} == {"Deep House"}

    r = c.post(f"/api/sessions/{sid}/event", json={"track_id": nxt2["track"]["id"], "outcome": "switch", "listened_sec": 40}).json()
    assert r["learned"] is None                  # leaving via "Switch it up" isn't a verdict on the track
    sw = c.post(f"/api/sessions/{sid}/switch").json()["track"]
    assert (sw["section"]["n"], sw["section"]["kind"], sw["section"]["position"]) == (2, "switch", 1)
    assert sw["genre"] != "Deep House"
    after = c.post(f"/api/sessions/{sid}/next").json()["track"]
    assert (after["section"]["n"], after["section"]["position"], after["genre"]) == (2, 2, sw["genre"])

    logged = d.conn.execute("SELECT section, propensity FROM plays WHERE session_id=? ORDER BY id", (sid,)).fetchall()
    assert [r["section"] for r in logged] == [1, 1, 1, 2, 2]
    assert logged[0]["propensity"] is None and all(0 < r["propensity"] <= 1 for r in logged[1:])
    assert c.get("/").status_code == 200


# ---- sections ----------------------------------------------------------------
def serve(st, pick):
    """Do what the API does after a pick: it plays, and counts toward its section."""
    if pick.section:
        st.section, st.section_played, st.section_served = pick.section, [], 0
    st.played.append(pick.track)
    st.section_played.append(pick.track)
    st.section_served += 1


def start_state(rec, seed, rng):
    return SessionState(played=[seed], section=rec.new_section(seed, "start", rng),
                        section_played=[seed], section_served=1)


def test_sections_hold_a_theme_then_move_on():
    tracks = make_catalogue()
    rec = Recommender(tracks)
    rng = np.random.default_rng(3)
    st = start_state(rec, tracks[45], rng)
    sections = [(st.section, [tracks[45]])]
    for _ in range(20):
        p = rec.next_track(st, rng=rng)
        serve(st, p)
        if p.section:
            sections.append((p.section, []))
        sections[-1][1].append(p.track)

    assert len(sections) >= 3
    assert [s.kind for s, _ in sections[:2]] == ["start", "next"]
    for sec, played in sections:
        assert all(t["genre"] == sec.genre for t in played)           # on theme
    for sec, played in sections[:-1]:
        assert 4 <= sec.length <= 6 and len(played) == sec.length      # full length, then on
    themes = [(s.genre, s.band) for s, _ in sections]
    assert all(a != b for a, b in zip(themes, themes[1:]))             # every new section is a change


def test_switch_jumps_somewhere_else():
    tracks = make_catalogue()
    rec = Recommender(tracks)
    rng = np.random.default_rng(4)
    st = start_state(rec, tracks[45], rng)                             # deep house
    for _ in range(2):
        serve(st, rec.next_track(st, rng=rng))
    stay = rec.next_track(st, rng=rng)
    sw = rec.next_track(st, switch=True, rng=rng)
    assert stay.section is None and scene(stay.track) == "1"
    assert sw.section.kind == "switch" and scene(sw.track) != "1"
    assert sw.components["sim"] < stay.components["sim"]

    serve(st, sw)                                                      # the new section holds
    nxt = rec.next_track(st, rng=rng)
    assert nxt.section is None and nxt.track["genre"] == sw.track["genre"]


def test_section_moves_on_when_its_genre_runs_out():
    tracks = make_catalogue(n_per=3)
    rec = Recommender(tracks)
    rng = np.random.default_rng(5)
    seed = tracks[3]                                                   # deep house, 2 more left
    st = SessionState(played=[seed], section=Section("Deep House", "mid", 6),
                      section_played=[seed], section_served=1)
    for _ in range(2):
        p = rec.next_track(st, rng=rng)
        assert p.section is None
        serve(st, p)
    p = rec.next_track(st, rng=rng)
    assert p.section.kind == "next" and p.track["genre"] != "Deep House"


def test_energy_bands_follow_the_catalogue():
    tracks = make_catalogue()
    assert [Recommender(tracks).band({"energy": e}) for e in (0.25, 0.6, 0.9)] == ["low", "mid", "high"]
    for t in tracks:                                                   # same music, mastered louder
        t["energy"] = 0.7 + 0.3 * t["energy"]
    assert Recommender(tracks).band({"energy": 0.78}) == "low"
    assert Section("Lo-Fi", "low", 4).label == "Chilled Lo-Fi"
    assert Section(None, "high", 4).label == "Peak-time mix"


def test_pick_logs_its_probability():
    tracks = make_catalogue()
    rec = Recommender(tracks)
    st = SessionState(played=[tracks[45]])
    assert rec.next_track(st, top_k=1).propensity == 1.0
    flat = rec.next_track(st, temperature=1e9, rng=np.random.default_rng(0))
    assert flat.propensity == pytest.approx(0.1)                       # uniform over the top 10


def test_old_database_gets_new_columns(tmp_path):
    import sqlite3
    path = tmp_path / "old.db"
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE plays (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, track_id TEXT, "
                "ts REAL, outcome TEXT, listened_sec REAL, components TEXT)")
    con.commit()
    con.close()
    from app.db import DB
    d = DB(path)
    d.log_queued("s", "t", None, section=1, propensity=0.5)
    assert dict(d.conn.execute("SELECT section, propensity FROM plays").fetchone()) == {"section": 1, "propensity": 0.5}
