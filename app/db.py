"""SQLite storage. Small enough for an MVP; see supabase/schema.sql for the pgvector version."""
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from pathlib import Path

import numpy as np

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS tracks (
    id TEXT PRIMARY KEY,
    title TEXT, artist TEXT, artist_id TEXT,
    genre TEXT, mood TEXT, tags TEXT,
    duration REAL, play_count INTEGER, artwork TEXT,
    bpm REAL, key TEXT,
    energy REAL, brightness REAL, percussive REAL,
    vector BLOB,
    analysed INTEGER DEFAULT 0,
    added_at REAL
);
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    created_at REAL,
    vibe TEXT DEFAULT 'hold'          -- 'up' | 'down' | 'hold'
);
CREATE TABLE IF NOT EXISTS plays (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT, track_id TEXT,
    ts REAL,
    outcome TEXT,                     -- 'queued' | 'skip' | 'complete' | 'like'
    listened_sec REAL,
    components TEXT                   -- JSON feature vector the recommender scored it with
);
CREATE TABLE IF NOT EXISTS weights (
    name TEXT PRIMARY KEY,
    value TEXT
);
CREATE INDEX IF NOT EXISTS plays_session ON plays(session_id, ts);
"""


class DB:
    def __init__(self, path: Path | str | None = None):
        path = Path(path or config.DB_PATH)
        if str(path) != ":memory:":
            path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)

    # ---- tracks ----------------------------------------------------------
    def upsert_track(self, t: dict) -> None:
        cols = ["id", "title", "artist", "artist_id", "genre", "mood", "tags",
                "duration", "play_count", "artwork"]
        vals = [t.get(c) for c in cols]
        self.conn.execute(
            f"INSERT INTO tracks ({','.join(cols)}, added_at) VALUES ({','.join('?' * len(cols))}, ?) "
            f"ON CONFLICT(id) DO UPDATE SET play_count=excluded.play_count",
            vals + [time.time()],
        )
        self.conn.commit()

    def save_features(self, track_id: str, f: dict) -> None:
        vec = np.asarray(f["vector"], dtype=np.float32).tobytes()
        self.conn.execute(
            "UPDATE tracks SET bpm=?, key=?, energy=?, brightness=?, percussive=?, vector=?, analysed=1 WHERE id=?",
            (f["bpm"], f["key"], f["energy"], f["brightness"], f["percussive"], vec, track_id),
        )
        self.conn.commit()

    def mark_failed(self, track_id: str) -> None:
        self.conn.execute("UPDATE tracks SET analysed=-1 WHERE id=?", (track_id,))
        self.conn.commit()

    def unanalysed(self, limit: int = 1000) -> list[str]:
        rows = self.conn.execute("SELECT id FROM tracks WHERE analysed=0 LIMIT ?", (limit,))
        return [r["id"] for r in rows]

    def analysed_tracks(self) -> list[dict]:
        rows = self.conn.execute("SELECT * FROM tracks WHERE analysed=1").fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["vector"] = np.frombuffer(d["vector"], dtype=np.float32)
            out.append(d)
        return out

    def get_track(self, track_id: str) -> dict | None:
        r = self.conn.execute("SELECT * FROM tracks WHERE id=?", (track_id,)).fetchone()
        if not r:
            return None
        d = dict(r)
        d.pop("vector", None)
        return d

    def search(self, q: str, limit: int = 20) -> list[dict]:
        like = f"%{q}%"
        rows = self.conn.execute(
            "SELECT id,title,artist,genre,mood,bpm,key,energy,artwork FROM tracks "
            "WHERE analysed=1 AND (title LIKE ? OR artist LIKE ? OR genre LIKE ? OR mood LIKE ? OR tags LIKE ?) "
            "ORDER BY play_count DESC LIMIT ?",
            (like, like, like, like, like, limit),
        )
        return [dict(r) for r in rows]

    def count(self) -> dict:
        r = self.conn.execute(
            "SELECT COUNT(*) total, SUM(analysed=1) analysed, SUM(analysed=-1) failed FROM tracks"
        ).fetchone()
        return {k: (r[k] or 0) for k in ("total", "analysed", "failed")}

    # ---- sessions --------------------------------------------------------
    def new_session(self) -> str:
        sid = uuid.uuid4().hex[:12]
        self.conn.execute("INSERT INTO sessions (id, created_at) VALUES (?, ?)", (sid, time.time()))
        self.conn.commit()
        return sid

    def set_vibe(self, sid: str, vibe: str) -> None:
        self.conn.execute("UPDATE sessions SET vibe=? WHERE id=?", (vibe, sid))
        self.conn.commit()

    def get_session(self, sid: str) -> dict | None:
        r = self.conn.execute("SELECT * FROM sessions WHERE id=?", (sid,)).fetchone()
        return dict(r) if r else None

    def log_queued(self, sid: str, track_id: str, components: dict | None) -> int:
        cur = self.conn.execute(
            "INSERT INTO plays (session_id, track_id, ts, outcome, components) VALUES (?,?,?,?,?)",
            (sid, track_id, time.time(), "queued", json.dumps(components) if components else None),
        )
        self.conn.commit()
        return cur.lastrowid

    def set_outcome(self, sid: str, track_id: str, outcome: str, listened: float | None) -> dict | None:
        r = self.conn.execute(
            "SELECT * FROM plays WHERE session_id=? AND track_id=? ORDER BY ts DESC LIMIT 1",
            (sid, track_id),
        ).fetchone()
        if not r:
            return None
        self.conn.execute("UPDATE plays SET outcome=?, listened_sec=? WHERE id=?", (outcome, listened, r["id"]))
        self.conn.commit()
        d = dict(r)
        d["components"] = json.loads(d["components"]) if d["components"] else None
        return d

    def history(self, sid: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT track_id, outcome, listened_sec FROM plays WHERE session_id=? ORDER BY ts", (sid,)
        )
        return [dict(r) for r in rows]

    # ---- learned weights -------------------------------------------------
    def load_weights(self, name: str = "global") -> dict | None:
        r = self.conn.execute("SELECT value FROM weights WHERE name=?", (name,)).fetchone()
        return json.loads(r["value"]) if r else None

    def save_weights(self, w: dict, name: str = "global") -> None:
        self.conn.execute(
            "INSERT INTO weights (name, value) VALUES (?, ?) ON CONFLICT(name) DO UPDATE SET value=excluded.value",
            (name, json.dumps(w)),
        )
        self.conn.commit()
