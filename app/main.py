"""FastAPI app: session API + serves the web player."""
from __future__ import annotations

import random
import threading
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import config
from .audius import AudiusClient
from .db import DB
from .recommender import SKIP_THRESHOLD_SEC, Recommender, SessionState

app = FastAPI(title="Vibe DJ")
db = DB()
audius = AudiusClient()
rec = Recommender(db.analysed_tracks(), weights=db.load_weights())
_lock = threading.Lock()   # recommender weights are shared across sessions

STATIC = config.ROOT / "static"
app.mount("/static", StaticFiles(directory=STATIC), name="static")


# ---- helpers ---------------------------------------------------------------
def build_state(sid: str) -> SessionState:
    sess = db.get_session(sid)
    if not sess:
        raise HTTPException(404, "session not found")
    st = SessionState(vibe=sess["vibe"])
    for h in db.history(sid):
        i = rec.by_id.get(h["track_id"])
        if i is None:
            continue
        t = rec.tracks[i]
        if h["outcome"] == "dropped":   # prefetched but never played
            continue
        if h["outcome"] == "skip" and (h["listened_sec"] or 0) < SKIP_THRESHOLD_SEC:
            st.skipped.append(t)
        else:
            st.played.append(t)
    return st


def public(t: dict, reason: str | None = None) -> dict:
    keep = ("id", "title", "artist", "genre", "mood", "bpm", "key", "energy", "artwork", "duration")
    out = {k: t.get(k) for k in keep}
    out["stream_url"] = audius.stream_url(t["id"])
    out["link"] = f"https://audius.co/tracks/{t['id']}"
    if reason:
        out["reason"] = reason
    return out


# ---- routes ----------------------------------------------------------------
@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/stats")
def stats():
    return {**db.count(), "audius_keys": audius.pool.configured,
            "weights": {k: round(v, 3) for k, v in rec.weights.items()}}


@app.get("/api/search")
def search(q: str, limit: int = 20):
    return [public(t) for t in db.search(q, limit)]


class StartReq(BaseModel):
    seed_track_id: str | None = None


@app.post("/api/sessions")
def start_session(req: StartReq):
    if not rec.tracks:
        raise HTTPException(409, "catalogue is empty, run scripts/ingest.py first")
    if req.seed_track_id and req.seed_track_id in rec.by_id:
        first = rec.tracks[rec.by_id[req.seed_track_id]]
    else:
        # no seed: start from something popular
        pool = sorted(rec.tracks, key=lambda t: -(t.get("play_count") or 0))[:50]
        first = random.choice(pool)
    sid = db.new_session()
    db.log_queued(sid, first["id"], None)
    return {"session_id": sid, "track": public(first, "your starting point")}


@app.post("/api/sessions/{sid}/next")
def next_track(sid: str):
    st = build_state(sid)
    with _lock:
        res = rec.next_track(st)
    if not res:
        raise HTTPException(410, "ran out of tracks, ingest more")
    track, comps = res
    db.log_queued(sid, track["id"], comps)
    prev = st.played[-1] if st.played else None
    return {"track": public(track, rec.explain(comps, prev, track)), "components": comps}


class EventReq(BaseModel):
    track_id: str
    outcome: Literal["skip", "complete", "like", "dropped"]
    listened_sec: float = 0.0


@app.post("/api/sessions/{sid}/event")
def event(sid: str, req: EventReq):
    row = db.set_outcome(sid, req.track_id, req.outcome, req.listened_sec)
    if not row:
        raise HTTPException(404, "track not in this session")
    comps = row["components"]
    learned = None
    if comps:   # the seed track has no components, nothing to learn from
        if req.outcome == "skip" and req.listened_sec < SKIP_THRESHOLD_SEC:
            learned = False
        elif req.outcome in ("complete", "like"):
            learned = True
        if learned is not None:
            with _lock:
                rec.learn(comps, learned)
                if req.outcome == "like":   # a like counts double
                    rec.learn(comps, True)
                db.save_weights(rec.weights)
    return {"ok": True, "learned": learned}


class VibeReq(BaseModel):
    vibe: Literal["up", "down", "hold"]


@app.post("/api/sessions/{sid}/vibe")
def set_vibe(sid: str, req: VibeReq):
    db.set_vibe(sid, req.vibe)
    return {"ok": True}


@app.post("/api/admin/reload")
def reload_catalogue():
    """Call after ingesting more tracks so the running server picks them up."""
    with _lock:
        rec.reload(db.analysed_tracks())
    return db.count()
