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
from .recommender import SKIP_THRESHOLD_SEC, Recommender, Section, SessionState

app = FastAPI(title="Vibe DJ")
db = DB()
audius = AudiusClient()
rec = Recommender(db.analysed_tracks(), weights=db.load_weights())
_lock = threading.Lock()   # recommender weights are shared across sessions

STATIC = config.ROOT / "static"
app.mount("/static", StaticFiles(directory=STATIC), name="static")


# ---- helpers ---------------------------------------------------------------
def build_state(sid: str) -> tuple[SessionState, int | None]:
    """Rebuild the session from the play log. Returns the state and the current section number."""
    sess = db.get_session(sid)
    if not sess:
        raise HTTPException(404, "session not found")
    sec = db.current_section(sid)
    st = SessionState(vibe=sess["vibe"])
    if sec:
        st.section = Section(sec["genre"], sec["band"], sec["length"], sec["kind"])
    for h in db.history(sid):
        i = rec.by_id.get(h["track_id"])
        if i is None:
            continue
        t = rec.tracks[i]
        if h["outcome"] == "dropped":   # prefetched but never played
            continue
        in_sec = sec is not None and h["section"] == sec["n"]
        st.section_served += in_sec
        if h["outcome"] == "skip" and (h["listened_sec"] or 0) < SKIP_THRESHOLD_SEC:
            st.skipped.append(t)
        else:
            st.played.append(t)
            if in_sec:
                st.section_played.append(t)
    return st, sec["n"] if sec else None


def section_info(n: int, sec: Section, position: int) -> dict:
    return {"n": n, "label": sec.label, "kind": sec.kind, "position": position, "length": sec.length}


def public(t: dict, reason: str | None = None, section: dict | None = None) -> dict:
    keep = ("id", "title", "artist", "genre", "mood", "bpm", "key", "energy", "artwork", "duration")
    out = {k: t.get(k) for k in keep}
    out["stream_url"] = audius.stream_url(t["id"])
    out["link"] = f"https://audius.co/tracks/{t['id']}"
    if reason:
        out["reason"] = reason
    if section:
        out["section"] = section
    return out


def serve_next(sid: str, switch: bool = False) -> dict:
    st, n = build_state(sid)
    with _lock:
        pick = rec.next_track(st, switch=switch)
    if not pick:
        raise HTTPException(410, "ran out of tracks, ingest more")
    sec, position = st.section, st.section_served + 1
    if pick.section:   # this pick opens a new section
        sec, position = pick.section, 1
        n = db.start_section(sid, sec)
    db.log_queued(sid, pick.track["id"], pick.components, section=n, propensity=pick.propensity)
    prev = st.played[-1] if st.played else None
    info = section_info(n, sec, position) if sec else None
    return {"track": public(pick.track, rec.explain(pick.components, prev, pick.track), info),
            "components": pick.components}


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
    sec = rec.new_section(first, "start")
    n = db.start_section(sid, sec)
    db.log_queued(sid, first["id"], None, section=n)
    return {"session_id": sid, "track": public(first, "your starting point", section_info(n, sec, 1))}


@app.post("/api/sessions/{sid}/next")
def next_track(sid: str):
    return serve_next(sid)


@app.post("/api/sessions/{sid}/switch")
def switch_section(sid: str):
    """Jump to a new section far from what's playing (the DJ button)."""
    return serve_next(sid, switch=True)


class EventReq(BaseModel):
    track_id: str
    outcome: Literal["skip", "complete", "like", "switch", "dropped"]   # switch: left via "Switch it up", not learned from
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
