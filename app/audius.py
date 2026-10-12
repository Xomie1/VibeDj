"""Thin client for the Audius REST API (https://api.audius.co/v1).

Every request rotates to the next configured API key, so each key's rate limit
(free plan: 10 req/s, 500k req/month) only carries a share of the traffic.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode

import httpx

from . import config

log = logging.getLogger(__name__)

MAX_WAIT_SEC = 10        # longest a request blocks waiting for a resting key
MAX_REST_SEC = 3600      # cap on the 2, 4, 8... s rest after repeated rate limits
BAD_KEY_REST_SEC = 3600  # a key Audius refuses (401) sits out this long


@dataclass(eq=False)
class Credential:
    """One API key + bearer token pair from https://api.audius.co/plans."""
    api_key: str = ""
    bearer_token: str = ""
    resting_until: float = 0.0
    strikes: int = 0   # rate limits in a row

    @property
    def label(self) -> str:
        # The API key is public, so it can go in logs. Never log the bearer token.
        k = self.api_key
        return f"{k[:6]}...{k[-4:]}" if k else "unnamed key"


class KeyPool:
    """Round-robin over credentials, skipping any that are resting after a 429/401."""

    def __init__(self, pairs, clock=time.monotonic):
        creds = [Credential(k, t) for k, t in pairs if k or t]
        self.configured = len(creds)
        self.creds = creds or [Credential()]   # no keys: one anonymous slot, app_name only
        self.clock = clock
        self._next = 0
        self._lock = threading.Lock()   # ingest shares one client across threads

    def __len__(self) -> int:
        return len(self.creds)

    def pick(self) -> tuple[Credential, float]:
        """Next key in the rotation, and how many seconds to wait before using it."""
        with self._lock:
            now = self.clock()
            n = len(self.creds)
            order = [(self._next + s) % n for s in range(n)]
            i = next((i for i in order if self.creds[i].resting_until <= now),
                     min(order, key=lambda i: self.creds[i].resting_until))
            self._next = (i + 1) % n
            c = self.creds[i]
            return c, max(0.0, c.resting_until - now)

    def bench(self, cred: Credential, seconds: float | None = None) -> float:
        """Take a key out of rotation for `seconds`, default 2, 4, 8... s on repeated rate limits."""
        with self._lock:
            cred.strikes += 1
            if seconds is None:
                seconds = min(2.0 ** cred.strikes, MAX_REST_SEC)
            cred.resting_until = max(cred.resting_until, self.clock() + seconds)
            return seconds

    def ok(self, cred: Credential) -> None:
        with self._lock:
            cred.strikes = 0


def _retry_after(r: httpx.Response) -> float | None:
    try:
        return max(0.0, float(r.headers["Retry-After"]))
    except (KeyError, ValueError):
        return None


class AudiusClient:
    def __init__(self, base: str | None = None, credentials: list[tuple[str, str]] | None = None,
                 app_name: str | None = None, transport: httpx.BaseTransport | None = None):
        self.base = (base or config.AUDIUS_API_BASE).rstrip("/")
        self.app_name = app_name or config.AUDIUS_APP_NAME
        self.pool = KeyPool(config.AUDIUS_CREDENTIALS if credentials is None else credentials)
        self.http = httpx.Client(headers={"Accept": "application/json"}, timeout=20,
                                 follow_redirects=True, transport=transport)

    # ---- low level -------------------------------------------------------
    def _params(self, cred: Credential, params: dict) -> dict:
        out = {"app_name": self.app_name, **{k: v for k, v in params.items() if v is not None}}
        if cred.api_key:
            out["api_key"] = cred.api_key
        return out

    def _send(self, path: str, params: dict, stream: bool = False) -> httpx.Response:
        """GET with key rotation. A key that gets rate limited (429) or refused (401)
        sits out for a while and the request moves on to the next key."""
        for _ in range(len(self.pool) + 2):
            cred, wait = self.pool.pick()
            if wait:
                time.sleep(min(wait, MAX_WAIT_SEC))
            headers = {"Authorization": f"Bearer {cred.bearer_token}"} if cred.bearer_token else {}
            req = self.http.build_request("GET", f"{self.base}{path}",
                                          params=self._params(cred, params), headers=headers)
            r = self.http.send(req, stream=stream)
            if r.status_code == 429:
                secs = self.pool.bench(cred, _retry_after(r))
                log.warning("Audius rate limited %s, resting it %.0fs", cred.label, secs)
            elif r.status_code == 401 and self.pool.configured > 1:
                self.pool.bench(cred, BAD_KEY_REST_SEC)
                log.warning("Audius refused %s (401), check it at https://api.audius.co/plans", cred.label)
            else:
                self.pool.ok(cred)
                return r
            r.close()
        return r   # every attempt was turned away; the caller's raise_for_status reports it

    def _get(self, path: str, **params: Any) -> Any:
        r = self._send(path, params)
        r.raise_for_status()
        return r.json().get("data")

    # ---- catalogue -------------------------------------------------------
    def search_tracks(self, query: str, limit: int = 20) -> list[dict]:
        return self._get("/tracks/search", query=query, limit=limit) or []

    def trending(self, genre: str | None = None, time: str = "month", limit: int = 100) -> list[dict]:
        return self._get("/tracks/trending", genre=genre, time=time, limit=limit) or []

    def get_track(self, track_id: str) -> dict | None:
        return self._get(f"/tracks/{track_id}")

    def stream_url(self, track_id: str) -> str:
        """URL the browser <audio> tag can play directly.

        Carries the next API key in the rotation (public, safe in the browser),
        never a bearer token.
        """
        cred, _ = self.pool.pick()
        return f"{self.base}/tracks/{track_id}/stream?{urlencode(self._params(cred, {}))}"

    def download_head(self, track_id: str, max_bytes: int | None = None) -> bytes:
        """Download the first chunk of the mp3, enough to analyse."""
        max_bytes = max_bytes or config.ANALYSIS_BYTES
        buf = bytearray()
        r = self._send(f"/tracks/{track_id}/stream", {}, stream=True)
        try:
            r.raise_for_status()
            for chunk in r.iter_bytes():
                buf.extend(chunk)
                if len(buf) >= max_bytes:
                    break
        finally:
            r.close()
        return bytes(buf)


def normalise_track(t: dict) -> dict:
    """Pick the fields we care about from an Audius track object."""
    user = t.get("user") or {}
    art = t.get("artwork") or {}
    return {
        "id": t["id"],
        "title": t.get("title", ""),
        "artist": user.get("name") or user.get("handle") or "",
        "artist_id": user.get("id"),
        "genre": t.get("genre"),
        "mood": t.get("mood"),
        "tags": t.get("tags"),
        "duration": t.get("duration"),
        "play_count": t.get("play_count") or 0,
        "artwork": art.get("480x480") or art.get("150x150") if isinstance(art, dict) else None,
        # Some uploads ship with bpm / key metadata; we use it when present.
        "meta_bpm": t.get("bpm"),
        "meta_key": t.get("musical_key"),
        "streamable": t.get("is_streamable", True),
    }
