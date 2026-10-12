"""Thin client for the Audius REST API (https://api.audius.co/v1)."""
from __future__ import annotations

import time
from typing import Any

import httpx

from . import config


class AudiusClient:
    def __init__(self, base: str | None = None, token: str | None = None, app_name: str | None = None):
        self.base = (base or config.AUDIUS_API_BASE).rstrip("/")
        self.app_name = app_name or config.AUDIUS_APP_NAME
        headers = {"Accept": "application/json"}
        token = token if token is not None else config.AUDIUS_BEARER_TOKEN
        if token:
            headers["Authorization"] = f"Bearer {token}"
        self.http = httpx.Client(headers=headers, timeout=20, follow_redirects=True)

    # ---- low level -------------------------------------------------------
    def _get(self, path: str, **params: Any) -> Any:
        params = {k: v for k, v in params.items() if v is not None}
        params.setdefault("app_name", self.app_name)
        for attempt in range(3):
            r = self.http.get(f"{self.base}{path}", params=params)
            if r.status_code == 429:  # rate limited, back off
                time.sleep(2 ** attempt)
                continue
            r.raise_for_status()
            return r.json().get("data")
        r.raise_for_status()

    # ---- catalogue -------------------------------------------------------
    def search_tracks(self, query: str, limit: int = 20) -> list[dict]:
        return self._get("/tracks/search", query=query, limit=limit) or []

    def trending(self, genre: str | None = None, time: str = "month", limit: int = 100) -> list[dict]:
        return self._get("/tracks/trending", genre=genre, time=time, limit=limit) or []

    def get_track(self, track_id: str) -> dict | None:
        return self._get(f"/tracks/{track_id}")

    def stream_url(self, track_id: str) -> str:
        """URL the browser <audio> tag can play directly."""
        return f"{self.base}/tracks/{track_id}/stream?app_name={self.app_name}"

    def download_head(self, track_id: str, max_bytes: int | None = None) -> bytes:
        """Download the first chunk of the mp3, enough to analyse."""
        max_bytes = max_bytes or config.ANALYSIS_BYTES
        buf = bytearray()
        with self.http.stream("GET", self.stream_url(track_id)) as r:
            r.raise_for_status()
            for chunk in r.iter_bytes():
                buf.extend(chunk)
                if len(buf) >= max_bytes:
                    break
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
