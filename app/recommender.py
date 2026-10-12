"""The 'keep the vibe going' engine.

For every candidate track we compute a handful of interpretable components
(sound similarity, tempo fit, key fit, energy fit...), then combine them with
weights. The weights start as sensible defaults and get nudged by a tiny online
logistic regression every time the listener skips or finishes a track.

Why components + learned weights instead of one big model?
  - works with zero training data on day one
  - every pick is explainable ("same key, +4 BPM, slightly more energy")
  - the skip signal has something concrete to learn from
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .keys import key_compat

COMPONENTS = ["sim", "bpm", "key", "energy", "genre", "fresh_artist", "skip_sim", "popularity"]

DEFAULT_WEIGHTS = {
    "bias": 0.0,
    "sim": 2.0,          # sounds like what's been playing
    "bpm": 1.5,          # tempo is mixable
    "key": 0.8,          # harmonically compatible
    "energy": 1.5,       # matches where the energy should go next
    "genre": 0.5,
    "fresh_artist": 0.6, # avoid the same artist back to back
    "skip_sim": -1.5,    # away from stuff they just skipped
    "popularity": 0.3,
}

SKIP_THRESHOLD_SEC = 30.0   # bailing before this counts as a skip
CONTEXT_LEN = 5             # how many recent tracks define the vibe
DECAY = 0.6                 # most recent track matters most


def bpm_fit(a: float | None, b: float | None) -> float:
    """1.0 for matching tempo, falls off past ~6%. Half/double time counts as a match."""
    if not a or not b:
        return 0.5
    best = min(abs(b * r - a) / a for r in (0.5, 1.0, 2.0))
    return math.exp(-((best / 0.06) ** 2))


@dataclass
class SessionState:
    played: list[dict] = field(default_factory=list)    # tracks that stuck (completed / liked / playing)
    skipped: list[dict] = field(default_factory=list)
    vibe: str = "hold"                                   # 'up' | 'down' | 'hold'


class Recommender:
    def __init__(self, tracks: list[dict], weights: dict | None = None, lr: float = 0.05):
        self.lr = lr
        self.weights = {**DEFAULT_WEIGHTS, **(weights or {})}
        self.reload(tracks)

    def reload(self, tracks: list[dict]) -> None:
        self.tracks = tracks
        self.by_id = {t["id"]: i for i, t in enumerate(tracks)}
        if tracks:
            self.V = np.stack([t["vector"] for t in tracks]).astype(np.float32)
            pc = np.array([t.get("play_count") or 0 for t in tracks], dtype=np.float64)
            self.pop = np.log1p(pc) / (np.log1p(pc.max()) or 1)
        else:
            self.V = np.zeros((0, 1), np.float32)
            self.pop = np.zeros(0)

    # ---- context ---------------------------------------------------------
    def context_vector(self, recent: list[dict]) -> np.ndarray | None:
        if not recent:
            return None
        vecs, ws = [], []
        for i, t in enumerate(reversed(recent[-CONTEXT_LEN:])):
            vecs.append(self.V[self.by_id[t["id"]]])
            ws.append(DECAY ** i)
        c = np.average(np.stack(vecs), axis=0, weights=ws)
        return c / (np.linalg.norm(c) + 1e-9)

    def target_energy(self, state: SessionState) -> float | None:
        es = [t["energy"] for t in state.played[-3:] if t.get("energy") is not None]
        if not es:
            return None
        last = es[-1]
        drift = (es[-1] - es[0]) / max(len(es) - 1, 1)   # carry the current trend a little
        nudge = {"up": 0.08, "down": -0.08}.get(state.vibe, 0.0)
        step = float(np.clip(0.5 * drift + nudge, -0.15, 0.15))
        return float(np.clip(last + step, 0.0, 1.0))

    # ---- scoring ---------------------------------------------------------
    def components(self, state: SessionState) -> tuple[np.ndarray, list[int]]:
        """Return (n_candidates x n_components) matrix and candidate indices."""
        seen = {t["id"] for t in state.played} | {t["id"] for t in state.skipped}
        cand = [i for i, t in enumerate(self.tracks) if t["id"] not in seen]
        if not cand:
            return np.zeros((0, len(COMPONENTS))), []

        last = state.played[-1] if state.played else None
        ctx = self.context_vector(state.played)
        tgt_e = self.target_energy(state)
        recent_artists = {t.get("artist_id") or t.get("artist") for t in state.played[-3:]}
        recent_genres = [t.get("genre") for t in state.played[-CONTEXT_LEN:] if t.get("genre")]

        Vc = self.V[cand]
        sim = (Vc @ ctx + 1) / 2 if ctx is not None else np.full(len(cand), 0.5)
        if state.skipped:
            S = self.V[[self.by_id[t["id"]] for t in state.skipped[-5:]]]
            skip_sim = ((Vc @ S.T).max(axis=1) + 1) / 2
        else:
            skip_sim = np.zeros(len(cand))

        rows = []
        for j, i in enumerate(cand):
            t = self.tracks[i]
            rows.append([
                sim[j],
                bpm_fit(last.get("bpm"), t.get("bpm")) if last else 0.5,
                key_compat(last.get("key"), t.get("key")) if last else 0.5,
                1 - min(abs((t.get("energy") or 0.5) - tgt_e) / 0.3, 1) if tgt_e is not None else 0.5,
                (recent_genres.count(t.get("genre")) / len(recent_genres)) if recent_genres else 0.5,
                0.0 if (t.get("artist_id") or t.get("artist")) in recent_artists else 1.0,
                skip_sim[j],
                self.pop[i],
            ])
        return np.asarray(rows, dtype=np.float64), cand

    def score(self, X: np.ndarray) -> np.ndarray:
        w = np.array([self.weights[c] for c in COMPONENTS])
        return X @ w + self.weights["bias"]

    def next_track(self, state: SessionState, temperature: float = 0.15, top_k: int = 10,
                   rng: np.random.Generator | None = None) -> tuple[dict, dict] | None:
        """Pick the next track. Samples from the top-k so it doesn't play the same set every time."""
        X, cand = self.components(state)
        if not cand:
            return None
        s = self.score(X)
        top = np.argsort(-s)[:top_k]
        p = np.exp((s[top] - s[top].max()) / max(temperature, 1e-6))
        p /= p.sum()
        rng = rng or np.random.default_rng()
        pick = int(rng.choice(top, p=p))
        comps = {c: round(float(v), 4) for c, v in zip(COMPONENTS, X[pick])}
        return self.tracks[cand[pick]], comps

    # ---- learning from skips --------------------------------------------
    def learn(self, comps: dict, liked: bool) -> None:
        """One SGD step of logistic regression: P(listened) = sigmoid(w . x)."""
        x = np.array([comps[c] for c in COMPONENTS])
        w = np.array([self.weights[c] for c in COMPONENTS])
        z = float(x @ w + self.weights["bias"])
        p = 1 / (1 + math.exp(-np.clip(z, -30, 30)))
        g = (1.0 if liked else 0.0) - p
        for c, xi in zip(COMPONENTS, x):
            self.weights[c] += self.lr * g * xi
        self.weights["bias"] += self.lr * g

    @staticmethod
    def explain(comps: dict, prev: dict | None, track: dict) -> str:
        """Human-readable reason, handy for the UI and later for the DJ voice."""
        bits = []
        if prev and prev.get("bpm") and track.get("bpm"):
            d = track["bpm"] - prev["bpm"]
            bits.append(f"{track['bpm']:.0f} BPM ({d:+.0f})")
        if comps.get("key", 0) >= 0.85 and track.get("key"):
            bits.append(f"key {track['key']} mixes clean")
        if comps.get("sim", 0) > 0.8:
            bits.append("very close sound")
        if prev and prev.get("energy") is not None and track.get("energy") is not None:
            de = track["energy"] - prev["energy"]
            if de > 0.05:
                bits.append("lifting the energy")
            elif de < -0.05:
                bits.append("easing off")
        return ", ".join(bits) or "keeping the vibe"
