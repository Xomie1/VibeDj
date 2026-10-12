"""Audio feature extraction.

Replaces what Spotify used to give you (tempo, energy, key...) by analysing the
audio ourselves with librosa. Produces:
  - bpm, camelot key
  - energy, brightness, percussiveness (0..1, roughly)
  - a 'timbre' vector (MFCC + chroma means) used for sound similarity

Swap `timbre_vector` for a learned embedding (CLAP, MERT) later for better results.
"""
from __future__ import annotations

import tempfile

import numpy as np

from .keys import to_camelot

SR = 22050

# Krumhansl-Schmuckler key profiles
_MAJ = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MIN = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def load_mp3_bytes(data: bytes, skip_seconds: float = 15.0, duration: float = 45.0) -> np.ndarray:
    """Decode an mp3 chunk. Skips the intro since intros lie about the vibe."""
    import librosa
    with tempfile.NamedTemporaryFile(suffix=".mp3") as f:
        f.write(data)
        f.flush()
        y, _ = librosa.load(f.name, sr=SR, mono=True)
    start = int(skip_seconds * SR)
    if len(y) > start + SR * 10:   # only skip if enough audio remains
        y = y[start:]
    return y[: int(duration * SR)]


def estimate_key(chroma_mean: np.ndarray) -> str:
    best, best_score = (0, False), -np.inf
    for pc in range(12):
        for minor, prof in ((False, _MAJ), (True, _MIN)):
            score = np.corrcoef(chroma_mean, np.roll(prof, pc))[0, 1]
            if score > best_score:
                best, best_score = (pc, minor), score
    return to_camelot(*best)


def analyse(y: np.ndarray, sr: int = SR) -> dict:
    import librosa

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    bpm = float(np.atleast_1d(tempo)[0])
    # Fold silly tempos into a DJ-friendly 70-180 range
    while bpm and bpm < 70:
        bpm *= 2
    while bpm > 180:
        bpm /= 2

    rms = librosa.feature.rms(y=y)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    harm, perc = librosa.effects.hpss(y)
    chroma = librosa.feature.chroma_cqt(y=harm, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    chroma_mean = chroma.mean(axis=1)
    energy = float(np.clip(rms.mean() / 0.25, 0, 1))
    brightness = float(np.clip(centroid.mean() / 4000.0, 0, 1))
    percussive = float(np.sum(perc ** 2) / (np.sum(y ** 2) + 1e-9))

    timbre = np.concatenate([
        mfcc.mean(axis=1) / 100.0,   # coarse scaling so neither block dominates
        chroma_mean / (chroma_mean.sum() + 1e-9),
    ])
    timbre = timbre / (np.linalg.norm(timbre) + 1e-9)

    return {
        "bpm": round(bpm, 1),
        "key": estimate_key(chroma_mean),
        "energy": round(energy, 3),
        "brightness": round(brightness, 3),
        "percussive": round(float(np.clip(percussive, 0, 1)), 3),
        "vector": timbre.astype(np.float32),
    }


def analyse_mp3(data: bytes) -> dict:
    return analyse(load_mp3_bytes(data))
