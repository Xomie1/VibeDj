"""Musical key handling: Camelot wheel + harmonic-mixing compatibility.

Camelot notation: number 1-12 around the circle of fifths, A = minor, B = major.
Moving +/-1 on the number, or switching A<->B on the same number, sounds smooth.
"""
from __future__ import annotations

import re

PITCHES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_FLATS = {"Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#"}

# pitch class -> camelot number
_MAJOR = {0: 8, 7: 9, 2: 10, 9: 11, 4: 12, 11: 1, 6: 2, 1: 3, 8: 4, 3: 5, 10: 6, 5: 7}
_MINOR = {9: 8, 4: 9, 11: 10, 6: 11, 1: 12, 8: 1, 3: 2, 10: 3, 5: 4, 0: 5, 7: 6, 2: 7}


def to_camelot(pitch_class: int, minor: bool) -> str:
    table = _MINOR if minor else _MAJOR
    return f"{table[pitch_class % 12]}{'A' if minor else 'B'}"


def parse_key(text: str | None) -> str | None:
    """Turn things like 'A minor', 'F# Major', 'Bbm', '8A' into Camelot ('8A')."""
    if not text:
        return None
    s = text.strip()
    m = re.fullmatch(r"(1[0-2]|[1-9])\s*([ABab])", s)
    if m:
        return f"{int(m.group(1))}{m.group(2).upper()}"
    m = re.fullmatch(r"([A-Ga-g])([#b♯♭]?)\s*(maj(or)?|min(or)?|m)?", s, flags=re.I)
    if not m:
        return None
    note = m.group(1).upper() + m.group(2).replace("♯", "#").replace("♭", "b")
    note = _FLATS.get(note, note)
    if note not in PITCHES:
        return None
    qual = (m.group(3) or "").lower()
    minor = qual.startswith("min") or qual == "m"
    return to_camelot(PITCHES.index(note), minor)


def key_compat(a: str | None, b: str | None) -> float:
    """1.0 = same key, ~0.8 = neighbour / relative, lower = clashing. 0.5 if unknown."""
    if not a or not b:
        return 0.5
    na, la = int(a[:-1]), a[-1]
    nb, lb = int(b[:-1]), b[-1]
    step = min((na - nb) % 12, (nb - na) % 12)
    if step == 0 and la == lb:
        return 1.0
    if step == 0:            # relative major/minor
        return 0.85
    if step == 1 and la == lb:  # neighbour on the wheel
        return 0.85
    if step == 2 and la == lb:  # energy boost move
        return 0.6
    if step == 1:
        return 0.5
    return max(0.0, 0.4 - 0.05 * step)
