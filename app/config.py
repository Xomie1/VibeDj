"""Settings, read from environment / .env."""
import os
import re
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

ROOT = Path(__file__).resolve().parent.parent

AUDIUS_API_BASE = os.getenv("AUDIUS_API_BASE", "https://api.audius.co/v1")
AUDIUS_APP_NAME = os.getenv("AUDIUS_APP_NAME", "vibe-dj")


def audius_credentials(env=os.environ) -> list[tuple[str, str]]:
    """(api_key, bearer_token) pairs from AUDIUS_API_KEY_1 / AUDIUS_BEARER_TOKEN_1, _2, _3...

    Free keys from https://api.audius.co/plans. Each has its own rate limit, so the
    client rotates through all of them. The unnumbered AUDIUS_API_KEY /
    AUDIUS_BEARER_TOKEN still work as one more pair.
    """
    pairs = {}
    for name, value in env.items():
        m = re.fullmatch(r"AUDIUS_(API_KEY|BEARER_TOKEN)(?:_(\d+))?", name)
        if m and value.strip():
            n = int(m.group(2)) if m.group(2) else 0
            pairs.setdefault(n, {})[m.group(1)] = value.strip()
    return [(p.get("API_KEY", ""), p.get("BEARER_TOKEN", "")) for _, p in sorted(pairs.items())]


AUDIUS_CREDENTIALS = audius_credentials()

DB_PATH = Path(os.getenv("VIBE_DB_PATH", ROOT / "data" / "vibe.db"))

# How much of each track to download for analysis (bytes). ~1.5MB is roughly 60-90s of mp3.
ANALYSIS_BYTES = int(os.getenv("ANALYSIS_BYTES", 1_500_000))
