"""Settings, read from environment / .env."""
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

ROOT = Path(__file__).resolve().parent.parent

AUDIUS_API_BASE = os.getenv("AUDIUS_API_BASE", "https://api.audius.co/v1")
# Free key from https://api.audius.co/plans. Optional, but gives higher rate limits.
AUDIUS_BEARER_TOKEN = os.getenv("AUDIUS_BEARER_TOKEN", "")
AUDIUS_APP_NAME = os.getenv("AUDIUS_APP_NAME", "vibe-dj")

DB_PATH = Path(os.getenv("VIBE_DB_PATH", ROOT / "data" / "vibe.db"))

# How much of each track to download for analysis (bytes). ~1.5MB is roughly 60-90s of mp3.
ANALYSIS_BYTES = int(os.getenv("ANALYSIS_BYTES", 1_500_000))
