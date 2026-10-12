# Vibe DJ

A standalone AI DJ. Pick a starting song (or hit "Surprise me") and it keeps
picking the next track to hold the vibe: similar sound, mixable tempo,
compatible key, and energy that moves smoothly instead of jumping around.
It learns from your skips.

Music comes from [Audius](https://audius.co), an open catalogue you can
stream from legally and for free.

## How it works

```
Audius API ──> ingest.py ──> librosa analysis ──> SQLite catalogue
                              (bpm, key, energy,
                               timbre vector)
                                                      │
 Browser player <── FastAPI <── Recommender <─────────┘
      │                              ▲
      └── skip / finish / like ──────┘  (online learning)
```

**Analysis.** Spotify no longer gives new apps tempo/energy/key, so we compute
them ourselves from the first ~60s of each track (skipping the intro):

- BPM via beat tracking, folded into 70 to 180
- Key via chroma + Krumhansl profiles, stored in Camelot notation (8A, 5B...)
- Energy, brightness, percussiveness
- A timbre vector (MFCC + chroma) for "sounds like" similarity

If the uploader supplied BPM or key metadata, that wins over the estimate.

**Picking the next track.** Every candidate gets scored on interpretable components:

| Component | Meaning |
|---|---|
| `sim` | cosine similarity to the session's recent sound (last 5 tracks, recency-weighted) |
| `bpm` | tempo fit with the current track, half/double time counts |
| `key` | Camelot wheel compatibility (same key, neighbours, relative major/minor) |
| `energy` | fit to the *target* energy: current level + recent trend + your vibe setting |
| `genre` | share of recent tracks in the same genre |
| `fresh_artist` | 0 if the artist played in the last 3 tracks |
| `skip_sim` | similarity to stuff you just skipped (negative weight) |
| `popularity` | log play count, a light prior |

Score = weighted sum. It samples from the top 10 with a softmax so sessions
don't come out identical every time.

**Learning.** Skipping in the first 30 seconds is a negative label, finishing
is positive, a like counts double. Each one runs a single SGD step of logistic
regression on the components, so the weights drift toward what you actually
care about (maybe you don't mind key clashes but hate tempo jumps). Weights are
saved, so it gets better over time.

**Vibe control.** "Turn up" / "Wind down" pushes the target energy up or down
a notch per track. "Hold" keeps it steady.

**Sections.** Like Spotify's DJ, a session plays in themed blocks of 4 to 6
tracks. A theme is a genre plus an energy band (Chilled / Warm / Peak-time,
split into thirds of your catalogue). Picks stay in the section's genre; when the
section is done, the next one starts on the closest-sounding *different* theme,
so the move is smooth. "Switch it up" jumps straight to a section in another
genre, picked from the 40% of those tracks that sound least like what's playing. Every pick
also logs the probability it was chosen with, so later we can evaluate and train
the picker off-policy (the way Spotify's BaRT does).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env    # optional: add free Audius API keys for higher limits
```

Each key from https://api.audius.co/plans has its own rate limit (free plan:
10 req/s, 500k req/month). Put as many pairs as you like in `.env`
(`AUDIUS_API_KEY_1` + `AUDIUS_BEARER_TOKEN_1`, `_2`, `_3`...) and the client
rotates through them, one key per request. A key that gets rate limited (429)
or refused (401) sits out for a while and the next one takes over. Only the
public API key reaches the browser; bearer tokens stay on the server.
`GET /api/stats` shows how many keys were loaded.

You need ffmpeg installed for mp3 decoding if your libsndfile is old
(`brew install ffmpeg` / `sudo apt install ffmpeg` / `winget install ffmpeg`).

## Build the catalogue

```bash
python -m scripts.ingest                                  # trending across ~13 genres
python -m scripts.ingest --genres House Afrobeat --limit 100
python -m scripts.ingest --search "amapiano" "lofi study"
```

Start small (the default gives roughly 500 to 1,000 tracks). Analysis takes a
second or two per track. Re-running is safe; it only analyses new tracks.

## Run

```bash
uvicorn app.main:app --reload
```

Open http://localhost:8000.

## API

| Method | Path | |
|---|---|---|
| GET | `/api/search?q=` | search the local catalogue |
| POST | `/api/sessions` | `{seed_track_id?}` starts a session, returns first track |
| POST | `/api/sessions/{id}/next` | next pick + reason + scoring components; `track.section` = `{n, label, kind, position, length}` |
| POST | `/api/sessions/{id}/switch` | "Switch it up": next pick opens a new section far from the current one |
| POST | `/api/sessions/{id}/event` | `{track_id, outcome: skip/complete/like/switch/dropped, listened_sec}` |
| POST | `/api/sessions/{id}/vibe` | `{vibe: up/down/hold}` |
| GET | `/api/stats` | catalogue counts, Audius keys loaded, current learned weights |
| POST | `/api/admin/reload` | reload catalogue after ingesting |

## Tests

```bash
pytest -q
```

Runs on a synthetic catalogue, no network needed.

## Roadmap

1. **Better embeddings.** Swap the MFCC vector for CLAP or MERT embeddings. Biggest quality jump.
2. **Postgres + pgvector.** `supabase/schema.sql` has the schema. Needed once the catalogue is in the tens of thousands.
3. **Per-user weights** instead of one global set (needs accounts, "Log in with Audius" works).
4. **DJ voice.** An LLM writes a short line from `explain()` + track info, TTS speaks it between tracks.
5. **Crossfades** with two audio elements, timed off the BPM.
6. **Offline eval.** Replay logged sessions and measure skip rate per weight setting.

Always credit artists; Audius asks that apps do.
