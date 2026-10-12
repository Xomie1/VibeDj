# Vibe DJ: context for Claude Code

Standalone AI DJ on the Audius open catalogue. Goal: get as close as practical
to Spotify's DJ experience (personalised stream, sections, spoken commentary,
voice/text requests) without Spotify's data or licences.

## Commands
- Install: `pip install -r requirements.txt` (ffmpeg needed for mp3 decode on some systems)
- Build catalogue: `python -m scripts.ingest` (see `--help`)
- Run: `uvicorn app.main:app --reload`, then http://localhost:8000
- Tests: `pytest -q` (synthetic catalogue, no network)

## Layout
- `app/audius.py`: REST client for https://api.audius.co/v1. Rotates through the API key + bearer token pairs in `.env`
  (`AUDIUS_API_KEY_n` / `AUDIUS_BEARER_TOKEN_n`), resting keys that hit 429/401. Bearer tokens never go in browser URLs.
- `app/features.py`: librosa analysis, BPM / Camelot key / energy / brightness / percussive + 25-d timbre vector.
- `app/keys.py`: key parsing, Camelot wheel compatibility.
- `app/recommender.py`: component scoring (sim, bpm, key, energy, genre, fresh_artist, skip_sim, popularity), softmax top-k sampling, online logistic regression on skip/complete/like.
  Sections: a `Section` theme (genre + catalogue-relative energy band) holds for 4 to 6 tracks; `next_track(switch=True)` jumps far. Each `Pick` carries its sampling propensity (logged in `plays.propensity`).
- `app/db.py`: SQLite. `supabase/schema.sql` is the pgvector version for later.
- `app/main.py`: FastAPI session API + serves `static/index.html` (vanilla JS player).

## Status
- All tests pass offline.
- The live Audius calls (ingest, streaming) were written against the docs but NOT yet run against the real API. First job: run ingest, fix whatever breaks (field names, redirects, genre names).

## Conventions
- Keep the recommender explainable: every pick returns its components and a `reason` string.
- New scoring signals go in `COMPONENTS` + `DEFAULT_WEIGHTS` so the learner can weight them.
- Add a test in `tests/` for every new behaviour; keep tests network-free.

## Roadmap: Spotify DJ parity (in priority order)
Evidence for each Spotify behaviour is in `docs/spotify-dj-notes.md`.

1. **Sections.** (Done.) Spotify's DJ plays themed blocks and the DJ button jumps to the next one. Group picks into sections of 4 to 6 tracks around a theme (genre, mood, energy band). Add `POST /sessions/{id}/switch` that starts a new section far from the current one (low `sim`), and a "Switch it up" button.
2. **Requests.** Text box (then Web Speech API for voice). An LLM turns "cry-in-the-car songs" into a structured filter `{genres, mood, energy_range, bpm_range, novelty}`; the recommender applies it as hard filters + a target. Keep the parser behind an interface so the model is swappable.
3. **Commentary.** Between sections, generate 1 to 2 short spoken lines from ONLY provided facts (track/artist metadata, `reason`, section theme). Spotify grounds its LLM in editor-written facts; we ground in metadata. No invented facts.
4. **Voice.** TTS for the commentary (start with an open model like Kokoro or Piper; stock voice only, no voice cloning of real people). Duck the music under the voice.
5. **Long-term taste profile.** Persist likes/completes across sessions per user; mix "rediscover" picks (liked before, not played recently) with new uploads.
6. **Collaborative signal.** Audius exposes users' favourites/reposts. Build an item-item co-occurrence graph from them, add `co_listen` as a component.
7. **Graded skips.** Replace the binary 30 s threshold with graded outcomes (very brief / brief / most / full), as in Spotify's MSSD dataset.
8. **Exploration.** Epsilon-greedy slot every N tracks with propensity logging, so the learner gets unbiased data.
9. **Crossfades + better embeddings** (CLAP/MERT), then Postgres/pgvector.
