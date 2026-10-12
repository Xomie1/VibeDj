"""Build the catalogue: pull tracks from Audius, then analyse their audio.

    python -m scripts.ingest                      # trending across default genres
    python -m scripts.ingest --genres House Techno --limit 100
    python -m scripts.ingest --search "lofi beats" "afrobeats"
    python -m scripts.ingest --analyse-only       # just process what's queued

Safe to re-run: already-analysed tracks are skipped.
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.audius import AudiusClient, normalise_track  # noqa: E402
from app.db import DB  # noqa: E402
from app.features import analyse_mp3  # noqa: E402
from app.keys import parse_key  # noqa: E402

DEFAULT_GENRES = [
    "Electronic", "House", "Deep House", "Techno", "Hip-Hop/Rap", "R&B/Soul",
    "Lo-Fi", "Pop", "Alternative", "Ambient", "Afrobeat", "Dancehall", "Drum & Bass",
]


def collect(client: AudiusClient, db: DB, genres, searches, limit, meta: dict) -> int:
    n = 0
    batches = []
    for g in genres:
        for window in ("month", "allTime"):
            try:
                batches.append((f"trending {g}/{window}", client.trending(genre=g, time=window, limit=limit)))
            except Exception as e:  # one bad genre shouldn't kill the run
                print(f"  ! {g}/{window}: {e}")
    for q in searches:
        try:
            batches.append((f"search '{q}'", client.search_tracks(q, limit=min(limit, 50))))
        except Exception as e:
            print(f"  ! search {q}: {e}")

    for label, tracks in batches:
        added = 0
        for raw in tracks:
            t = normalise_track(raw)
            if not t["streamable"] or not t["duration"] or t["duration"] < 60:
                continue  # skip snippets and non-streamable uploads
            db.upsert_track(t)
            meta[t["id"]] = (t["meta_bpm"], t["meta_key"])
            added += 1
        n += added
        print(f"  {label}: {added} tracks")
    return n


def analyse_one(client: AudiusClient, track_id: str, meta: dict):
    data = client.download_head(track_id)
    f = analyse_mp3(data)
    bpm, key = meta.get(track_id, (None, None))
    # Uploader-supplied metadata beats our estimate when it exists
    if bpm and 60 <= float(bpm) <= 200:
        f["bpm"] = float(bpm)
    if parse_key(key):
        f["key"] = parse_key(key)
    return f


def analyse_pending(client: AudiusClient, db: DB, meta: dict, workers: int, max_tracks: int) -> None:
    ids = db.unanalysed(max_tracks)
    print(f"Analysing {len(ids)} tracks with {workers} workers...")
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(analyse_one, client, tid, meta): tid for tid in ids}
        for fut in as_completed(futs):
            tid = futs[fut]
            try:
                db.save_features(tid, fut.result())
            except Exception as e:
                db.mark_failed(tid)
                print(f"  ! {tid}: {type(e).__name__}: {e}")
            done += 1
            if done % 25 == 0:
                print(f"  {done}/{len(ids)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--genres", nargs="*", default=DEFAULT_GENRES)
    ap.add_argument("--search", nargs="*", default=[])
    ap.add_argument("--limit", type=int, default=50, help="tracks per genre/window")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-analyse", type=int, default=2000)
    ap.add_argument("--analyse-only", action="store_true")
    args = ap.parse_args()

    client, db, meta = AudiusClient(), DB(), {}
    if not args.analyse_only:
        print("Collecting tracks from Audius...")
        n = collect(client, db, args.genres, args.search, args.limit, meta)
        print(f"Collected {n} (duplicates merged).")
    analyse_pending(client, db, meta, args.workers, args.max_analyse)
    print("Catalogue:", db.count())
    print("If the server is running, POST /api/admin/reload to pick up new tracks.")


if __name__ == "__main__":
    main()
