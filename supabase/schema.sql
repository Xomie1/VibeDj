-- Postgres + pgvector version of the catalogue, for when SQLite gets slow.
-- Vector size matches the MFCC(13) + chroma(12) vector. Change to 512 for CLAP.
create extension if not exists vector;

create table if not exists tracks (
    id text primary key,
    title text, artist text, artist_id text,
    genre text, mood text, tags text,
    duration real, play_count integer, artwork text,
    bpm real, key text,
    energy real, brightness real, percussive real,
    embedding vector(25),
    analysed smallint default 0,
    added_at timestamptz default now()
);

-- Approximate nearest neighbour search for candidate retrieval
create index if not exists tracks_embedding_idx
    on tracks using hnsw (embedding vector_cosine_ops);
create index if not exists tracks_bpm_idx on tracks (bpm);

create table if not exists sessions (
    id text primary key,
    created_at timestamptz default now(),
    vibe text default 'hold'
);

create table if not exists plays (
    id bigserial primary key,
    session_id text references sessions(id),
    track_id text references tracks(id),
    ts timestamptz default now(),
    outcome text,
    listened_sec real,
    components jsonb
);

create table if not exists weights (
    name text primary key,
    value jsonb
);

-- Example: 200 nearest candidates to a context vector within a tempo window.
-- select id, 1 - (embedding <=> $1) as sim
-- from tracks
-- where analysed = 1 and bpm between $2 * 0.92 and $2 * 1.08
-- order by embedding <=> $1
-- limit 200;
