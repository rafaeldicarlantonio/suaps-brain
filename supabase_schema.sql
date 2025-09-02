-- supabase_schema.sql
-- Postgres (Supabase). Ids are UUIDs via pgcrypto; timestamps are tz-aware.
-- This file is idempotent: all CREATEs use IF NOT EXISTS.

-- Extensions
create extension if not exists "pgcrypto";

-- ---------------------------------------------------------------------
-- Helpers: updated_at trigger
-- ---------------------------------------------------------------------
create or replace function set_updated_at()
returns trigger language plpgsql as $$
begin
  new.updated_at = now();
  return new;
end $$;

-- =====================================================================
-- USERS (optional; analytics, ownership – not used for auth in MVP)
-- =====================================================================
create table if not exists public.users (
  id         uuid primary key default gen_random_uuid(),
  email      text unique,
  name       text,
  role       text check (role in ('researcher','staff','director','admin')) default 'staff',
  created_at timestamptz not null default now()
);

-- =====================================================================
-- SESSIONS (per chat thread)
-- =====================================================================
create table if not exists public.sessions (
  id         uuid primary key default gen_random_uuid(),
  user_id    uuid references public.users(id),
  title      text,
  created_at timestamptz not null default now()
);

-- =====================================================================
-- MESSAGES (chat turns + simple observability)
-- =====================================================================
create table if not exists public.messages (
  id         uuid primary key default gen_random_uuid(),
  session_id uuid not null references public.sessions(id) on delete cascade,
  role       text not null check (role in ('user','assistant','system')),
  content    text not null,
  tokens     int,
  latency_ms int,
  model      text,
  created_at timestamptz not null default now()
);
create index if not exists idx_messages_session_created
  on public.messages(session_id, created_at);

-- =====================================================================
-- FILES (uploaded or synced; normalized text lives here)
-- =====================================================================
create table if not exists public.files (
  id            uuid primary key default gen_random_uuid(),
  filename      text,
  mime_type     text,
  bytes         int,
  storage_url   text,
  text_extracted text,
  created_at    timestamptz not null default now()
);
create index if not exists idx_files_created on public.files(created_at);

-- =====================================================================
-- MEMORIES (canonical stored knowledge; chunk-level or note-level)
-- =====================================================================
create table if not exists public.memories (
  id              uuid primary key default gen_random_uuid(),
  type            text not null check (type in ('episodic','semantic','procedural')),
  title           text,
  text            text not null,
  tags            text[] default '{}',
  source          text,               -- 'upload'|'ingest'|'chat'|'wiki'...
  file_id         uuid references public.files(id) on delete set null,
  session_id      uuid references public.sessions(id) on delete set null,
  author_user_id  uuid references public.users(id),
  role_view       text[] default '{}', -- e.g. {'researcher','director'}
  created_at      timestamptz not null default now(),
  updated_at      timestamptz not null default now(),
  dedupe_hash     char(64) unique,     -- sha256 of normalized text
  embedding_id    text                 -- pinecone id for reverse lookup
);
create index if not exists idx_memories_type on public.memories(type);
create index if not exists idx_memories_tags on public.memories using gin(tags);
create index if not exists idx_memories_created on public.memories(created_at);

drop trigger if exists trg_memories_set_updated on public.memories;
create trigger trg_memories_set_updated
  before update on public.memories
  for each row execute function set_updated_at();

-- =====================================================================
-- ENTITIES (graph)
-- =====================================================================
create table if not exists public.entities (
  id         uuid primary key default gen_random_uuid(),
  name       text not null,
  type       text not null check (type in ('person','org','project','artifact','concept')),
  canonical  boolean default false,
  created_at timestamptz not null default now()
);
create unique index if not exists uq_entities_name_type on public.entities(name, type);

-- Memory ↔ Entity (mentions)
create table if not exists public.entity_mentions (
  entity_id uuid not null references public.entities(id) on delete cascade,
  memory_id uuid not null references public.memories(id) on delete cascade,
  weight    real default 1.0,
  primary key (entity_id, memory_id)
);

-- Entity ↔ Entity (edges)
create table if not exists public.entity_edges (
  src    uuid not null references public.entities(id) on delete cascade,
  dst    uuid not null references public.entities(id) on delete cascade,
  rel    text not null, -- e.g. 'sponsors','decides','derived_from','contradicts'
  weight real default 1.0,
  primary key (src, dst, rel)
);

-- =====================================================================
-- TOOL RUNS (observability for autosave/red-team/ingest steps)
-- =====================================================================
create table if not exists public.tool_runs (
  id          uuid primary key default gen_random_uuid(),
  name        text not null,   -- 'redteam','autosave_extract','ingest','retrieval'...
  input_json  jsonb,
  output_json jsonb,
  success     boolean,
  latency_ms  int,
  created_at  timestamptz not null default now()
);
create index if not exists idx_tool_runs_created on public.tool_runs(created_at);

-- =====================================================================
-- Convenience views (debug)
-- =====================================================================
create or replace view public.debug_recent_memories as
select id, type, title, tags, source, created_at
from public.memories
order by created_at desc
limit 200;

-- =====================================================================
-- Optional: simple FTS (disabled by default; uncomment to enable)
-- =====================================================================
-- create extension if not exists "unaccent";
-- alter table public.memories add column if not exists tsv tsvector;
-- create index if not exists idx_memories_tsv on public.memories using gin(tsv);
-- create or replace function memories_tsv_update() returns trigger language plpgsql as $$
-- begin
--   new.tsv := to_tsvector('simple', coalesce(new.title,'') || ' ' || coalesce(new.text,''));
--   return new;
-- end $$;
-- drop trigger if exists trg_memories_tsv on public.memories;
-- create trigger trg_memories_tsv before insert or update on public.memories
-- for each row execute function memories_tsv_update();

-- =====================================================================
-- RLS note (Supabase)
-- =====================================================================
-- For service-role key usage (server-to-DB), you typically keep RLS disabled
-- on these tables during MVP. If RLS is enabled in your project, add permissive
-- policies for the service role, or disable RLS explicitly per table:
--   alter table public.memories disable row level security;
-- (repeat for others if needed)
