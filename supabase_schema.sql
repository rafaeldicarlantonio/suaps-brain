create extension if not exists pgcrypto;

create table if not exists users (
  id uuid primary key default gen_random_uuid(),
  email text unique,
  role text default 'member',
  created_at timestamptz default now()
);

create table if not exists sessions (
  id uuid primary key default gen_random_uuid(),
  user_id uuid references users(id) on delete cascade,
  channel text default 'gpt_actions',
  started_at timestamptz default now(),
  ended_at timestamptz
);

create table if not exists messages (
  id uuid primary key default gen_random_uuid(),
  session_id uuid references sessions(id) on delete cascade,
  user_id uuid references users(id) on delete cascade,
  role text check (role in ('user','assistant','system')),
  content text,
  tokens int,
  created_at timestamptz default now()
);

do $$ begin
  if not exists (select 1 from pg_type where typname = 'memory_type') then
    create type memory_type as enum ('episodic','semantic','procedural');
  end if;
end $$;

create table if not exists memories (
  id uuid primary key default gen_random_uuid(),
  user_id uuid references users(id) on delete set null,
  type memory_type not null,
  title text,
  content text not null,
  importance int default 3,
  tags text[],
  created_at timestamptz default now(),
  updated_at timestamptz default now(),
  deleted boolean default false
);
