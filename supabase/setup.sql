-- Run this once in your Supabase project's SQL editor
-- (supabase.com → your project → SQL Editor → New query)

-- 1. Enable the pgvector extension
create extension if not exists vector;

-- 2. Create the chunks table
--    embedding dimension is 1536 for OpenAI text-embedding-3-small
--    change to 1024 if you switch to Anthropic voyage-3
create table if not exists training_chunks (
  id          bigserial primary key,
  source      text not null,        -- original filename
  chunk_index int  not null,        -- position within that file
  content     text not null,        -- the actual passage text
  embedding   vector(1536)          -- change to 1024 for voyage-3
);

-- 3. Index for fast cosine-similarity search
--    'lists' should be roughly sqrt(number of rows); 50 is fine for < 2500 chunks
create index if not exists training_chunks_embedding_idx
  on training_chunks
  using ivfflat (embedding vector_cosine_ops)
  with (lists = 50);

-- 4. Helper function the chatbot will call to find relevant chunks
--    Returns the top k chunks most similar to a query embedding
create or replace function match_chunks(
  query_embedding vector(1536),   -- change to 1024 if using voyage-3
  match_count     int default 5
)
returns table (
  id          bigint,
  source      text,
  chunk_index int,
  content     text,
  similarity  float
)
language sql stable
as $$
  select
    id,
    source,
    chunk_index,
    content,
    1 - (embedding <=> query_embedding) as similarity
  from training_chunks
  order by embedding <=> query_embedding
  limit match_count;

