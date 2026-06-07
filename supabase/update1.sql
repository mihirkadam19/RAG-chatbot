create or replace function match_chunks(
  query_embedding vector(1536),
  query_text text,
  match_count int default 5
)
returns table (
  id bigint,
  source text,
  chunk_index int,
  content text,
  similarity float
)
language sql stable
as $$
  with vector_results as (
    select
      id, source, chunk_index, content,
      1 - (embedding <=> query_embedding) as similarity
    from training_chunks
    order by embedding <=> query_embedding
    limit match_count * 2
  ),
  fts_results as (
    select
      id, source, chunk_index, content,
      ts_rank(to_tsvector('english', content), websearch_to_tsquery('english', query_text)) as similarity
    from training_chunks
    where to_tsvector('english', content) @@ websearch_to_tsquery('english', query_text)
    limit match_count * 2
  ),
  combined as (
    select * from vector_results
    union
    select * from fts_results
  )
  select id, source, chunk_index, content,
    max(similarity) as similarity
  from combined
  group by id, source, chunk_index, content
  order by max(similarity) desc
  limit match_count;
$$;