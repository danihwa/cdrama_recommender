-- Provides semantic similarity search with optional filters

create or replace function match_documents (
  query_embedding vector(3072),
  match_threshold float,
  match_count int,
  filter_year int default null,
  filter_score float default null,
  exclude_ids int[] default null,
  filter_genres text[] default null,
  exclude_genres text[] default null
)
returns table (
  id int,
  title text,
  native_title text,
  year int,
  synopsis text,
  mdl_score numeric,
  genres text[],
  tags text[],
  watchers int,
  mdl_url text,
  similarity float
)
language sql stable
SET search_path = public
as $$
  select * from (
    select
      id,
      title,
      native_title,
      year,
      synopsis,
      mdl_score,
      genres,
      tags,
      watchers,
      mdl_url,
      1 - (cdramas.embedding <=> query_embedding) as similarity
    from cdramas
    where (exclude_ids is null or not (cdramas.id = any(exclude_ids)))
      and (filter_year is null or cdramas.year >= filter_year)
      and (filter_score is null or cdramas.mdl_score >= filter_score)
      and (filter_genres is null or cdramas.genres && filter_genres)
      and (exclude_genres is null or not (cdramas.genres && exclude_genres))
  ) sub
  where similarity > match_threshold
  order by similarity desc
  limit match_count;
$$;
