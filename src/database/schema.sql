CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS cdramas (
    id              SERIAL PRIMARY KEY,
    mdl_id          INTEGER UNIQUE NOT NULL,
    mdl_url         TEXT NOT NULL,
    title           TEXT NOT NULL,
    native_title    TEXT,
    synopsis        TEXT,
    episodes        INTEGER,
    year            INTEGER,
    genres          TEXT[],
    tags            TEXT[],
    mdl_score       NUMERIC(3,1),
    watchers        INTEGER,
    embedding       vector(1536),
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_cdramas_mdl_id ON cdramas(mdl_id);
CREATE INDEX IF NOT EXISTS idx_cdramas_score ON cdramas(mdl_score DESC);
CREATE INDEX IF NOT EXISTS idx_cdramas_tags ON cdramas USING GIN(tags);
CREATE INDEX IF NOT EXISTS idx_cdramas_genres ON cdramas USING GIN(genres);
CREATE INDEX IF NOT EXISTS idx_cdramas_embedding ON cdramas USING hnsw (embedding vector_cosine_ops);