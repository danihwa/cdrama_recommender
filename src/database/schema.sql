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
    embedding       vector(3072),
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
