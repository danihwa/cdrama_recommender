"""Load embedded dramas from a parquet file into local Postgres."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.database.connection import get_db_connection


UPSERT_SQL = """
    INSERT INTO cdramas (
        mdl_id, mdl_url, title, native_title, synopsis,
        episodes, year, genres, tags, mdl_score, watchers, embedding
    ) VALUES (
        %(mdl_id)s, %(mdl_url)s, %(title)s, %(native_title)s, %(synopsis)s,
        %(episodes)s, %(year)s, %(genres)s, %(tags)s, %(mdl_score)s, %(watchers)s, %(embedding)s
    )
    ON CONFLICT (mdl_id) DO UPDATE SET
        mdl_url      = EXCLUDED.mdl_url,
        title        = EXCLUDED.title,
        native_title = EXCLUDED.native_title,
        synopsis     = EXCLUDED.synopsis,
        episodes     = EXCLUDED.episodes,
        year         = EXCLUDED.year,
        genres       = EXCLUDED.genres,
        tags         = EXCLUDED.tags,
        mdl_score    = EXCLUDED.mdl_score,
        watchers     = EXCLUDED.watchers,
        embedding    = EXCLUDED.embedding,
        updated_at   = CURRENT_TIMESTAMP
"""


def prepare_record(row: dict) -> dict:
    """Coerce numpy/pandas containers to the plain Python lists psycopg + pgvector expect."""
    return {
        "mdl_id": int(row["mdl_id"]),
        "mdl_url": row["mdl_url"],
        "title": row["title"],
        "native_title": row.get("native_title"),
        "synopsis": row.get("synopsis"),
        "episodes": int(row["episodes"]) if row.get("episodes") is not None else None,
        "year": int(row["year"]) if row.get("year") is not None else None,
        "genres": list(row["genres"]),
        "tags": list(row["tags"]),
        "mdl_score": float(row["mdl_score"]) if row.get("mdl_score") is not None else None,
        "watchers": int(row["watchers"]) if row.get("watchers") is not None else None,
        "embedding": list(row["embedding"]),
    }


def insert_dramas(parquet_path: str | Path, batch_size: int = 100) -> None:
    """
    Read dramas_with_vectors.parquet and upsert into cdramas in batches.

    Uses ON CONFLICT (mdl_id) so re-runs are safe — if a drama already
    exists, the row gets updated instead of throwing. Batching keeps
    memory bounded when uploading thousands of 3072-dim vectors.
    """
    conn = get_db_connection()

    print(f"Loading data from {parquet_path}...")
    df = pd.read_parquet(parquet_path)

    records = [prepare_record(row) for row in df.to_dict(orient="records")]
    total = len(records)
    print(f"Found {total} dramas. Starting upsert...")

    success, failed = 0, 0

    with conn.cursor() as cur:
        for i in range(0, total, batch_size):
            batch = records[i : i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (total + batch_size - 1) // batch_size

            try:
                cur.executemany(UPSERT_SQL, batch)
                success += len(batch)
                print(f"  Batch {batch_num}/{total_batches} OK ({success}/{total})")
            except Exception as e:
                failed += len(batch)
                print(f"  Batch {batch_num}/{total_batches} FAIL — {e}")

    conn.close()
    print(f"\nDone! {success} upserted, {failed} failed.")


if __name__ == "__main__":
    # to run: uv run src/database/loader.py
    DATA_FILE = Path("data/cleaned/dramas_with_vectors.parquet")
    insert_dramas(DATA_FILE, batch_size=100)
