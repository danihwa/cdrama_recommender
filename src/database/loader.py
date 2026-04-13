import pandas as pd
from pathlib import Path
from src.database.connection import get_db_connection


def prepare_record(row: dict) -> dict:
    """
    Clean up a single record before sending to Supabase.

    Ensures list columns (embedding, genres, tags) are plain Python lists.
    pgvector needs a standard list, not a numpy array or pandas Series.
    """
    return {
        **row,
        "embedding": list(row["embedding"]),  # ensure plain Python list
        "genres": list(row["genres"]),
        "tags": list(row["tags"]),
    }


def insert_dramas(parquet_path: str | Path, batch_size: int = 100):
    """
    Reads dramas_with_vectors.parquet and upserts into Supabase in batches.

    Uses upsert (not insert) so re-runs are safe — if a drama already
    exists by mdl_id, it gets updated instead of throwing an error.
    Batching prevents timeouts when uploading large 1536-dim vectors.
    """
    supabase = get_db_connection()

    print(f"Loading data from {parquet_path}...")
    df = pd.read_parquet(parquet_path)

    records = [prepare_record(row) for row in df.to_dict(orient="records")]
    total = len(records)
    print(f"Found {total} dramas. Starting upsert...")

    success, failed = 0, 0

    for i in range(0, total, batch_size):
        batch = records[i : i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (total + batch_size - 1) // batch_size

        try:
            supabase.table("cdramas").upsert(
                batch, on_conflict="mdl_id"  # safe to re-run: update if exists
            ).execute()

            success += len(batch)
            print(f"  Batch {batch_num}/{total_batches} ✓ ({success}/{total})")

        except Exception as e:
            failed += len(batch)
            print(f"  Batch {batch_num}/{total_batches} ✗ — {e}")

    print(f"\nDone! {success} upserted, {failed} failed.")


if __name__ == "__main__":
    # to run: uv run src/database/loader.py
    DATA_FILE = Path("data/cleaned/dramas_with_vectors.parquet")
    insert_dramas(DATA_FILE, batch_size=100)