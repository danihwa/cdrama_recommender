import os
from supabase import create_client, Client
from src.env import load_secrets


load_secrets()


def get_db_connection() -> Client:
    """Return an authenticated Supabase client.

    Reads SUPABASE_URL and SUPABASE_SECRET_KEY from the environment
    (loaded via dotenv). Uses the secret/service-role key to bypass
    Row-Level Security.

    Raises:
        ValueError: If either environment variable is missing.
    """
    url: str | None = os.environ.get("SUPABASE_URL")

    key: str | None = os.environ.get("SUPABASE_SECRET_KEY")

    if not url or not key:
        raise ValueError(
            "SUPABASE_URL or SUPABASE_SECRET_KEY is missing from .env file!"
        )

    try:
        return create_client(url, key)
    except Exception as e:
        raise ConnectionError(
            f"Failed to connect to Supabase: {e}"
        ) from e
