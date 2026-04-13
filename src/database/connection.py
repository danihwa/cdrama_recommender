import os
from dotenv import load_dotenv
from supabase import create_client, Client
from postgrest.types import CountMethod
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


def test_connection():
    """Verify Supabase connectivity by querying the cdramas table.

    Prints the current row count on success, or troubleshooting tips
    on failure.
    """
    try:
        supabase = get_db_connection()
        # perform a simple count query to verify the key has access
        response = (
            supabase.table("cdramas")
            .select("*", count=CountMethod.exact)
            .limit(1)
            .execute()
        )

        print("Supabase SDK Connection Successful!")
        print(f"Current rows in 'cdramas' table: {response.count}")

    except Exception as e:
        print(f"Connection failed: {e}")
        print("\nTroubleshooting Tips:")
        print("1. Ensure SUPABASE_URL is 'https://xxx.supabase.co'")
        print(
            "2. Ensure SUPABASE_SECRET_KEY is the 'secret' key (not the publishable key) to bypass RLS."
        )


if __name__ == "__main__":
    test_connection()