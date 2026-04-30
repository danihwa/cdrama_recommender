"""Smoke test: can we reach Supabase with the configured creds?

Not a behavioral test — just a reachability check run before the real
suite to fail fast on misconfigured env or network issues.
"""

from postgrest.types import CountMethod

from src.database.connection import get_db_connection
from src.env import load_secrets


def test_supabase_reachable():
    load_secrets()
    supabase = get_db_connection()
    response = (
        supabase.table("cdramas")
        .select("*", count=CountMethod.exact)
        .limit(1)
        .execute()
    )
    assert isinstance(response.count, int)
