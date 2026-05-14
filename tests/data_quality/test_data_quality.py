"""Thin runner for the SQL-based data-quality checks.

Each ``.sql`` file under ``checks/`` is a single query that should
return zero rows when the data is healthy. Anything it returns is a
violation. This pattern (a "negative SELECT") is the standard idiom
for SQL-only data quality testing — dbt, Soda, and Great Expectations
all build on it. The same files can also be run interactively in psql
with ``\\i tests/data_quality/checks/<category>/<check>.sql``.

Conventions for check files:
  - First non-blank line should be ``-- description: <one-line>``.
    The runner pulls it into the failure message.
  - Use ``LIMIT 100`` to keep failure output manageable on big breakages.
  - Identify the offending row(s) by ``mdl_id`` so a debugger can find them.

Run:
    uv run pytest tests/data_quality -v
"""

from __future__ import annotations

from pathlib import Path

import psycopg
import pytest

CHECKS_DIR = Path(__file__).parent / "checks"


def _discover_checks() -> list[Path]:
    return sorted(CHECKS_DIR.rglob("*.sql"))


def _check_id(path: Path) -> str:
    """Stable, human-readable test id like 'schema/mdl_score_in_range'."""
    return path.relative_to(CHECKS_DIR).with_suffix("").as_posix()


def _description(sql_text: str) -> str:
    """Pull the ``-- description:`` line out of the SQL, if present."""
    for line in sql_text.splitlines():
        line = line.strip()
        if line.lower().startswith("-- description:"):
            return line.split(":", 1)[1].strip()
    return ""


CHECK_PATHS = _discover_checks()


@pytest.mark.db
@pytest.mark.parametrize("check_path", CHECK_PATHS, ids=[_check_id(p) for p in CHECK_PATHS])
def test_data_quality_check(db_conn: psycopg.Connection, check_path: Path) -> None:
    """A check passes when its query returns no rows."""
    sql_text = check_path.read_text(encoding="utf-8")
    description = _description(sql_text) or "(no description)"

    with db_conn.cursor() as cur:
        cur.execute(sql_text)
        rows = cur.fetchall()

    assert not rows, (
        f"\n  check:       {_check_id(check_path)}\n"
        f"  description: {description}\n"
        f"  violations:  {len(rows)} (showing up to 3) → {rows[:3]}"
    )
