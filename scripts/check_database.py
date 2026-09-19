"""Check connectivity and migration metadata without exposing credentials."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
API_SRC = ROOT / "apps" / "api" / "src"
if str(API_SRC) not in sys.path:
    sys.path.insert(0, str(API_SRC))

from proctoring_api.persistence.database import Database  # noqa: E402
from proctoring_api.settings import Settings  # noqa: E402


async def _check(database_url: str) -> int:
    settings = Settings(database_url=database_url)
    database = Database(settings)
    status = await database.health()
    await database.dispose()
    if status != "ready":
        print("Database unavailable", file=sys.stderr)
        return 1
    print("Database connection ready")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database-url",
        default=os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./storage/sessions/proctoring.sqlite3"),
    )
    return asyncio.run(_check(parser.parse_args().database_url))


if __name__ == "__main__":
    raise SystemExit(main())

