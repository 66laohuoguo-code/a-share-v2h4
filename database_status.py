"""Print the date range and size of a stock_daily SQLite database."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path


def read_status(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Database does not exist: {path}")
    conn = sqlite3.connect(path)
    try:
        row = conn.execute(
            "SELECT MIN(trade_date), MAX(trade_date), COUNT(*), COUNT(DISTINCT code) FROM stock_daily"
        ).fetchone()
    finally:
        conn.close()
    return {
        "database": str(path),
        "min_date": row[0],
        "max_date": row[1],
        "rows": int(row[2]),
        "stocks": int(row[3]),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Inspect a stock_daily SQLite database.")
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--field", choices=["database", "min_date", "max_date", "rows", "stocks"])
    args = parser.parse_args(argv)
    status = read_status(args.database)
    if args.field:
        print(status[args.field])
    else:
        print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
