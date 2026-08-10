"""Build causal residual-momentum snapshots from weekly specific returns."""

from __future__ import annotations

import argparse
from bisect import bisect_left
from datetime import datetime
import json
import math
from pathlib import Path
import sqlite3


def create_schema(conn):
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS weekly_residual_momentum (
            model_date TEXT NOT NULL,
            code TEXT NOT NULL,
            lookback_weeks INTEGER NOT NULL,
            skip_weeks INTEGER NOT NULL,
            residual_momentum REAL NOT NULL,
            observations INTEGER NOT NULL,
            source_start_date TEXT NOT NULL,
            source_end_date TEXT NOT NULL,
            PRIMARY KEY (model_date, code, lookback_weeks, skip_weeks)
        ) WITHOUT ROWID;

        CREATE INDEX IF NOT EXISTS idx_residual_momentum_code
        ON weekly_residual_momentum(code, model_date);

        CREATE TABLE IF NOT EXISTS residual_momentum_cache_metadata (
            cache_key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        ) WITHOUT ROWID;
        """
    )


def source_signature(conn):
    row = conn.execute(
        """
        SELECT COUNT(*), MIN(model_date), MAX(model_date)
        FROM weekly_specific_return
        """
    ).fetchone()
    return {
        "source_rows": int(row[0] or 0),
        "source_min_date": row[1],
        "source_max_date": row[2],
    }


def cache_key(lookback_weeks, skip_weeks, minimum_observations):
    return f"v1:{int(lookback_weeks)}:{int(skip_weeks)}:{int(minimum_observations)}"


def existing_metadata(conn, key):
    row = conn.execute(
        "SELECT value FROM residual_momentum_cache_metadata WHERE cache_key=?",
        (key,),
    ).fetchone()
    return json.loads(row[0]) if row else None


def iter_code_histories(conn):
    cursor = conn.execute(
        """
        SELECT code, model_date, specific_return
        FROM weekly_specific_return
        ORDER BY code, model_date
        """
    )
    current_code = None
    dates = []
    returns = []
    for code, model_date, specific_return in cursor:
        code = str(code).zfill(6)
        if current_code is not None and code != current_code:
            yield current_code, dates, returns
            dates = []
            returns = []
        current_code = code
        dates.append(str(model_date))
        returns.append(float(specific_return))
    if current_code is not None:
        yield current_code, dates, returns


def build(args):
    database = Path(args.database).resolve()
    if not database.exists():
        raise FileNotFoundError(database)
    conn = sqlite3.connect(database, timeout=120.0)
    source_conn = sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA temp_store=MEMORY")
        create_schema(conn)
        signature = source_signature(conn)
        if signature["source_rows"] <= 0:
            raise ValueError("weekly_specific_return contains no rows")
        key = cache_key(
            args.lookback_weeks,
            args.skip_weeks,
            args.minimum_observations,
        )
        current = existing_metadata(conn, key)
        existing_rows = int(
            conn.execute(
                """
                SELECT COUNT(*) FROM weekly_residual_momentum
                WHERE lookback_weeks=? AND skip_weeks=?
                """,
                (args.lookback_weeks, args.skip_weeks),
            ).fetchone()[0]
        )
        if (
            not args.overwrite
            and current is not None
            and all(current.get(name) == value for name, value in signature.items())
            and existing_rows == int(current.get("output_rows", -1))
        ):
            print(
                f"Residual-momentum cache is current: {existing_rows:,} rows "
                f"through {signature['source_max_date']}",
                flush=True,
            )
            return

        global_dates = [
            str(row[0])
            for row in conn.execute(
                "SELECT DISTINCT model_date FROM weekly_specific_return ORDER BY model_date"
            )
        ]
        date_index = {date: index for index, date in enumerate(global_dates)}
        conn.execute(
            """
            DELETE FROM weekly_residual_momentum
            WHERE lookback_weeks=? AND skip_weeks=?
            """,
            (args.lookback_weeks, args.skip_weeks),
        )
        conn.commit()

        insert_sql = """
            INSERT INTO weekly_residual_momentum
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        pending = []
        output_rows = 0
        code_count = 0
        for code, dates, specific_returns in iter_code_histories(source_conn):
            positions = [date_index[date] for date in dates]
            logs = [math.log1p(min(max(value, -0.95), 10.0)) for value in specific_returns]
            cumulative = [0.0]
            for value in logs:
                cumulative.append(cumulative[-1] + value)
            for current_date, current_position in zip(dates, positions):
                first_position = current_position - int(args.lookback_weeks)
                last_position_exclusive = current_position - int(args.skip_weeks)
                if last_position_exclusive <= first_position:
                    continue
                left = bisect_left(positions, first_position)
                right = bisect_left(positions, last_position_exclusive)
                observations = right - left
                if observations < int(args.minimum_observations):
                    continue
                momentum = math.expm1(cumulative[right] - cumulative[left])
                pending.append(
                    (
                        current_date,
                        code,
                        int(args.lookback_weeks),
                        int(args.skip_weeks),
                        float(momentum),
                        int(observations),
                        dates[left],
                        dates[right - 1],
                    )
                )
                if len(pending) >= int(args.batch_size):
                    conn.executemany(insert_sql, pending)
                    output_rows += len(pending)
                    pending.clear()
                    conn.commit()
            code_count += 1
            if code_count % 250 == 0:
                print(
                    f"Residual-momentum progress: {code_count:,} stocks, "
                    f"{output_rows + len(pending):,} rows",
                    flush=True,
                )
        if pending:
            conn.executemany(insert_sql, pending)
            output_rows += len(pending)
            conn.commit()

        metadata = {
            **signature,
            "output_rows": int(output_rows),
            "lookback_weeks": int(args.lookback_weeks),
            "skip_weeks": int(args.skip_weeks),
            "minimum_observations": int(args.minimum_observations),
            "built_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        }
        conn.execute(
            """
            INSERT OR REPLACE INTO residual_momentum_cache_metadata(cache_key, value)
            VALUES (?, ?)
            """,
            (key, json.dumps(metadata, ensure_ascii=False, sort_keys=True)),
        )
        conn.commit()
        print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)
    finally:
        source_conn.close()
        conn.close()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--lookback-weeks", type=int, default=52)
    parser.add_argument("--skip-weeks", type=int, default=4)
    parser.add_argument("--minimum-observations", type=int, default=26)
    parser.add_argument("--batch-size", type=int, default=50000)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
