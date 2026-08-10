"""Print compact progress for the risk-model build."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATABASE = (
    PROJECT_ROOT / "data" / "processed" / "csmar_risk_model_v1.sqlite"
)


def safe_query(conn, query, params=()):
    try:
        return conn.execute(query, params).fetchone()
    except sqlite3.OperationalError:
        return None


def run(args):
    conn = sqlite3.connect(args.database, timeout=10)
    try:
        imports = safe_query(
            conn,
            """
            SELECT COUNT(*),
                   SUM(status='complete'),
                   SUM(status='running'),
                   SUM(status='error')
            FROM source_imports
            """,
        )
        raw = safe_query(
            conn,
            """
            SELECT COUNT(*),
                   SUM(status='complete'),
                   SUM(status='error'),
                   MAX(updated_at)
            FROM risk_model_build_state
            WHERE stage='raw'
            """,
        )
        model = safe_query(
            conn,
            """
            SELECT COUNT(*),
                   SUM(status='complete'),
                   SUM(status='error'),
                   MAX(item),
                   MAX(updated_at)
            FROM risk_model_build_state
            WHERE stage='model'
            """,
        )
        cap = safe_query(
            conn,
            """
            SELECT COUNT(*), COUNT(DISTINCT code),
                   MIN(trade_date), MAX(trade_date)
            FROM stock_market_cap
            """,
        )
        exposure = safe_query(
            conn,
            """
            SELECT COUNT(*), COUNT(DISTINCT model_date),
                   MIN(model_date), MAX(model_date)
            FROM weekly_exposure
            """,
        )
        alpha = safe_query(
            conn,
            """
            SELECT COUNT(*), COUNT(DISTINCT model_date),
                   MIN(model_date), MAX(model_date)
            FROM weekly_v31_alpha
            """,
        )
        payload = {
            "database": str(args.database),
            "source_imports": (
                {
                    "total": imports[0],
                    "complete": imports[1] or 0,
                    "running": imports[2] or 0,
                    "errors": imports[3] or 0,
                }
                if imports
                else None
            ),
            "market_cap": (
                {
                    "rows": cap[0],
                    "stocks": cap[1],
                    "min_date": cap[2],
                    "max_date": cap[3],
                }
                if cap
                else None
            ),
            "raw_feature_progress": (
                {
                    "items": raw[0],
                    "complete": raw[1] or 0,
                    "errors": raw[2] or 0,
                    "last_update": raw[3],
                }
                if raw
                else None
            ),
            "model_progress": (
                {
                    "items": model[0],
                    "complete": model[1] or 0,
                    "errors": model[2] or 0,
                    "last_model_date": model[3],
                    "last_update": model[4],
                }
                if model
                else None
            ),
            "weekly_exposure": (
                {
                    "rows": exposure[0],
                    "dates": exposure[1],
                    "min_date": exposure[2],
                    "max_date": exposure[3],
                }
                if exposure
                else None
            ),
            "v31_alpha": (
                {
                    "rows": alpha[0],
                    "dates": alpha[1],
                    "min_date": alpha[2],
                    "max_date": alpha[3],
                }
                if alpha
                else None
            ),
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    finally:
        conn.close()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Show risk-model build progress.")
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    args = parser.parse_args(argv)
    args.database = args.database.resolve()
    if not args.database.exists():
        parser.error(f"Database does not exist: {args.database}")
    return args


if __name__ == "__main__":
    run(parse_args())
