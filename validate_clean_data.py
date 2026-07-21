import argparse
import json
import sqlite3
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Validate a cleaned stock_daily database.")
    parser.add_argument("database_path", nargs="?", type=Path)
    parser.add_argument("--database", dest="database_option", type=Path)
    args = parser.parse_args()
    db_path = (
        args.database_option
        or args.database_path
        or Path("data/processed/resset_stock_daily.sqlite")
    )
    conn = sqlite3.connect(db_path)
    checks = {
        "rows": conn.execute("SELECT COUNT(*) FROM stock_daily").fetchone()[0],
        "stocks": conn.execute("SELECT COUNT(DISTINCT code) FROM stock_daily").fetchone()[0],
        "date_range": conn.execute("SELECT MIN(trade_date), MAX(trade_date) FROM stock_daily").fetchone(),
        "non_cny_rows": conn.execute(
            "SELECT COUNT(*) FROM stock_daily WHERE currency != 'CNY' OR currency IS NULL"
        ).fetchone()[0],
        "non_norm_state_rows": conn.execute(
            "SELECT COUNT(*) FROM stock_daily WHERE listed_state != 'Norm' OR listed_state IS NULL"
        ).fetchone()[0],
        "st_name_prefix_rows": conn.execute(
            """
            SELECT COUNT(*)
            FROM stock_daily
            WHERE UPPER(name) LIKE '*ST%'
               OR UPPER(name) LIKE 'ST%'
               OR UPPER(name) LIKE 'SST%'
            """
        ).fetchone()[0],
        "duplicate_code_date_rows": conn.execute(
            """
            SELECT COUNT(*)
            FROM (
                SELECT code, trade_date, COUNT(*) AS n
                FROM stock_daily
                GROUP BY code, trade_date
                HAVING n > 1
            )
            """
        ).fetchone()[0],
        "missing_open_rows": conn.execute(
            "SELECT COUNT(*) FROM stock_daily WHERE open IS NULL"
        ).fetchone()[0],
        "missing_capital_return_rows": conn.execute(
            "SELECT COUNT(*) FROM stock_daily WHERE capital_return IS NULL"
        ).fetchone()[0],
        "missing_amount_rows": conn.execute(
            "SELECT COUNT(*) FROM stock_daily WHERE amount IS NULL"
        ).fetchone()[0],
        "missing_turnover_rows": conn.execute(
            "SELECT COUNT(*) FROM stock_daily WHERE turnover_total IS NULL"
        ).fetchone()[0],
        "missing_industry_rows": conn.execute(
            "SELECT COUNT(*) FROM stock_daily "
            "WHERE industry_1 IS NULL OR TRIM(industry_1) = '' OR UPPER(industry_1) = 'UNKNOWN'"
        ).fetchone()[0],
        "latest_date_rows": conn.execute(
            "SELECT COUNT(*) FROM stock_daily WHERE trade_date = (SELECT MAX(trade_date) FROM stock_daily)"
        ).fetchone()[0],
        "sample": conn.execute(
            """
            SELECT code, name, trade_date, close, daily_return
            FROM stock_daily
            ORDER BY trade_date, code
            LIMIT 5
            """
        ).fetchall(),
    }
    conn.close()
    print(json.dumps(checks, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
