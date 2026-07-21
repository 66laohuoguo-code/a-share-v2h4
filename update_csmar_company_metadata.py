"""Incrementally update CSMAR company and industry metadata in an existing database."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3

from build_csmar_database import load_company_file


COMPANY_EXTRA_COLUMNS = {
    "company_status": "TEXT",
    "former_code": "TEXT",
    "established_date": "TEXT",
    "issue_date": "TEXT",
    "industry_code_c": "TEXT",
    "industry_name_c": "TEXT",
    "industry_code_d": "TEXT",
    "industry_name_d": "TEXT",
}

COMPANY_COLUMNS = (
    "code",
    "name",
    "company_name",
    "list_date",
    "currency",
    "market_type",
    "ab_cross_code",
    "h_cross_code",
    "status_date",
    "company_status",
    "former_code",
    "established_date",
    "issue_date",
    "industry_code_c",
    "industry_name_c",
    "industry_code_d",
    "industry_name_d",
    "source_file",
    "source_sheet",
)


def now_iso():
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def table_exists(conn, table):
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone() is not None


def database_stats(conn):
    row = conn.execute(
        """
        SELECT COUNT(*), COUNT(DISTINCT code),
               SUM(industry_1 IS NULL OR industry_1 = 'UNKNOWN'),
               COUNT(DISTINCT CASE
                   WHEN industry_1 IS NULL OR industry_1 = 'UNKNOWN' THEN code
               END)
        FROM stock_daily
        """
    ).fetchone()
    return dict(
        zip(("rows", "stocks", "unknown_industry_rows", "unknown_industry_stocks"), row)
    )


def ensure_company_schema(conn):
    columns = {row[1] for row in conn.execute("PRAGMA table_info(csmar_company)")}
    for name, column_type in COMPANY_EXTRA_COLUMNS.items():
        if name not in columns:
            conn.execute(f"ALTER TABLE csmar_company ADD COLUMN {name} {column_type}")


def create_mapping_table(conn, companies):
    conn.execute("DROP TABLE IF EXISTS temp.company_industry_update")
    conn.execute(
        """
        CREATE TEMP TABLE company_industry_update (
            code TEXT PRIMARY KEY,
            industry_1 TEXT NOT NULL,
            industry_2 TEXT NOT NULL
        ) WITHOUT ROWID
        """
    )
    conn.executemany(
        "INSERT INTO company_industry_update VALUES (?,?,?)",
        [
            (company["code"], company["industry_1"], company["industry_2"])
            for company in companies.values()
            if company.get("industry_1") and company.get("industry_2")
        ],
    )


def coverage_report(conn):
    matched_stocks = conn.execute(
        """
        SELECT COUNT(*)
        FROM stock_meta m
        JOIN company_industry_update c ON c.code = m.code
        """
    ).fetchone()[0]
    missing_codes = [
        row[0]
        for row in conn.execute(
            """
            SELECT m.code
            FROM stock_meta m
            LEFT JOIN company_industry_update c ON c.code = m.code
            WHERE c.code IS NULL
            ORDER BY m.code
            LIMIT 100
            """
        )
    ]
    mismatch_rows = conn.execute(
        """
        SELECT m.code, m.name, m.industry_1, m.industry_2,
               c.industry_1, c.industry_2
        FROM stock_meta m
        JOIN company_industry_update c ON c.code = m.code
        WHERE m.industry_1 IS NOT NULL AND m.industry_1 <> 'UNKNOWN'
          AND (m.industry_1 <> c.industry_1 OR m.industry_2 <> c.industry_2)
        ORDER BY m.code
        """
    ).fetchall()
    return {
        "matched_database_stocks": matched_stocks,
        "unmatched_database_stocks": conn.execute(
            """
            SELECT COUNT(*)
            FROM stock_meta m
            LEFT JOIN company_industry_update c ON c.code = m.code
            WHERE c.code IS NULL
            """
        ).fetchone()[0],
        "unmatched_code_sample": missing_codes,
        "known_industry_mismatches": len(mismatch_rows),
        "known_industry_mismatch_rows": [
            {
                "code": row[0],
                "name": row[1],
                "database_industry_1": row[2],
                "database_industry_2": row[3],
                "company_industry_1": row[4],
                "company_industry_2": row[5],
            }
            for row in mismatch_rows[:100]
        ],
    }


def upsert_companies(conn, companies):
    placeholders = ",".join("?" for _ in COMPANY_COLUMNS)
    assignments = []
    for column in COMPANY_COLUMNS[1:]:
        if column in {"name", "market_type", "source_file", "source_sheet"}:
            assignments.append(f"{column}=excluded.{column}")
        else:
            assignments.append(
                f"{column}=COALESCE(excluded.{column},csmar_company.{column})"
            )
    conn.executemany(
        f"""
        INSERT INTO csmar_company ({','.join(COMPANY_COLUMNS)})
        VALUES ({placeholders})
        ON CONFLICT(code) DO UPDATE SET {','.join(assignments)}
        """,
        [
            tuple(company.get(column) for column in COMPANY_COLUMNS)
            for company in companies.values()
        ],
    )


def update_industries(conn, table, mode):
    if mode == "missing":
        condition = "industry_1 IS NULL OR industry_1 = 'UNKNOWN' OR industry_2 IS NULL OR industry_2 = 'UNKNOWN'"
    else:
        condition = "1=1"
    cursor = conn.execute(
        f"""
        UPDATE {table}
        SET industry_1 = (
                SELECT c.industry_1 FROM company_industry_update c WHERE c.code = {table}.code
            ),
            industry_2 = (
                SELECT c.industry_2 FROM company_industry_update c WHERE c.code = {table}.code
            )
        WHERE ({condition})
          AND EXISTS (
              SELECT 1 FROM company_industry_update c WHERE c.code = {table}.code
          )
        """
    )
    return cursor.rowcount


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--company-file", type=Path, required=True)
    parser.add_argument(
        "--industry-mode",
        choices=("missing", "all"),
        default="missing",
        help="Fill only missing industries (default) or replace every industry mapping.",
    )
    parser.add_argument("--report", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def run(args):
    database = args.database.resolve()
    company_file = args.company_file.resolve()
    report_path = (
        args.report.resolve()
        if args.report
        else database.with_name(database.stem + "_company_metadata_update_report.json")
    )
    if not database.exists():
        raise FileNotFoundError(f"Database does not exist: {database}")
    if not company_file.exists():
        raise FileNotFoundError(f"Company workbook does not exist: {company_file}")

    companies = load_company_file(company_file, {})
    industry_companies = {
        code: company
        for code, company in companies.items()
        if company.get("industry_1") and company.get("industry_2")
    }
    if not companies or not industry_companies:
        raise ValueError("No valid A-share company or 2012 CSRC industry rows were found.")

    with sqlite3.connect(database, timeout=60) as conn:
        conn.execute("PRAGMA busy_timeout=60000")
        for table in ("stock_daily", "stock_meta", "csmar_company", "project_metadata"):
            if not table_exists(conn, table):
                raise ValueError(f"Required database table is missing: {table}")
        before = database_stats(conn)
        create_mapping_table(conn, companies)
        coverage = coverage_report(conn)
        company_rows_before = conn.execute("SELECT COUNT(*) FROM csmar_company").fetchone()[0]

        updated_daily_rows = 0
        updated_meta_rows = 0
        if not args.dry_run:
            ensure_company_schema(conn)
            upsert_companies(conn, companies)
            updated_daily_rows = update_industries(conn, "stock_daily", args.industry_mode)
            updated_meta_rows = update_industries(conn, "stock_meta", args.industry_mode)
            completed_at = now_iso()
            metadata = {
                "industry_source": str(company_file),
                "industry_source_sha256": sha256_file(company_file),
                "industry_update_mode": args.industry_mode,
                "industry_updated_at": completed_at,
            }
            conn.executemany(
                "INSERT OR REPLACE INTO project_metadata(key,value) VALUES (?,?)",
                metadata.items(),
            )
            conn.commit()
        after = database_stats(conn)
        company_rows_after = conn.execute("SELECT COUNT(*) FROM csmar_company").fetchone()[0]

    report = {
        "database": str(database),
        "company_file": str(company_file),
        "company_file_sha256": sha256_file(company_file),
        "dry_run": bool(args.dry_run),
        "industry_mode": args.industry_mode,
        "source_a_share_companies": len(companies),
        "source_industry_mappings": len(industry_companies),
        "coverage": coverage,
        "before": before,
        "after": after,
        "company_rows_before": company_rows_before,
        "company_rows_after": company_rows_after,
        "updated_daily_rows": updated_daily_rows,
        "updated_stock_meta_rows": updated_meta_rows,
        "completed_at": now_iso(),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Report: {report_path}")
    return report


if __name__ == "__main__":
    run(parse_args())
