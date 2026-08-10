"""Build a point-in-time sidecar database for the A-share risk model."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import time
import zipfile

from import_csmar_forward_quotation import (
    iter_sheet_rows,
    load_shared_strings,
    parse_code,
    parse_date,
    parse_float,
    workbook_sheets,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_RAW_ROOT = PROJECT_ROOT / "data" / "raw" / "CSMAR raw data"
DEFAULT_RISK_RAW_ROOT = DEFAULT_RAW_ROOT / "risk model"
DEFAULT_FORWARD_MARKET_ROOT = PROJECT_ROOT / "data" / "raw" / "market_data"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "csmar_risk_model_v1.sqlite"

FINANCIAL_SPECS = {
    "balance_statement": {
        "filename": ("FS_Combas.xlsx", "资产负债表.xlsx"),
        "columns": {
            "A001000000": "total_assets",
            "A002000000": "total_liabilities",
            "A003100000": "parent_equity",
            "A003000000": "total_equity",
            "A001101000": "cash",
            "A001100000": "current_assets",
            "A002100000": "current_liabilities",
        },
    },
    "income_statement": {
        "filename": ("FS_Comins.xlsx", "利润表.xlsx"),
        "columns": {
            "B001100000": "total_operating_revenue_ytd",
            "B001101000": "operating_revenue_ytd",
            "B001300000": "operating_profit_ytd",
            "B001000000": "total_profit_ytd",
            "B002000000": "net_profit_ytd",
            "B002000101": "parent_net_profit_ytd",
            "B001216000": "research_development_ytd",
        },
    },
    "cashflow_statement": {
        "filename": ("FS_Comscfd.xlsx", "现金流量表.xlsx"),
        "columns": {
            "C001000000": "operating_cashflow_ytd",
            "C002006000": "capital_expenditure_ytd",
            "C003006000": "dividend_interest_paid_ytd",
        },
    },
}

REFERENCE_SPECS = {
    "industry_history": {
        "filename": ("STK_INDUSTRYCLASS.xlsx", "上市公司行业分类.xlsx"),
        "columns": (
            "Symbol",
            "IndustryClassificationID",
            "IndustryClassificationName",
            "ImplementDate",
            "IndustryCode",
            "IndustryName",
            "InstitutionID",
        ),
    },
    "risk_free_daily": {
        "filename": ("TRD_Nrrate.xlsx", "无风险利率.xlsx"),
        "columns": ("Nrr1", "Clsdt", "Nrrdata", "Nrrdaydt", "Nrrwkdt", "Nrrmtdt"),
    },
    "market_return_daily": {
        "filename": ("TRD_Cndalym.xlsx", "日市场回报率.xlsx"),
        "columns": (
            "Markettype",
            "Trddt",
            "Cnshrtrdtl",
            "Cnvaltrdtl",
            "Cdretwdeq",
            "Cdretwdos",
            "Cdretwdtl",
            "Cdnstkcal",
        ),
        "column_aliases": {
            "Cnshrtrdtl": ("Dnshrtrdtl",),
            "Cnvaltrdtl": ("Dnvaltrdtl",),
            "Cdretwdeq": ("Dretwdeq",),
            "Cdretwdos": ("Dretwdos",),
            "Cdretwdtl": ("Dretwdtl",),
            "Cdnstkcal": ("Dnstkcal",),
        },
    },
    "index_daily": {
        "filename": ("TRD_Index.xlsx", "指数.xlsx"),
        "columns": (
            "Indexcd",
            "Trddt",
            "Opnindex",
            "Hiindex",
            "Loindex",
            "Clsindex",
            "Retindex",
        ),
    },
}

MARKET_CAP_HEADERS = ("Stkcd", "Trddt", "Dsmvosd", "Dsmvtll")
FORWARD_MARKET_CAP_HEADERS = (
    "TradingDate",
    "Symbol",
    "StateCode",
    "MarketValue",
)
BASE_FINANCIAL_HEADERS = (
    "Stkcd",
    "ShortName",
    "Accper",
    "Typrep",
    "IfCorrect",
    "DeclareDate",
)


def now_iso():
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def clean_text(value):
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def parse_int(value):
    number = parse_float(value)
    return int(number) if number is not None else None


def correction_flag(value):
    text = str(value or "").strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return 1
    number = parse_float(value)
    return int(bool(number)) if number is not None else 0


def normalize_sheet_path(path):
    text = str(path or "").replace("\\", "/").lstrip("/")
    while text.startswith("xl/xl/"):
        text = text[3:]
    return text


def find_unique_file(root, filename):
    filenames = (filename,) if isinstance(filename, str) else tuple(filename)
    matches = sorted(
        {path for candidate in filenames for path in root.rglob(candidate)},
        key=lambda path: str(path).lower(),
    )
    if not matches:
        raise FileNotFoundError(
            f"Required CSMAR file was not found; accepted names: {filenames}"
        )
    if len(matches) > 1:
        locations = ", ".join(str(path) for path in matches)
        raise ValueError(f"More than one matching source file was found: {locations}")
    return matches[0]


def iter_csmar_rows(path, required_headers, header_aliases=None):
    required = set(required_headers)
    with zipfile.ZipFile(path) as archive:
        shared_strings = load_shared_strings(archive)
        for sheet_name, sheet_path in workbook_sheets(archive):
            sheet_path = normalize_sheet_path(sheet_path)
            rows = iter_sheet_rows(archive, sheet_path, shared_strings)
            try:
                header = next(rows)
            except StopIteration:
                continue
            columns = {
                str(value).strip(): index
                for index, value in enumerate(header)
                if value is not None and str(value).strip()
            }
            for canonical, aliases in (header_aliases or {}).items():
                if canonical in columns:
                    continue
                for alias in aliases:
                    if alias in columns:
                        columns[canonical] = columns[alias]
                        break
            missing = sorted(required - set(columns))
            if missing:
                raise ValueError(
                    f"{path.name}/{sheet_name} is missing required columns: {missing}"
                )
            for _ in range(2):
                next(rows, None)
            for row_number, row in enumerate(rows, start=4):
                yield sheet_name, row_number, row, columns


def row_value(row, columns, name):
    index = columns.get(name)
    if index is None or index >= len(row):
        return None
    return row[index]


def create_schema(conn):
    conn.executescript(
        """
        PRAGMA journal_mode=WAL;
        PRAGMA synchronous=NORMAL;
        PRAGMA temp_store=MEMORY;
        PRAGMA cache_size=-262144;
        PRAGMA foreign_keys=OFF;

        CREATE TABLE IF NOT EXISTS source_imports (
            source_key TEXT PRIMARY KEY,
            stage TEXT NOT NULL,
            source_path TEXT NOT NULL,
            source_size INTEGER,
            source_mtime_ns INTEGER,
            status TEXT NOT NULL,
            rows_imported INTEGER NOT NULL DEFAULT 0,
            started_at TEXT,
            completed_at TEXT,
            message TEXT
        ) WITHOUT ROWID;

        CREATE TABLE IF NOT EXISTS stock_market_cap (
            code TEXT NOT NULL,
            trade_date TEXT NOT NULL,
            float_market_cap REAL,
            total_market_cap REAL,
            source_file TEXT NOT NULL,
            PRIMARY KEY (code, trade_date)
        ) WITHOUT ROWID;

        CREATE TABLE IF NOT EXISTS financial_report_dates (
            code TEXT NOT NULL,
            report_period TEXT NOT NULL,
            actual_disclosure_date TEXT,
            first_scheduled_date TEXT,
            source TEXT,
            retrieved_at TEXT,
            PRIMARY KEY (code, report_period)
        ) WITHOUT ROWID;

        CREATE TABLE IF NOT EXISTS balance_statement (
            code TEXT NOT NULL,
            name TEXT,
            report_period TEXT NOT NULL,
            report_type TEXT NOT NULL,
            is_corrected INTEGER NOT NULL,
            correction_disclosure_date TEXT,
            total_assets REAL,
            total_liabilities REAL,
            parent_equity REAL,
            total_equity REAL,
            cash REAL,
            current_assets REAL,
            current_liabilities REAL,
            source_file TEXT NOT NULL,
            PRIMARY KEY (code, report_period, report_type)
        ) WITHOUT ROWID;

        CREATE TABLE IF NOT EXISTS income_statement (
            code TEXT NOT NULL,
            name TEXT,
            report_period TEXT NOT NULL,
            report_type TEXT NOT NULL,
            is_corrected INTEGER NOT NULL,
            correction_disclosure_date TEXT,
            total_operating_revenue_ytd REAL,
            operating_revenue_ytd REAL,
            operating_profit_ytd REAL,
            total_profit_ytd REAL,
            net_profit_ytd REAL,
            parent_net_profit_ytd REAL,
            research_development_ytd REAL,
            source_file TEXT NOT NULL,
            PRIMARY KEY (code, report_period, report_type)
        ) WITHOUT ROWID;

        CREATE TABLE IF NOT EXISTS cashflow_statement (
            code TEXT NOT NULL,
            name TEXT,
            report_period TEXT NOT NULL,
            report_type TEXT NOT NULL,
            is_corrected INTEGER NOT NULL,
            correction_disclosure_date TEXT,
            operating_cashflow_ytd REAL,
            capital_expenditure_ytd REAL,
            dividend_interest_paid_ytd REAL,
            source_file TEXT NOT NULL,
            PRIMARY KEY (code, report_period, report_type)
        ) WITHOUT ROWID;

        CREATE TABLE IF NOT EXISTS industry_history (
            code TEXT NOT NULL,
            classification_id TEXT NOT NULL,
            classification_name TEXT NOT NULL,
            implement_date TEXT NOT NULL,
            industry_code TEXT,
            industry_name TEXT,
            institution_id TEXT,
            source_file TEXT NOT NULL,
            PRIMARY KEY (
                code, classification_id, implement_date, industry_code
            )
        ) WITHOUT ROWID;

        CREATE TABLE IF NOT EXISTS risk_free_daily (
            benchmark_code TEXT NOT NULL,
            trade_date TEXT NOT NULL,
            annual_rate_pct REAL,
            daily_rate_pct REAL,
            weekly_rate_pct REAL,
            monthly_rate_pct REAL,
            source_file TEXT NOT NULL,
            PRIMARY KEY (benchmark_code, trade_date)
        ) WITHOUT ROWID;

        CREATE TABLE IF NOT EXISTS market_return_daily (
            market_type INTEGER NOT NULL,
            trade_date TEXT NOT NULL,
            volume REAL,
            amount REAL,
            equal_weight_return REAL,
            float_weight_return REAL,
            total_weight_return REAL,
            stock_count INTEGER,
            source_file TEXT NOT NULL,
            PRIMARY KEY (market_type, trade_date)
        ) WITHOUT ROWID;

        CREATE TABLE IF NOT EXISTS index_daily (
            index_code TEXT NOT NULL,
            trade_date TEXT NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            daily_return REAL,
            source_file TEXT NOT NULL,
            PRIMARY KEY (index_code, trade_date)
        ) WITHOUT ROWID;

        CREATE TABLE IF NOT EXISTS financial_pit (
            code TEXT NOT NULL,
            name TEXT,
            report_period TEXT NOT NULL,
            actual_disclosure_date TEXT,
            available_date TEXT,
            total_assets REAL,
            total_liabilities REAL,
            parent_equity REAL,
            total_equity REAL,
            cash REAL,
            current_assets REAL,
            current_liabilities REAL,
            operating_revenue_ytd REAL,
            operating_profit_ytd REAL,
            total_profit_ytd REAL,
            net_profit_ytd REAL,
            parent_net_profit_ytd REAL,
            research_development_ytd REAL,
            operating_cashflow_ytd REAL,
            capital_expenditure_ytd REAL,
            dividend_interest_paid_ytd REAL,
            revenue_ttm REAL,
            net_profit_ttm REAL,
            parent_net_profit_ttm REAL,
            operating_cashflow_ttm REAL,
            capital_expenditure_ttm REAL,
            ttm_available_date TEXT,
            revenue_growth REAL,
            earnings_growth REAL,
            growth_available_date TEXT,
            is_corrected INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (code, report_period)
        ) WITHOUT ROWID;

        CREATE TABLE IF NOT EXISTS risk_data_metadata (
            key TEXT PRIMARY KEY,
            value TEXT
        ) WITHOUT ROWID;

        CREATE INDEX IF NOT EXISTS idx_market_cap_date
        ON stock_market_cap(trade_date);
        CREATE INDEX IF NOT EXISTS idx_financial_pit_available
        ON financial_pit(code, available_date, report_period);
        CREATE INDEX IF NOT EXISTS idx_industry_history_lookup
        ON industry_history(classification_name, code, implement_date);
        CREATE INDEX IF NOT EXISTS idx_market_return_date
        ON market_return_daily(trade_date, market_type);
        CREATE INDEX IF NOT EXISTS idx_risk_free_date
        ON risk_free_daily(trade_date, benchmark_code);
        """
    )


def source_signature(path):
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns


def source_is_complete(conn, source_key, path):
    row = conn.execute(
        """
        SELECT status, source_size, source_mtime_ns
        FROM source_imports WHERE source_key=?
        """,
        (source_key,),
    ).fetchone()
    if row is None:
        return False
    size, mtime_ns = source_signature(path)
    return row == ("complete", size, mtime_ns)


def mark_source(conn, source_key, stage, path, status, rows=0, message=None):
    size, mtime_ns = source_signature(path)
    timestamp = now_iso()
    conn.execute(
        """
        INSERT INTO source_imports (
            source_key, stage, source_path, source_size, source_mtime_ns,
            status, rows_imported, started_at, completed_at, message
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(source_key) DO UPDATE SET
            stage=excluded.stage,
            source_path=excluded.source_path,
            source_size=excluded.source_size,
            source_mtime_ns=excluded.source_mtime_ns,
            status=excluded.status,
            rows_imported=excluded.rows_imported,
            started_at=CASE
                WHEN excluded.status='running' THEN excluded.started_at
                ELSE source_imports.started_at
            END,
            completed_at=excluded.completed_at,
            message=excluded.message
        """,
        (
            source_key,
            stage,
            str(path),
            size,
            mtime_ns,
            status,
            rows,
            timestamp if status == "running" else None,
            timestamp if status == "complete" else None,
            message,
        ),
    )
    conn.commit()


def relative_source(path, raw_root):
    try:
        return str(path.resolve().relative_to(raw_root.resolve()))
    except ValueError:
        return str(path.resolve())


def import_report_dates(conn, risk_raw_root, overwrite=False):
    path = find_unique_file(risk_raw_root, "cninfo_financial_report_dates.sqlite")
    source_key = "reference:cninfo_financial_report_dates"
    if not overwrite and source_is_complete(conn, source_key, path):
        print(f"[skip] {path.name}", flush=True)
        return
    mark_source(conn, source_key, "reference", path, "running")
    conn.execute("DELETE FROM financial_report_dates")
    source = sqlite3.connect(path)
    try:
        rows = source.execute(
            """
            SELECT code, report_period, actual_disclosure_date,
                   first_scheduled_date, source, retrieved_at
            FROM financial_report_dates
            """
        )
        count = 0
        batch = []
        for row in rows:
            batch.append(row)
            if len(batch) >= 10000:
                conn.executemany(
                    """
                    INSERT OR REPLACE INTO financial_report_dates
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    batch,
                )
                count += len(batch)
                batch.clear()
        if batch:
            conn.executemany(
                "INSERT OR REPLACE INTO financial_report_dates VALUES (?, ?, ?, ?, ?, ?)",
                batch,
            )
            count += len(batch)
        conn.commit()
    finally:
        source.close()
    mark_source(conn, source_key, "reference", path, "complete", count)
    print(f"[done] report dates: {count:,}", flush=True)


def import_financial_table(
    conn,
    raw_root,
    risk_raw_root,
    table,
    spec,
    start_date,
    end_date,
    batch_size,
    overwrite=False,
):
    path = find_unique_file(risk_raw_root, spec["filename"])
    source_file = relative_source(path, raw_root)
    source_key = f"financial:{table}:{source_file}"
    if not overwrite and source_is_complete(conn, source_key, path):
        print(f"[skip] {path.name}", flush=True)
        return

    mark_source(conn, source_key, "financial", path, "running")
    conn.execute(f"DELETE FROM {table} WHERE source_file=?", (source_file,))
    conn.commit()

    value_columns = list(spec["columns"].values())
    insert_columns = [
        "code",
        "name",
        "report_period",
        "report_type",
        "is_corrected",
        "correction_disclosure_date",
        *value_columns,
        "source_file",
    ]
    placeholders = ",".join("?" for _ in insert_columns)
    statement = (
        f"INSERT OR REPLACE INTO {table} "
        f"({','.join(insert_columns)}) VALUES ({placeholders})"
    )
    required = set(BASE_FINANCIAL_HEADERS) | set(spec["columns"])
    batch = []
    count = 0
    started = time.monotonic()

    for _, _, row, columns in iter_csmar_rows(
        path, required, spec.get("column_aliases")
    ):
        code = parse_code(row_value(row, columns, "Stkcd"))
        report_period = parse_date(row_value(row, columns, "Accper"))
        report_type = str(row_value(row, columns, "Typrep") or "").strip().upper()
        if not code or not report_period or report_type != "A":
            continue
        if report_period < start_date or report_period > end_date:
            continue
        values = [
            code,
            clean_text(row_value(row, columns, "ShortName")),
            report_period,
            report_type,
            correction_flag(row_value(row, columns, "IfCorrect")),
            parse_date(row_value(row, columns, "DeclareDate")),
        ]
        values.extend(
            parse_float(row_value(row, columns, source_column))
            for source_column in spec["columns"]
        )
        values.append(source_file)
        batch.append(tuple(values))
        if len(batch) >= batch_size:
            conn.executemany(statement, batch)
            conn.commit()
            count += len(batch)
            batch.clear()
            if count % (batch_size * 5) == 0:
                elapsed = max(time.monotonic() - started, 0.001)
                print(
                    f"  {path.name}: {count:,} rows "
                    f"({count / elapsed:,.0f} rows/s)",
                    flush=True,
                )
    if batch:
        conn.executemany(statement, batch)
        conn.commit()
        count += len(batch)

    mark_source(conn, source_key, "financial", path, "complete", count)
    print(f"[done] {table}: {count:,}", flush=True)


def import_industry(
    conn,
    raw_root,
    risk_raw_root,
    batch_size,
    overwrite=False,
    industry_file=None,
):
    spec = REFERENCE_SPECS["industry_history"]
    path = (
        Path(industry_file).resolve()
        if industry_file is not None
        else find_unique_file(risk_raw_root, spec["filename"])
    )
    if not path.is_file():
        raise FileNotFoundError(f"Industry-history workbook does not exist: {path}")
    source_file = relative_source(path, raw_root)
    source_key = f"reference:industry:{source_file}"
    if not overwrite and source_is_complete(conn, source_key, path):
        print(f"[skip] {path.name}", flush=True)
        return
    mark_source(conn, source_key, "reference", path, "running")
    conn.execute("DELETE FROM industry_history WHERE source_file=?", (source_file,))
    batch = []
    count = 0
    for _, _, row, columns in iter_csmar_rows(
        path, spec["columns"], spec.get("column_aliases")
    ):
        code = parse_code(row_value(row, columns, "Symbol"))
        classification_id = clean_text(
            row_value(row, columns, "IndustryClassificationID")
        )
        classification_name = clean_text(
            row_value(row, columns, "IndustryClassificationName")
        )
        implement_date = parse_date(row_value(row, columns, "ImplementDate"))
        industry_code = clean_text(row_value(row, columns, "IndustryCode"))
        if not all((code, classification_id, classification_name, implement_date)):
            continue
        batch.append(
            (
                code,
                classification_id,
                classification_name,
                implement_date,
                industry_code or "",
                clean_text(row_value(row, columns, "IndustryName")),
                clean_text(row_value(row, columns, "InstitutionID")),
                source_file,
            )
        )
        if len(batch) >= batch_size:
            conn.executemany(
                """
                INSERT OR REPLACE INTO industry_history
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                batch,
            )
            conn.commit()
            count += len(batch)
            batch.clear()
    if batch:
        conn.executemany(
            "INSERT OR REPLACE INTO industry_history VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            batch,
        )
        conn.commit()
        count += len(batch)
    mark_source(conn, source_key, "reference", path, "complete", count)
    print(f"[done] industry history: {count:,}", flush=True)


def import_reference_table(
    conn,
    raw_root,
    risk_raw_root,
    table,
    spec,
    batch_size,
    overwrite=False,
):
    path = find_unique_file(risk_raw_root, spec["filename"])
    source_file = relative_source(path, raw_root)
    source_key = f"reference:{table}:{source_file}"
    if not overwrite and source_is_complete(conn, source_key, path):
        print(f"[skip] {path.name}", flush=True)
        return
    mark_source(conn, source_key, "reference", path, "running")
    conn.execute(f"DELETE FROM {table} WHERE source_file=?", (source_file,))
    conn.commit()

    statements = {
        "risk_free_daily": (
            """
            INSERT OR REPLACE INTO risk_free_daily
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            lambda row, columns: (
                clean_text(row_value(row, columns, "Nrr1")),
                parse_date(row_value(row, columns, "Clsdt")),
                parse_float(row_value(row, columns, "Nrrdata")),
                parse_float(row_value(row, columns, "Nrrdaydt")),
                parse_float(row_value(row, columns, "Nrrwkdt")),
                parse_float(row_value(row, columns, "Nrrmtdt")),
                source_file,
            ),
        ),
        "market_return_daily": (
            """
            INSERT OR REPLACE INTO market_return_daily
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            lambda row, columns: (
                parse_int(row_value(row, columns, "Markettype")),
                parse_date(row_value(row, columns, "Trddt")),
                parse_float(row_value(row, columns, "Cnshrtrdtl")),
                parse_float(row_value(row, columns, "Cnvaltrdtl")),
                parse_float(row_value(row, columns, "Cdretwdeq")),
                parse_float(row_value(row, columns, "Cdretwdos")),
                parse_float(row_value(row, columns, "Cdretwdtl")),
                parse_int(row_value(row, columns, "Cdnstkcal")),
                source_file,
            ),
        ),
        "index_daily": (
            """
            INSERT OR REPLACE INTO index_daily
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            lambda row, columns: (
                clean_text(row_value(row, columns, "Indexcd")),
                parse_date(row_value(row, columns, "Trddt")),
                parse_float(row_value(row, columns, "Opnindex")),
                parse_float(row_value(row, columns, "Hiindex")),
                parse_float(row_value(row, columns, "Loindex")),
                parse_float(row_value(row, columns, "Clsindex")),
                parse_float(row_value(row, columns, "Retindex")),
                source_file,
            ),
        ),
    }
    statement, transform = statements[table]
    batch = []
    count = 0
    for _, _, row, columns in iter_csmar_rows(
        path, spec["columns"], spec.get("column_aliases")
    ):
        values = transform(row, columns)
        if values[0] is None or values[1] is None:
            continue
        batch.append(values)
        if len(batch) >= batch_size:
            conn.executemany(statement, batch)
            conn.commit()
            count += len(batch)
            batch.clear()
    if batch:
        conn.executemany(statement, batch)
        conn.commit()
        count += len(batch)
    mark_source(conn, source_key, "reference", path, "complete", count)
    print(f"[done] {table}: {count:,}", flush=True)


def discover_daily_files(raw_root):
    return sorted(
        (
            path
            for path in raw_root.rglob("TRD_Dalyr*.xlsx")
            if "risk model" not in str(path.parent).lower()
        ),
        key=lambda path: str(path).lower(),
    )


def import_market_caps(
    conn,
    raw_root,
    start_date,
    end_date,
    multiplier,
    batch_size,
    overwrite=False,
):
    paths = discover_daily_files(raw_root)
    if not paths:
        raise FileNotFoundError(f"No TRD_Dalyr*.xlsx files found below {raw_root}")
    for file_index, path in enumerate(paths, start=1):
        source_file = relative_source(path, raw_root)
        source_key = f"market_cap:{source_file}"
        if not overwrite and source_is_complete(conn, source_key, path):
            print(f"[skip {file_index}/{len(paths)}] {source_file}", flush=True)
            continue
        mark_source(conn, source_key, "market_cap", path, "running")
        conn.execute("DELETE FROM stock_market_cap WHERE source_file=?", (source_file,))
        conn.commit()
        batch = []
        count = 0
        started = time.monotonic()
        for _, _, row, columns in iter_csmar_rows(path, MARKET_CAP_HEADERS):
            code = parse_code(row_value(row, columns, "Stkcd"))
            trade_date = parse_date(row_value(row, columns, "Trddt"))
            if not code or not trade_date:
                continue
            if trade_date < start_date or trade_date > end_date:
                continue
            float_cap = parse_float(row_value(row, columns, "Dsmvosd"))
            total_cap = parse_float(row_value(row, columns, "Dsmvtll"))
            batch.append(
                (
                    code,
                    trade_date,
                    float_cap * multiplier if float_cap is not None else None,
                    total_cap * multiplier if total_cap is not None else None,
                    source_file,
                )
            )
            if len(batch) >= batch_size:
                conn.executemany(
                    """
                    INSERT OR REPLACE INTO stock_market_cap
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    batch,
                )
                conn.commit()
                count += len(batch)
                batch.clear()
                if count % (batch_size * 10) == 0:
                    elapsed = max(time.monotonic() - started, 0.001)
                    print(
                        f"  [{file_index}/{len(paths)}] {path.name}: "
                        f"{count:,} rows ({count / elapsed:,.0f} rows/s)",
                        flush=True,
                    )
        if batch:
            conn.executemany(
                "INSERT OR REPLACE INTO stock_market_cap VALUES (?, ?, ?, ?, ?)",
                batch,
            )
            conn.commit()
            count += len(batch)
        mark_source(conn, source_key, "market_cap", path, "complete", count)
        print(
            f"[done {file_index}/{len(paths)}] {source_file}: {count:,}",
            flush=True,
        )


def import_forward_market_caps(
    conn,
    raw_root,
    forward_root,
    start_date,
    end_date,
    batch_size,
    overwrite=False,
):
    if not forward_root.exists():
        print(f"[skip] forward market-data directory not found: {forward_root}")
        return
    paths = sorted(
        (
            path
            for path in forward_root.glob("*.xlsx")
            if not path.name.startswith("~$")
        ),
        key=lambda path: path.name.lower(),
    )
    if not paths:
        print(f"[skip] no forward market-data XLSX files in {forward_root}")
        return

    for file_index, path in enumerate(paths, start=1):
        source_file = relative_source(path, raw_root)
        source_key = f"market_cap_forward:{path.resolve()}"
        if not overwrite and source_is_complete(conn, source_key, path):
            print(
                f"[skip forward {file_index}/{len(paths)}] {path.name}",
                flush=True,
            )
            continue
        mark_source(conn, source_key, "market_cap_forward", path, "running")
        conn.execute(
            "DELETE FROM stock_market_cap WHERE source_file=?", (source_file,)
        )
        conn.commit()
        batch = []
        count = 0
        for _, _, row, columns in iter_csmar_rows(
            path, FORWARD_MARKET_CAP_HEADERS
        ):
            code = parse_code(row_value(row, columns, "Symbol"))
            trade_date = parse_date(row_value(row, columns, "TradingDate"))
            state_code = parse_int(row_value(row, columns, "StateCode"))
            if not code or not trade_date or state_code == 2:
                continue
            if trade_date < start_date or trade_date > end_date:
                continue
            total_cap = parse_float(row_value(row, columns, "MarketValue"))
            float_cap = None
            for header in ("AValue", "CirculatedMarketValue"):
                candidate = parse_float(row_value(row, columns, header))
                if candidate is not None and candidate > 0:
                    float_cap = candidate
                    break
            if total_cap is None or total_cap <= 0:
                continue
            batch.append(
                (code, trade_date, float_cap, total_cap, source_file)
            )
            if len(batch) >= batch_size:
                conn.executemany(
                    """
                    INSERT OR REPLACE INTO stock_market_cap
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    batch,
                )
                conn.commit()
                count += len(batch)
                batch.clear()
        if batch:
            conn.executemany(
                "INSERT OR REPLACE INTO stock_market_cap VALUES (?, ?, ?, ?, ?)",
                batch,
            )
            conn.commit()
            count += len(batch)
        mark_source(
            conn,
            source_key,
            "market_cap_forward",
            path,
            "complete",
            count,
        )
        print(
            f"[done forward {file_index}/{len(paths)}] "
            f"{path.name}: {count:,}",
            flush=True,
        )


def max_date(*values):
    usable = [str(value) for value in values if value]
    return max(usable) if usable else None


def add_optional(*values):
    if any(value is None for value in values):
        return None
    return sum(values)


def ttm_value(current, prior_annual, prior_same):
    if current is None or prior_annual is None or prior_same is None:
        return None
    return current + prior_annual - prior_same


def derive_ttm_rows(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["code"]].append(row)
    updates = []
    ttm_by_key = {}

    for code, code_rows in grouped.items():
        by_period = {row["report_period"]: row for row in code_rows}
        for row in sorted(code_rows, key=lambda item: item["report_period"]):
            period = row["report_period"]
            year = int(period[:4])
            suffix = period[4:]
            if suffix == "-12-31":
                components = [row]
                revenue_ttm = row["operating_revenue_ytd"]
                net_profit_ttm = row["net_profit_ytd"]
                parent_profit_ttm = row["parent_net_profit_ytd"]
                cashflow_ttm = row["operating_cashflow_ytd"]
                capex_ttm = row["capital_expenditure_ytd"]
            else:
                prior_annual = by_period.get(f"{year - 1}-12-31")
                prior_same = by_period.get(f"{year - 1}{suffix}")
                if prior_annual is None or prior_same is None:
                    components = []
                    revenue_ttm = None
                    net_profit_ttm = None
                    parent_profit_ttm = None
                    cashflow_ttm = None
                    capex_ttm = None
                else:
                    components = [row, prior_annual, prior_same]
                    revenue_ttm = ttm_value(
                        row["operating_revenue_ytd"],
                        prior_annual["operating_revenue_ytd"],
                        prior_same["operating_revenue_ytd"],
                    )
                    net_profit_ttm = ttm_value(
                        row["net_profit_ytd"],
                        prior_annual["net_profit_ytd"],
                        prior_same["net_profit_ytd"],
                    )
                    parent_profit_ttm = ttm_value(
                        row["parent_net_profit_ytd"],
                        prior_annual["parent_net_profit_ytd"],
                        prior_same["parent_net_profit_ytd"],
                    )
                    cashflow_ttm = ttm_value(
                        row["operating_cashflow_ytd"],
                        prior_annual["operating_cashflow_ytd"],
                        prior_same["operating_cashflow_ytd"],
                    )
                    capex_ttm = ttm_value(
                        row["capital_expenditure_ytd"],
                        prior_annual["capital_expenditure_ytd"],
                        prior_same["capital_expenditure_ytd"],
                    )
            component_dates = [item["available_date"] for item in components]
            ttm_available = (
                max_date(*component_dates)
                if components and all(component_dates)
                else None
            )
            key = (code, period)
            ttm_by_key[key] = {
                "revenue_ttm": revenue_ttm,
                "net_profit_ttm": net_profit_ttm,
                "parent_net_profit_ttm": parent_profit_ttm,
                "operating_cashflow_ttm": cashflow_ttm,
                "capital_expenditure_ttm": capex_ttm,
                "ttm_available_date": ttm_available,
            }

        for row in code_rows:
            current = ttm_by_key[(code, row["report_period"])]
            year = int(row["report_period"][:4])
            prior_key = (code, f"{year - 1}{row['report_period'][4:]}")
            prior = ttm_by_key.get(prior_key)
            revenue_growth = None
            earnings_growth = None
            growth_available = None
            if prior and current["ttm_available_date"] and prior["ttm_available_date"]:
                prior_revenue = prior["revenue_ttm"]
                prior_earnings = prior["parent_net_profit_ttm"]
                if prior_revenue not in (None, 0):
                    revenue_growth = (
                        current["revenue_ttm"] - prior_revenue
                        if current["revenue_ttm"] is not None
                        else None
                    )
                    if revenue_growth is not None:
                        revenue_growth /= abs(prior_revenue)
                if prior_earnings not in (None, 0):
                    earnings_growth = (
                        current["parent_net_profit_ttm"] - prior_earnings
                        if current["parent_net_profit_ttm"] is not None
                        else None
                    )
                    if earnings_growth is not None:
                        earnings_growth /= abs(prior_earnings)
                growth_available = max_date(
                    current["ttm_available_date"], prior["ttm_available_date"]
                )
            updates.append(
                (
                    current["revenue_ttm"],
                    current["net_profit_ttm"],
                    current["parent_net_profit_ttm"],
                    current["operating_cashflow_ttm"],
                    current["capital_expenditure_ttm"],
                    current["ttm_available_date"],
                    revenue_growth,
                    earnings_growth,
                    growth_available,
                    code,
                    row["report_period"],
                )
            )
    return updates


def consolidate_financial_pit(conn):
    print("[build] point-in-time financial snapshots", flush=True)
    conn.execute("DELETE FROM financial_pit")
    conn.execute(
        """
        INSERT INTO financial_pit (
            code, name, report_period, actual_disclosure_date, available_date,
            total_assets, total_liabilities, parent_equity, total_equity,
            cash, current_assets, current_liabilities,
            operating_revenue_ytd, operating_profit_ytd, total_profit_ytd,
            net_profit_ytd, parent_net_profit_ytd, research_development_ytd,
            operating_cashflow_ytd, capital_expenditure_ytd,
            dividend_interest_paid_ytd, is_corrected
        )
        SELECT
            keys.code,
            COALESCE(b.name, i.name, c.name),
            keys.report_period,
            d.actual_disclosure_date,
            NULLIF(MAX(
                COALESCE(d.actual_disclosure_date, ''),
                CASE WHEN COALESCE(b.is_corrected, 0)=1
                     THEN COALESCE(b.correction_disclosure_date, '') ELSE '' END,
                CASE WHEN COALESCE(i.is_corrected, 0)=1
                     THEN COALESCE(i.correction_disclosure_date, '') ELSE '' END,
                CASE WHEN COALESCE(c.is_corrected, 0)=1
                     THEN COALESCE(c.correction_disclosure_date, '') ELSE '' END
            ), ''),
            b.total_assets,
            b.total_liabilities,
            b.parent_equity,
            b.total_equity,
            b.cash,
            b.current_assets,
            b.current_liabilities,
            COALESCE(i.operating_revenue_ytd, i.total_operating_revenue_ytd),
            i.operating_profit_ytd,
            i.total_profit_ytd,
            i.net_profit_ytd,
            i.parent_net_profit_ytd,
            i.research_development_ytd,
            c.operating_cashflow_ytd,
            c.capital_expenditure_ytd,
            c.dividend_interest_paid_ytd,
            MAX(
                COALESCE(b.is_corrected, 0),
                COALESCE(i.is_corrected, 0),
                COALESCE(c.is_corrected, 0)
            )
        FROM (
            SELECT code, report_period FROM balance_statement
            UNION
            SELECT code, report_period FROM income_statement
            UNION
            SELECT code, report_period FROM cashflow_statement
        ) AS keys
        LEFT JOIN balance_statement b
          ON b.code=keys.code AND b.report_period=keys.report_period
         AND b.report_type='A'
        LEFT JOIN income_statement i
          ON i.code=keys.code AND i.report_period=keys.report_period
         AND i.report_type='A'
        LEFT JOIN cashflow_statement c
          ON c.code=keys.code AND c.report_period=keys.report_period
         AND c.report_type='A'
        LEFT JOIN financial_report_dates d
          ON d.code=keys.code AND d.report_period=keys.report_period
        """
    )
    conn.commit()
    conn.row_factory = sqlite3.Row
    rows = [
        dict(row)
        for row in conn.execute(
            """
            SELECT code, report_period, available_date,
                   operating_revenue_ytd, net_profit_ytd,
                   parent_net_profit_ytd, operating_cashflow_ytd,
                   capital_expenditure_ytd
            FROM financial_pit
            ORDER BY code, report_period
            """
        )
    ]
    updates = derive_ttm_rows(rows)
    conn.executemany(
        """
        UPDATE financial_pit SET
            revenue_ttm=?,
            net_profit_ttm=?,
            parent_net_profit_ttm=?,
            operating_cashflow_ttm=?,
            capital_expenditure_ttm=?,
            ttm_available_date=?,
            revenue_growth=?,
            earnings_growth=?,
            growth_available_date=?
        WHERE code=? AND report_period=?
        """,
        updates,
    )
    conn.commit()
    conn.row_factory = None
    count, usable, ttm = conn.execute(
        """
        SELECT COUNT(*),
               SUM(available_date IS NOT NULL),
               SUM(ttm_available_date IS NOT NULL)
        FROM financial_pit
        """
    ).fetchone()
    print(
        f"[done] financial_pit: rows={count:,}, available={usable:,}, ttm={ttm:,}",
        flush=True,
    )


def update_metadata(conn, args):
    market_cap_stats = conn.execute(
        """
        SELECT COUNT(*), COUNT(DISTINCT code), MIN(trade_date), MAX(trade_date)
        FROM stock_market_cap
        """
    ).fetchone()
    financial_stats = conn.execute(
        """
        SELECT COUNT(*), COUNT(DISTINCT code), MIN(report_period), MAX(report_period)
        FROM financial_pit
        """
    ).fetchone()
    industry_stats = conn.execute(
        """
        SELECT COUNT(*), COUNT(DISTINCT code),
               MIN(implement_date), MAX(implement_date)
        FROM industry_history
        """
    ).fetchone()
    metadata = {
        "schema_version": "1",
        "built_at": now_iso(),
        "raw_root": str(args.raw_root),
        "risk_raw_root": str(args.risk_raw_root),
        "forward_market_root": str(args.forward_market_root),
        "start_date": args.start_date,
        "end_date": args.end_date,
        "market_cap_unit_multiplier": args.market_cap_unit_multiplier,
        "market_cap": market_cap_stats,
        "financial_pit": financial_stats,
        "industry_history": industry_stats,
    }
    conn.execute("DELETE FROM risk_data_metadata")
    conn.executemany(
        "INSERT INTO risk_data_metadata(key, value) VALUES (?, ?)",
        (
            (key, json.dumps(value, ensure_ascii=False))
            for key, value in metadata.items()
        ),
    )
    conn.commit()
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)


def run(args):
    args.output.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(args.output, timeout=120)
    try:
        create_schema(conn)
        stages = set(args.stage)
        if "all" in stages:
            stages = {"reference", "financial", "market-cap", "finalize"}

        if "reference" in stages:
            import_report_dates(conn, args.risk_raw_root, args.overwrite)
            import_industry(
                conn,
                args.raw_root,
                args.risk_raw_root,
                args.batch_size,
                args.overwrite,
                args.industry_file,
            )
            for table in ("risk_free_daily", "market_return_daily", "index_daily"):
                import_reference_table(
                    conn,
                    args.raw_root,
                    args.risk_raw_root,
                    table,
                    REFERENCE_SPECS[table],
                    args.batch_size,
                    args.overwrite,
                )

        if "financial" in stages:
            for table, spec in FINANCIAL_SPECS.items():
                import_financial_table(
                    conn,
                    args.raw_root,
                    args.risk_raw_root,
                    table,
                    spec,
                    args.financial_start_date,
                    args.end_date,
                    args.batch_size,
                    args.overwrite,
                )

        if "market-cap" in stages:
            import_market_caps(
                conn,
                args.raw_root,
                args.start_date,
                args.end_date,
                args.market_cap_unit_multiplier,
                args.batch_size,
                args.overwrite,
            )
            import_forward_market_caps(
                conn,
                args.raw_root,
                args.forward_market_root,
                args.start_date,
                args.end_date,
                args.batch_size,
                args.overwrite or args.overwrite_forward,
            )

        if "finalize" in stages:
            consolidate_financial_pit(conn)
            update_metadata(conn, args)
            conn.execute("PRAGMA optimize")
            conn.commit()
    finally:
        conn.close()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Build the point-in-time CSMAR risk-model sidecar database."
    )
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--risk-raw-root", type=Path, default=DEFAULT_RISK_RAW_ROOT)
    parser.add_argument(
        "--industry-file",
        type=Path,
        help="Explicit full-history STK_INDUSTRYCLASS workbook.",
    )
    parser.add_argument(
        "--forward-market-root",
        type=Path,
        default=DEFAULT_FORWARD_MARKET_ROOT,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--start-date", default="2019-01-01")
    parser.add_argument("--financial-start-date", default="2017-01-01")
    parser.add_argument("--end-date", default="2026-07-24")
    parser.add_argument("--market-cap-unit-multiplier", type=float, default=1000.0)
    parser.add_argument("--batch-size", type=int, default=10000)
    parser.add_argument(
        "--stage",
        action="append",
        choices=("all", "reference", "financial", "market-cap", "finalize"),
        default=[],
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-import source files even when their size and mtime are unchanged.",
    )
    parser.add_argument(
        "--overwrite-forward",
        action="store_true",
        help=(
            "Re-import only forward weekly market files while retaining cached "
            "historical market-cap workbooks."
        ),
    )
    args = parser.parse_args(argv)
    if not args.stage:
        args.stage = ["all"]
    args.raw_root = args.raw_root.resolve()
    args.risk_raw_root = args.risk_raw_root.resolve()
    if args.industry_file is not None:
        args.industry_file = args.industry_file.resolve()
    args.forward_market_root = args.forward_market_root.resolve()
    args.output = args.output.resolve()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.market_cap_unit_multiplier <= 0:
        parser.error("--market-cap-unit-multiplier must be positive")
    if args.financial_start_date > args.start_date:
        parser.error("--financial-start-date must not be later than --start-date")
    if args.start_date > args.end_date:
        parser.error("--start-date must not be later than --end-date")
    return args


if __name__ == "__main__":
    run(parse_args())
