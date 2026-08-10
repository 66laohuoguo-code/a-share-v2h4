"""Build a model-compatible SQLite database from full CSMAR XLSX exports."""

from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import Counter
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
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


A_MARKET_TYPES = {1, 4, 16, 32, 64}
SPECIAL_TREATMENT_STATES = {2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 16}
DAILY_REQUIRED_HEADERS = {
    "Stkcd",
    "Trddt",
    "Opnprc",
    "Hiprc",
    "Loprc",
    "Clsprc",
    "Dnshrtrd",
    "Dnvaltrd",
    "Dsmvosd",
    "Dsmvtll",
    "Dretwd",
    "Dretnd",
    "Adjprcwd",
    "Adjprcnd",
    "Markettype",
    "Trdsta",
    "PreClosePrice",
    "ChangeRatio",
}
STOCK_DAILY_COLUMNS = (
    "code",
    "name",
    "trade_date",
    "prev_close",
    "open",
    "high",
    "low",
    "close",
    "raw_prev_close",
    "raw_open",
    "raw_high",
    "raw_low",
    "raw_close",
    "adj_close_1",
    "adj_close_2",
    "volume",
    "amount",
    "turnover_total",
    "turnover_float",
    "adj_factor",
    "daily_return",
    "capital_return",
    "risk_free_return",
    "limit_down",
    "limit_up",
    "limit_status",
    "no_price_limit",
    "listed_state",
    "currency",
    "industry_1",
    "industry_2",
    "source_file",
    "source_sheet",
    "imported_at",
)

FILE_ALIASES = {
    "company": ("TRD_Co.xlsx", "股票基本信息.xlsx"),
    "adjust_factor": ("TRD_AdjustFactor.xlsx", "股票复权因子.xlsx"),
    "no_limit": ("TRD_NoLimit.xlsx", "无涨跌停限制.xlsx"),
    "industry": ("STK_INDUSTRYCLASS.xlsx", "上市公司行业分类.xlsx"),
}

CSRC_2001_CLASSIFICATION = "证监会行业分类2001年版"
CSRC_2012_CLASSIFICATION = "证监会行业分类2012年版"
CSRC_2012_EFFECTIVE_DATE = "2012-10-26"
ASSOCIATION_CLASSIFICATION = "中国上市公司协会上市公司行业分类"


def now_iso():
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def natural_key(path):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", str(path))]


def header_map(header):
    return {str(value).strip(): idx for idx, value in enumerate(header) if value is not None}


def row_value(row, columns, name):
    idx = columns.get(name)
    return row[idx] if idx is not None and idx < len(row) else None


def clean_text(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text or text in {"没有单位", "None", "NULL", "null"}:
        return None
    return text


def parse_int(value):
    number = parse_float(value)
    return int(number) if number is not None and math.isfinite(number) else None


def positive(value):
    number = parse_float(value)
    return number if number is not None and math.isfinite(number) and number > 0 else None


def nonnegative(value):
    number = parse_float(value)
    return number if number is not None and math.isfinite(number) and number >= 0 else None


def read_one_sheet(path):
    zf = zipfile.ZipFile(path)
    shared = load_shared_strings(zf)
    sheets = workbook_sheets(zf)
    if len(sheets) != 1:
        zf.close()
        raise ValueError(f"Expected one worksheet in {path}; found {len(sheets)}")
    sheet_name, sheet_path = sheets[0]
    return zf, sheet_name, iter_sheet_rows(zf, sheet_path, shared)


def find_alias_files(root, aliases):
    matches = []
    for filename in aliases:
        matches.extend(root.rglob(filename))
    return sorted(set(matches), key=natural_key)


class CausalIndustryHistory:
    def __init__(self, rows):
        grouped = {}
        for code, classification_name, implement_date, industry_code, industry_name in rows:
            key = (code, classification_name)
            grouped.setdefault(key, []).append(
                (implement_date, industry_code, industry_name)
            )
        self.history = {}
        for key, values in grouped.items():
            values.sort(key=lambda item: item[0])
            self.history[key] = ([item[0] for item in values], values)

    def get(self, code, trade_date):
        classification = (
            CSRC_2001_CLASSIFICATION
            if trade_date < CSRC_2012_EFFECTIVE_DATE
            else CSRC_2012_CLASSIFICATION
        )
        # The association classification is a dated fallback for newly listed
        # stocks not yet covered by the selected CSRC release. It shares the
        # leading national-economic-industry letter used by the broad model.
        for candidate in (classification, ASSOCIATION_CLASSIFICATION):
            dates_and_values = self.history.get((code, candidate))
            if not dates_and_values:
                continue
            dates, values = dates_and_values
            index = bisect_right(dates, trade_date) - 1
            if index < 0:
                continue
            implement_date, industry_code, industry_name = values[index]
            return {
                "implement_date": implement_date,
                "industry_code": industry_code,
                "industry_name": industry_name,
                "classification_name": candidate,
            }
        return None


def load_industry_history(path):
    records = []
    zf, _, rows = read_one_sheet(path)
    try:
        columns = header_map(next(rows, None) or [])
        required = {
            "Symbol", "IndustryClassificationName", "ImplementDate",
            "IndustryCode", "IndustryName",
        }
        missing = sorted(required - set(columns))
        if missing:
            raise ValueError(f"Missing industry-history headers in {path.name}: {missing}")
        for row in rows:
            code = parse_code(row_value(row, columns, "Symbol"))
            classification_name = clean_text(
                row_value(row, columns, "IndustryClassificationName")
            )
            implement_date = parse_date(row_value(row, columns, "ImplementDate"))
            industry_code = clean_text(row_value(row, columns, "IndustryCode"))
            if (
                code
                and classification_name in {
                    CSRC_2001_CLASSIFICATION,
                    CSRC_2012_CLASSIFICATION,
                    ASSOCIATION_CLASSIFICATION,
                }
                and implement_date
                and industry_code
            ):
                records.append(
                    (
                        code,
                        classification_name,
                        implement_date,
                        industry_code,
                        clean_text(row_value(row, columns, "IndustryName")),
                    )
                )
    finally:
        zf.close()
    return CausalIndustryHistory(records), len(records)


def load_legacy_metadata(path):
    metadata = {}
    if path is None or not path.exists():
        return metadata
    with sqlite3.connect(path) as conn:
        found = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='stock_meta'"
        ).fetchone()
        if not found:
            return metadata
        for code, name, industry_1, industry_2 in conn.execute(
            "SELECT code, name, industry_1, industry_2 FROM stock_meta"
        ):
            normalized = parse_code(code)
            if normalized:
                metadata[normalized] = {
                    "name": name or normalized,
                    "industry_1": industry_1 or "UNKNOWN",
                    "industry_2": industry_2 or industry_1 or "UNKNOWN",
                }
    return metadata


def load_company_file(path, legacy):
    companies = {}
    zf, sheet_name, rows = read_one_sheet(path)
    try:
        header = next(rows, None)
        columns = header_map(header or [])
        required = {"Stkcd", "Stknme", "Listdt", "Markettype"}
        missing = sorted(required - set(columns))
        if missing:
            raise ValueError(f"Missing company headers in {path.name}: {', '.join(missing)}")
        for row in rows:
            code = parse_code(row_value(row, columns, "Stkcd"))
            market_type = parse_int(row_value(row, columns, "Markettype"))
            if not code or market_type not in A_MARKET_TYPES:
                continue
            fallback = legacy.get(code, {})
            industry_code_c = clean_text(row_value(row, columns, "Nnindcd"))
            companies[code] = {
                "code": code,
                "name": clean_text(row_value(row, columns, "Stknme")) or fallback.get("name") or code,
                "company_name": clean_text(row_value(row, columns, "Conme")),
                "list_date": parse_date(row_value(row, columns, "Listdt")),
                "currency": clean_text(row_value(row, columns, "Curtrd")) or "CNY",
                "market_type": market_type,
                "ab_cross_code": parse_code(row_value(row, columns, "Crcd")),
                "h_cross_code": clean_text(row_value(row, columns, "Commnt")),
                "status_date": parse_date(row_value(row, columns, "Statdt")),
                "company_status": clean_text(row_value(row, columns, "Statco")),
                "former_code": clean_text(row_value(row, columns, "FormerCode")),
                "established_date": parse_date(row_value(row, columns, "Estbdt")),
                "issue_date": parse_date(row_value(row, columns, "Ipodt")),
                "industry_code_c": industry_code_c,
                "industry_name_c": clean_text(row_value(row, columns, "Nnindnme")),
                "industry_code_d": clean_text(row_value(row, columns, "IndcdZX")),
                "industry_name_d": clean_text(row_value(row, columns, "IndnmeZX")),
                "industry_1": industry_code_c[:1] if industry_code_c else fallback.get("industry_1"),
                "industry_2": industry_code_c or fallback.get("industry_2"),
                "source_file": path.name,
                "source_sheet": sheet_name,
            }
    finally:
        zf.close()
    return companies


def create_schema(conn):
    conn.executescript(
        """
        PRAGMA journal_mode=OFF;
        PRAGMA synchronous=OFF;
        PRAGMA temp_store=MEMORY;
        PRAGMA cache_size=-262144;
        PRAGMA foreign_keys=OFF;

        CREATE TABLE stock_daily (
            code TEXT NOT NULL,
            name TEXT,
            trade_date TEXT NOT NULL,
            prev_close REAL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            raw_prev_close REAL,
            raw_open REAL,
            raw_high REAL,
            raw_low REAL,
            raw_close REAL,
            adj_close_1 REAL,
            adj_close_2 REAL,
            volume REAL,
            amount REAL,
            turnover_total REAL,
            turnover_float REAL,
            adj_factor REAL,
            daily_return REAL,
            capital_return REAL,
            risk_free_return REAL,
            limit_down REAL,
            limit_up REAL,
            limit_status INTEGER,
            no_price_limit INTEGER NOT NULL DEFAULT 0,
            listed_state TEXT,
            currency TEXT,
            industry_1 TEXT,
            industry_2 TEXT,
            source_file TEXT,
            source_sheet TEXT,
            imported_at TEXT NOT NULL,
            PRIMARY KEY (code, trade_date)
        ) WITHOUT ROWID;

        CREATE TABLE csmar_company (
            code TEXT PRIMARY KEY,
            name TEXT,
            company_name TEXT,
            list_date TEXT,
            currency TEXT,
            market_type INTEGER,
            ab_cross_code TEXT,
            h_cross_code TEXT,
            status_date TEXT,
            company_status TEXT,
            former_code TEXT,
            established_date TEXT,
            issue_date TEXT,
            industry_code_c TEXT,
            industry_name_c TEXT,
            industry_code_d TEXT,
            industry_name_d TEXT,
            source_file TEXT,
            source_sheet TEXT
        ) WITHOUT ROWID;

        CREATE TABLE csmar_adjust_factor (
            code TEXT NOT NULL,
            trade_date TEXT NOT NULL,
            fward_factor REAL,
            bward_factor REAL,
            cumulate_fward_factor REAL,
            cumulate_bward_factor REAL,
            source_file TEXT,
            PRIMARY KEY (code, trade_date)
        ) WITHOUT ROWID;

        CREATE TABLE csmar_no_limit (
            code TEXT NOT NULL,
            trade_date TEXT NOT NULL,
            reason TEXT,
            source_file TEXT,
            PRIMARY KEY (code, trade_date)
        ) WITHOUT ROWID;

        CREATE TABLE project_metadata (
            key TEXT PRIMARY KEY,
            value TEXT
        ) WITHOUT ROWID;

        CREATE TABLE csmar_daily_import_state (
            source_file TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            written_rows INTEGER NOT NULL DEFAULT 0,
            completed_at TEXT
        ) WITHOUT ROWID;
        """
    )


def ensure_resume_schema(conn):
    required_tables = {"stock_daily", "csmar_company", "csmar_no_limit"}
    existing_tables = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }
    missing = sorted(required_tables - existing_tables)
    if missing:
        raise ValueError(
            f"Partial database cannot be resumed; missing tables: {missing}"
        )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS csmar_daily_import_state (
            source_file TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            written_rows INTEGER NOT NULL DEFAULT 0,
            completed_at TEXT
        ) WITHOUT ROWID
        """
    )
    completed_at = now_iso()
    for source_file, written_rows in conn.execute(
        """
        SELECT source_file, COUNT(*)
        FROM stock_daily
        WHERE source_file IS NOT NULL
        GROUP BY source_file
        """
    ).fetchall():
        conn.execute(
            """
            INSERT OR IGNORE INTO csmar_daily_import_state
            VALUES (?, 'complete', ?, ?)
            """,
            (source_file, int(written_rows), completed_at),
        )
    conn.commit()


def mark_daily_file_complete(conn, source_file, written_rows):
    conn.execute(
        """
        INSERT OR REPLACE INTO csmar_daily_import_state
        VALUES (?, 'complete', ?, ?)
        """,
        (str(source_file), int(written_rows), now_iso()),
    )
    conn.commit()


def insert_company_rows(conn, companies):
    columns = (
        "code", "name", "company_name", "list_date", "currency", "market_type",
        "ab_cross_code", "h_cross_code", "status_date", "company_status", "former_code",
        "established_date", "issue_date", "industry_code_c", "industry_name_c",
        "industry_code_d", "industry_name_d", "source_file", "source_sheet",
    )
    placeholders = ",".join("?" for _ in columns)
    conn.executemany(
        f"INSERT OR REPLACE INTO csmar_company ({','.join(columns)}) VALUES ({placeholders})",
        [tuple(company.get(column) for column in columns) for company in companies.values()],
    )
    conn.commit()


def import_adjust_factors(conn, path, start_date, end_date):
    counters = Counter()
    records = []
    zf, _, rows = read_one_sheet(path)
    try:
        columns = header_map(next(rows, None) or [])
        required = {
            "TradingDate", "Symbol", "FwardFactor", "BwardFactor",
            "CumulateFwardFactor", "CumulateBwardFactor",
        }
        if required - set(columns):
            raise ValueError(f"Missing adjustment-factor columns: {sorted(required - set(columns))}")
        for row in rows:
            counters["source_rows"] += 1
            code = parse_code(row_value(row, columns, "Symbol"))
            trade_date = parse_date(row_value(row, columns, "TradingDate"))
            if not code or not trade_date or not (start_date <= trade_date <= end_date):
                counters["skipped_rows"] += 1
                continue
            records.append(
                (
                    code,
                    trade_date,
                    parse_float(row_value(row, columns, "FwardFactor")),
                    parse_float(row_value(row, columns, "BwardFactor")),
                    parse_float(row_value(row, columns, "CumulateFwardFactor")),
                    parse_float(row_value(row, columns, "CumulateBwardFactor")),
                    path.name,
                )
            )
        conn.executemany(
            "INSERT OR REPLACE INTO csmar_adjust_factor VALUES (?,?,?,?,?,?,?)", records
        )
        conn.commit()
        counters["written_rows"] = len(records)
        return counters
    finally:
        zf.close()


def import_no_limit(conn, path, start_date, end_date):
    counters = Counter()
    records = []
    zf, _, rows = read_one_sheet(path)
    try:
        columns = header_map(next(rows, None) or [])
        required = {"TradingDate", "Symbol", "Reason"}
        if required - set(columns):
            raise ValueError(f"Missing no-limit columns: {sorted(required - set(columns))}")
        for row in rows:
            counters["source_rows"] += 1
            code = parse_code(row_value(row, columns, "Symbol"))
            trade_date = parse_date(row_value(row, columns, "TradingDate"))
            if not code or not trade_date or not (start_date <= trade_date <= end_date):
                counters["skipped_rows"] += 1
                continue
            records.append((code, trade_date, row_value(row, columns, "Reason"), path.name))
        conn.executemany(
            "INSERT OR REPLACE INTO csmar_no_limit VALUES (?,?,?,?)", records
        )
        conn.commit()
        counters["written_rows"] = len(records)
        return counters
    finally:
        zf.close()


def turnover_rate(volume, close, market_value_thousand):
    if not volume or not close or not market_value_thousand or market_value_thousand <= 0:
        return None
    shares = market_value_thousand * 1000.0 / close
    return volume * 100.0 / shares if shares > 0 else None


def daily_record(
    row,
    columns,
    source_file,
    source_sheet,
    source_root,
    imported_at,
    metadata,
    industry_history=None,
    no_limit_keys=None,
):
    code = parse_code(row_value(row, columns, "Stkcd"))
    trade_date = parse_date(row_value(row, columns, "Trddt"))
    market_type = parse_int(row_value(row, columns, "Markettype"))
    if not code or not trade_date:
        return None, "metadata_or_invalid_key"
    if market_type not in A_MARKET_TYPES:
        return None, "non_a_share"

    close = positive(row_value(row, columns, "Clsprc"))
    open_price = positive(row_value(row, columns, "Opnprc"))
    high = positive(row_value(row, columns, "Hiprc"))
    low = positive(row_value(row, columns, "Loprc"))
    daily_return = parse_float(row_value(row, columns, "Dretwd"))
    capital_return = parse_float(row_value(row, columns, "Dretnd"))
    if close is None or daily_return is None or capital_return is None:
        return None, "missing_required_price_or_return"

    prev_close = positive(row_value(row, columns, "PreClosePrice"))
    volume = nonnegative(row_value(row, columns, "Dnshrtrd"))
    amount = nonnegative(row_value(row, columns, "Dnvaltrd"))
    total_market_value = positive(row_value(row, columns, "Dsmvtll"))
    float_market_value = positive(row_value(row, columns, "Dsmvosd"))
    adj_close_1 = positive(row_value(row, columns, "Adjprcwd"))
    adj_close_2 = positive(row_value(row, columns, "Adjprcnd"))
    trading_state = parse_int(row_value(row, columns, "Trdsta"))
    info = metadata.get(code, {})
    point_in_time_industry = (
        industry_history.get(code, trade_date) if industry_history is not None else None
    )
    if point_in_time_industry:
        industry_1 = point_in_time_industry["industry_code"][:1]
        industry_2 = point_in_time_industry["industry_code"]
    elif industry_history is None:
        industry_1 = info.get("industry_1") or "UNKNOWN"
        industry_2 = info.get("industry_2") or industry_1
    else:
        industry_1 = "UNKNOWN"
        industry_2 = "UNKNOWN"

    change_ratio = parse_float(row_value(row, columns, "ChangeRatio"))
    if prev_close and change_ratio is not None:
        expected = close / prev_close - 1.0
        return_gap = abs(expected - change_ratio)
    else:
        return_gap = None

    record = {
        "code": code,
        "name": info.get("name") or code,
        "trade_date": trade_date,
        "prev_close": prev_close,
        "open": open_price,
        "high": high,
        "low": low,
        "close": close,
        "raw_prev_close": prev_close,
        "raw_open": open_price,
        "raw_high": high,
        "raw_low": low,
        "raw_close": close,
        "adj_close_1": adj_close_1,
        "adj_close_2": adj_close_2,
        "volume": volume,
        "amount": amount,
        "turnover_total": turnover_rate(volume, close, total_market_value),
        "turnover_float": turnover_rate(volume, close, float_market_value),
        "adj_factor": adj_close_1 / close if adj_close_1 and close else None,
        "daily_return": daily_return,
        "capital_return": capital_return,
        "risk_free_return": None,
        "limit_down": positive(row_value(row, columns, "LimitDown")),
        "limit_up": positive(row_value(row, columns, "LimitUp")),
        "limit_status": parse_int(row_value(row, columns, "LimitStatus")),
        "no_price_limit": int(
            no_limit_keys is not None and (code, trade_date) in no_limit_keys
        ),
        "listed_state": "ST" if trading_state in SPECIAL_TREATMENT_STATES else "Norm",
        "currency": "CNY",
        "industry_1": industry_1,
        "industry_2": industry_2,
        "source_file": str(source_file.relative_to(source_root)),
        "source_sheet": source_sheet,
        "imported_at": imported_at,
    }
    return record, return_gap


def insert_daily_batch(conn, records):
    placeholders = ",".join("?" for _ in STOCK_DAILY_COLUMNS)
    conn.executemany(
        f"INSERT OR REPLACE INTO stock_daily ({','.join(STOCK_DAILY_COLUMNS)}) VALUES ({placeholders})",
        [tuple(record.get(column) for column in STOCK_DAILY_COLUMNS) for record in records],
    )


def import_daily_file(
    conn,
    path,
    source_root,
    start_date,
    end_date,
    metadata,
    batch_size,
    industry_history=None,
    no_limit_keys=None,
):
    started = time.monotonic()
    counters = Counter()
    imported_at = now_iso()
    batch = []
    dates = set()
    min_date = None
    max_date = None
    zf, sheet_name, rows = read_one_sheet(path)
    try:
        header = next(rows, None)
        columns = header_map(header or [])
        missing = sorted(DAILY_REQUIRED_HEADERS - set(columns))
        if missing:
            raise ValueError(f"Missing daily headers in {path.name}: {', '.join(missing)}")

        for row in rows:
            counters["source_rows"] += 1
            record, diagnostic = daily_record(
                row,
                columns,
                path,
                sheet_name,
                source_root,
                imported_at,
                metadata,
                industry_history,
                no_limit_keys,
            )
            if record is None:
                counters[str(diagnostic)] += 1
                continue
            trade_date = record["trade_date"]
            if trade_date < start_date or trade_date > end_date:
                counters["outside_date_range"] += 1
                continue
            if diagnostic is not None and diagnostic > 5e-6:
                counters["change_ratio_mismatch"] += 1
            batch.append(record)
            dates.add(trade_date)
            min_date = trade_date if min_date is None or trade_date < min_date else min_date
            max_date = trade_date if max_date is None or trade_date > max_date else max_date
            if len(batch) >= batch_size:
                insert_daily_batch(conn, batch)
                counters["written_rows"] += len(batch)
                batch.clear()
                if counters["written_rows"] % 100_000 == 0:
                    elapsed = max(time.monotonic() - started, 0.001)
                    print(
                        f"    progress {counters['written_rows']:,} rows "
                        f"({counters['written_rows'] / elapsed:,.0f} rows/s)",
                        flush=True,
                    )
        if batch:
            insert_daily_batch(conn, batch)
            counters["written_rows"] += len(batch)
        conn.commit()
    finally:
        zf.close()
    elapsed = time.monotonic() - started
    return {
        **counters,
        "written_rows": int(counters.get("written_rows", 0)),
        "min_date": min_date,
        "max_date": max_date,
        "trading_dates": len(dates),
        "elapsed_seconds": round(elapsed, 2),
    }


def copy_financial_factors(conn, legacy_path):
    if legacy_path is None or not legacy_path.exists():
        return 0
    conn.execute("ATTACH DATABASE ? AS legacy", (str(legacy_path),))
    try:
        found = conn.execute(
            "SELECT 1 FROM legacy.sqlite_master WHERE type='table' AND name='financial_factors'"
        ).fetchone()
        if not found:
            return 0
        conn.execute("DROP TABLE IF EXISTS main.financial_factors")
        conn.execute("CREATE TABLE main.financial_factors AS SELECT * FROM legacy.financial_factors")
        count = conn.execute("SELECT COUNT(*) FROM main.financial_factors").fetchone()[0]
        conn.commit()
        return int(count)
    finally:
        conn.execute("DETACH DATABASE legacy")


def finalize_database(conn):
    conn.executescript(
        """
        CREATE INDEX idx_stock_daily_trade_date ON stock_daily(trade_date);
        CREATE INDEX idx_stock_daily_industry_date ON stock_daily(industry_1, trade_date);

        CREATE TABLE stock_meta AS
        SELECT d.code, d.name, d.trade_date AS latest_trade_date,
               d.close AS latest_close, d.listed_state, d.currency,
               d.industry_1, d.industry_2
        FROM stock_daily d
        JOIN (
            SELECT code, MAX(trade_date) AS max_date
            FROM stock_daily
            GROUP BY code
        ) latest
          ON d.code = latest.code AND d.trade_date = latest.max_date;
        CREATE UNIQUE INDEX idx_stock_meta_code ON stock_meta(code);

        CREATE VIEW markowitz_returns AS
        SELECT code, name, trade_date, daily_return, close, adj_close_1, adj_close_2,
               amount, volume, industry_1, industry_2
        FROM stock_daily
        WHERE daily_return IS NOT NULL;

        CREATE VIEW latest_prices AS
        SELECT d.*
        FROM stock_daily d
        JOIN (
            SELECT code, MAX(trade_date) AS latest_trade_date
            FROM stock_daily
            GROUP BY code
        ) latest
          ON d.code = latest.code AND d.trade_date = latest.latest_trade_date;

        ANALYZE;
        PRAGMA optimize;
        """
    )
    conn.commit()


def validate_database(conn, expected_start, expected_end):
    report = {}
    report["database"] = dict(
        zip(
            ("min_date", "max_date", "rows", "stocks", "trading_dates"),
            conn.execute(
                "SELECT MIN(trade_date), MAX(trade_date), COUNT(*), "
                "COUNT(DISTINCT code), COUNT(DISTINCT trade_date) FROM stock_daily"
            ).fetchone(),
        )
    )
    report["rows_by_year"] = {
        year: count
        for year, count in conn.execute(
            "SELECT SUBSTR(trade_date,1,4), COUNT(*) FROM stock_daily GROUP BY 1 ORDER BY 1"
        )
    }
    report["quality"] = dict(
        zip(
            (
                "missing_close",
                "missing_return",
                "invalid_ohlc",
                "unknown_industry_rows",
                "unknown_industry_stocks",
            ),
            conn.execute(
                """
                SELECT
                    SUM(close IS NULL OR close <= 0),
                    SUM(daily_return IS NULL),
                    SUM(
                        open IS NOT NULL AND high IS NOT NULL AND low IS NOT NULL
                        AND (high < open OR high < close OR low > open OR low > close)
                    ),
                    SUM(industry_1 IS NULL OR industry_1 = 'UNKNOWN'),
                    COUNT(DISTINCT CASE WHEN industry_1 IS NULL OR industry_1 = 'UNKNOWN' THEN code END)
                FROM stock_daily
                """
            ).fetchone(),
        )
    )
    report["midea_latest"] = conn.execute(
        """
        SELECT trade_date, close, raw_close, daily_return, capital_return, source_file
        FROM stock_daily WHERE code='000333' ORDER BY trade_date DESC LIMIT 1
        """
    ).fetchone()
    report["auxiliary_tables"] = {
        "companies": conn.execute("SELECT COUNT(*) FROM csmar_company").fetchone()[0],
        "adjust_factors": conn.execute("SELECT COUNT(*) FROM csmar_adjust_factor").fetchone()[0],
        "no_limit_days": conn.execute("SELECT COUNT(*) FROM csmar_no_limit").fetchone()[0],
        "financial_factors": conn.execute(
            "SELECT COUNT(*) FROM financial_factors"
        ).fetchone()[0]
        if conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='financial_factors'"
        ).fetchone()
        else 0,
    }
    min_date = report["database"]["min_date"]
    max_date = report["database"]["max_date"]
    if min_date is None or min_date[:4] != expected_start[:4]:
        raise ValueError(f"Database begins at {min_date}; expected coverage from {expected_start}")
    if max_date != expected_end:
        raise ValueError(f"Database ends at {max_date}; expected {expected_end}")
    if report["quality"]["missing_close"] or report["quality"]["missing_return"]:
        raise ValueError(f"Required-value validation failed: {report['quality']}")
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument(
        "--industry-file",
        type=Path,
        help=(
            "Explicit full-history STK_INDUSTRYCLASS workbook. When supplied, "
            "it replaces the industry workbook discovered below --source-dir."
        ),
    )
    parser.add_argument("--legacy-database", type=Path)
    parser.add_argument("--start-date", default="2019-01-01")
    parser.add_argument("--end-date", default="2026-07-17")
    parser.add_argument("--batch-size", type=int, default=20_000)
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--report", type=Path)
    return parser.parse_args(argv)


def run(args):
    source_dir = args.source_dir.resolve()
    database = args.database.resolve()
    legacy_database = args.legacy_database.resolve() if args.legacy_database else None
    report_path = (
        args.report.resolve()
        if args.report
        else database.with_name(database.stem + "_build_report.json")
    )
    build_path = database.with_suffix(database.suffix + ".building")

    daily_files = sorted(source_dir.rglob("TRD_Dalyr*.xlsx"), key=natural_key)
    company_files = find_alias_files(source_dir, FILE_ALIASES["company"])
    adjust_files = find_alias_files(source_dir, FILE_ALIASES["adjust_factor"])
    no_limit_files = find_alias_files(source_dir, FILE_ALIASES["no_limit"])
    if args.industry_file is not None:
        industry_path = args.industry_file.resolve()
        if not industry_path.is_file():
            raise FileNotFoundError(
                f"Explicit industry-history workbook does not exist: {industry_path}"
            )
        industry_files = [industry_path]
    else:
        industry_files = find_alias_files(source_dir, FILE_ALIASES["industry"])
    if (
        not daily_files
        or len(company_files) != 1
        or len(adjust_files) != 1
        or len(no_limit_files) != 1
        or len(industry_files) != 1
    ):
        raise ValueError(
            "Expected daily files plus exactly one company, adjustment-factor, "
            "no-limit and industry-history workbook."
        )
    if database.exists() and not args.reset:
        raise FileExistsError(f"Target database already exists: {database}")
    if build_path.exists() and args.reset:
        build_path.unlink()
    resume_partial_build = build_path.exists()

    database.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    legacy = load_legacy_metadata(legacy_database)
    companies = load_company_file(company_files[0], legacy)
    industry_history, industry_history_rows = load_industry_history(industry_files[0])
    metadata = dict(legacy)
    for code, company in companies.items():
        current = metadata.setdefault(code, {})
        current.setdefault("name", company["name"])
        if not current.get("name") or current.get("name") == code:
            current["name"] = company["name"]
        if company.get("industry_1"):
            current["industry_1"] = company["industry_1"]
        if company.get("industry_2"):
            current["industry_2"] = company["industry_2"]

    print("CSMAR full database build", flush=True)
    print(f"  source: {source_dir}", flush=True)
    print(f"  target: {database}", flush=True)
    print(f"  daily files: {len(daily_files)}", flush=True)
    print(f"  legacy industry mappings: {len(legacy):,}", flush=True)
    print(f"  CSMAR company rows: {len(companies):,}", flush=True)
    print(f"  causal CSRC industry rows: {industry_history_rows:,}", flush=True)
    print(f"  resume partial build: {resume_partial_build}", flush=True)

    build_started = time.monotonic()
    conn = sqlite3.connect(build_path)
    build_report = {
        "source_dir": str(source_dir),
        "database": str(database),
        "start_date": args.start_date,
        "end_date": args.end_date,
        "industry_source": str(industry_files[0]),
        "industry_policy": (
            f"{CSRC_2001_CLASSIFICATION} before {CSRC_2012_EFFECTIVE_DATE}; "
            f"{CSRC_2012_CLASSIFICATION} from {CSRC_2012_EFFECTIVE_DATE}; "
            f"dated fallback={ASSOCIATION_CLASSIFICATION}"
        ),
        "daily_files": [],
    }
    try:
        if resume_partial_build:
            ensure_resume_schema(conn)
            adjust_counts = {
                "written_rows": conn.execute(
                    "SELECT COUNT(*) FROM csmar_adjust_factor"
                ).fetchone()[0]
            }
            no_limit_counts = {
                "written_rows": conn.execute(
                    "SELECT COUNT(*) FROM csmar_no_limit"
                ).fetchone()[0]
            }
            print("  continuing the existing partial database", flush=True)
        else:
            create_schema(conn)
            insert_company_rows(conn, companies)
            adjust_counts = import_adjust_factors(
                conn, adjust_files[0], args.start_date, args.end_date
            )
            no_limit_counts = import_no_limit(
                conn, no_limit_files[0], args.start_date, args.end_date
            )
        no_limit_keys = set(
            conn.execute("SELECT code, trade_date FROM csmar_no_limit").fetchall()
        )
        print(f"  adjustment factors: {adjust_counts['written_rows']:,}", flush=True)
        print(f"  no-limit rows: {no_limit_counts['written_rows']:,}", flush=True)

        total_written = int(
            conn.execute("SELECT COUNT(*) FROM stock_daily").fetchone()[0]
        )
        for index, path in enumerate(daily_files, start=1):
            relative_path = str(path.relative_to(source_dir))
            completed = conn.execute(
                """
                SELECT written_rows
                FROM csmar_daily_import_state
                WHERE source_file=? AND status='complete'
                """,
                (relative_path,),
            ).fetchone()
            if completed is not None:
                print(
                    f"[{index}/{len(daily_files)}] [skip complete] {relative_path} "
                    f"({int(completed[0]):,} rows)",
                    flush=True,
                )
                build_report["daily_files"].append(
                    {
                        "file": relative_path,
                        "written_rows": int(completed[0]),
                        "resumed_skip": True,
                    }
                )
                continue
            print(f"[{index}/{len(daily_files)}] {relative_path}", flush=True)
            file_report = import_daily_file(
                conn,
                path,
                source_dir,
                args.start_date,
                args.end_date,
                metadata,
                args.batch_size,
                industry_history,
                no_limit_keys,
            )
            written_rows = int(file_report.get("written_rows", 0))
            total_written += written_rows
            mark_daily_file_complete(conn, relative_path, written_rows)
            file_report["file"] = relative_path
            build_report["daily_files"].append(file_report)
            print(
                f"    done {file_report['written_rows']:,} rows, "
                f"{file_report['min_date']} to {file_report['max_date']}, "
                f"{file_report['elapsed_seconds']:.1f}s; total {total_written:,}",
                flush=True,
            )

        financial_count = copy_financial_factors(conn, legacy_database)
        print(f"  copied financial factors: {financial_count:,}", flush=True)
        print("  creating indexes and stock metadata...", flush=True)
        finalize_database(conn)
        validation = validate_database(conn, args.start_date, args.end_date)
        build_report["validation"] = validation
        build_report["elapsed_seconds"] = round(time.monotonic() - build_started, 2)
        build_report["completed_at"] = now_iso()
        conn.execute(
            "INSERT OR REPLACE INTO project_metadata VALUES (?,?)",
            ("build_report", json.dumps(build_report, ensure_ascii=False)),
        )
        conn.execute(
            "INSERT OR REPLACE INTO project_metadata VALUES (?,?)",
            ("industry_source", str(industry_files[0])),
        )
        conn.commit()
    finally:
        conn.close()

    if database.exists():
        database.unlink()
    os.replace(build_path, database)
    report_path.write_text(
        json.dumps(build_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("Build complete", flush=True)
    print(json.dumps(build_report["validation"], ensure_ascii=False, indent=2), flush=True)
    print(f"Report: {report_path}", flush=True)


if __name__ == "__main__":
    run(parse_args())
