"""
Append CSMAR TRD_FwardQuotation.xlsx data to an existing stock-history
SQLite database for out-of-sample research.

Design choices
--------------
- The source file is CSMAR forward-adjusted daily quotation data.
- History on or before the cutoff remains untouched.
- Each incremental workbook is anchored to the latest database close before
  that workbook's first importable date. This keeps weekly files on the same
  continuous price chain instead of restarting from the original cutoff.
- CSMAR prices are re-scaled per stock and chained with ChangeRatio. This
  avoids artificial price jumps at both the provider boundary and week-to-week
  updates.
- StateCode == 2 rows are market holidays in this export and are skipped.
- Only codes with an earlier database close are appended. Newly listed codes
  cannot yet meet the strategy's history requirement and lack historical
  industry metadata in this export.
- The source file does not contain fresh stock names, industry codes or ST
  flags. These fields are carried forward from the last RESSET record.
"""

from __future__ import annotations

import argparse
import re
import sqlite3
import sys
import zipfile
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from xml.etree.ElementTree import iterparse


REL_NS = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
REQUIRED_HEADERS = {
    "TradingDate",
    "Symbol",
    "OpenPrice",
    "ClosePrice",
    "HighPrice",
    "LowPrice",
    "Volume",
    "Amount",
    "StateCode",
    "ChangeRatio",
    "TurnoverRate1",
}


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def column_index(cell_ref: str) -> int:
    letters = "".join(ch for ch in cell_ref if ch.isalpha()).upper()
    result = 0
    for ch in letters:
        result = result * 26 + ord(ch) - ord("A") + 1
    return result - 1


def is_blank(value) -> bool:
    return value is None or str(value).strip() == ""


def parse_float(value):
    if is_blank(value):
        return None
    try:
        return float(str(value).replace(",", "").strip())
    except ValueError:
        return None


def parse_code(value):
    if is_blank(value):
        return None
    text = re.sub(r"\D", "", str(value).strip())
    return text.zfill(6) if text else None


def parse_date(value):
    if is_blank(value):
        return None

    text = str(value).strip()
    if re.fullmatch(r"\d+(\.\d+)?", text):
        serial = float(text)
        if 20_000 <= serial <= 70_000:
            return (datetime(1899, 12, 30) + timedelta(days=int(serial))).date().isoformat()

    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y%m%d"):
        candidate = text[:10] if fmt != "%Y%m%d" else text[:8]
        try:
            return datetime.strptime(candidate, fmt).date().isoformat()
        except ValueError:
            pass
    return None


def load_shared_strings(zf):
    if "xl/sharedStrings.xml" not in zf.namelist():
        return []

    values = []
    with zf.open("xl/sharedStrings.xml") as handle:
        parts = []
        inside = False
        for event, elem in iterparse(handle, events=("start", "end")):
            tag = local_name(elem.tag)
            if event == "start" and tag == "si":
                parts = []
                inside = True
            elif event == "end" and inside and tag == "t":
                parts.append(elem.text or "")
            elif event == "end" and tag == "si":
                values.append("".join(parts))
                inside = False
            elem.clear()
    return values


def workbook_sheets(zf):
    relationships = {}
    with zf.open("xl/_rels/workbook.xml.rels") as handle:
        for _, elem in iterparse(handle, events=("end",)):
            if local_name(elem.tag) == "Relationship":
                relationships[elem.attrib["Id"]] = elem.attrib["Target"]
            elem.clear()

    sheets = []
    with zf.open("xl/workbook.xml") as handle:
        for _, elem in iterparse(handle, events=("end",)):
            if local_name(elem.tag) == "sheet":
                rel_id = elem.attrib.get(REL_NS + "id")
                target = relationships.get(rel_id)
                if target and not target.startswith("xl/"):
                    target = "xl/" + target.lstrip("/")
                sheets.append((elem.attrib.get("name", ""), target))
            elem.clear()
    return sheets


def cell_value(cell, shared_strings):
    cell_type = cell.attrib.get("t")

    if cell_type == "inlineStr":
        return "".join(
            child.text or ""
            for child in cell.iter()
            if local_name(child.tag) == "t"
        )

    raw = None
    for child in cell:
        if local_name(child.tag) == "v":
            raw = child.text
            break

    if raw is None:
        return None

    if cell_type == "s":
        try:
            return shared_strings[int(raw)]
        except (ValueError, IndexError):
            return raw

    return raw


def iter_sheet_rows(zf, sheet_path, shared_strings):
    with zf.open(sheet_path) as handle:
        current = {}
        for event, elem in iterparse(handle, events=("start", "end")):
            tag = local_name(elem.tag)

            if event == "start" and tag == "row":
                current = {}

            elif event == "end" and tag == "c":
                ref = elem.attrib.get("r", "")
                if ref:
                    current[column_index(ref)] = cell_value(elem, shared_strings)
                elem.clear()

            elif event == "end" and tag == "row":
                if current:
                    width = max(current) + 1
                    yield [current.get(i) for i in range(width)]
                else:
                    yield []
                elem.clear()


def header_map(header):
    mapping = {
        str(value).strip(): idx
        for idx, value in enumerate(header)
        if not is_blank(value)
    }
    missing = sorted(REQUIRED_HEADERS - set(mapping))
    if missing:
        raise ValueError(
            "This does not look like the expected CSMAR TRD_FwardQuotation export. "
            f"Missing headers: {', '.join(missing)}"
        )
    return mapping


def latest_database_meta_before(conn, source_start_date):
    rows = conn.execute(
        """
        SELECT d.code, d.name, d.close, d.listed_state, d.currency,
               d.industry_1, d.industry_2, d.trade_date
        FROM stock_daily d
        JOIN (
            SELECT code, MAX(trade_date) AS latest_trade_date
            FROM stock_daily
            WHERE trade_date < ?
            GROUP BY code
        ) x
          ON d.code = x.code
         AND d.trade_date = x.latest_trade_date
        """,
        (source_start_date,),
    ).fetchall()

    meta = {}
    for code, name, close, listed_state, currency, industry_1, industry_2, trade_date in rows:
        close_value = parse_float(close)
        if close_value is None or close_value <= 0:
            continue
        meta[str(code).zfill(6)] = {
            "name": name,
            "last_close": close_value,
            "listed_state": listed_state or "Norm",
            "currency": currency or "CNY",
            "industry_1": industry_1,
            "industry_2": industry_2,
            "anchor_date": trade_date,
        }
    return meta


def now_iso():
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def ensure_tables_exist(conn):
    found = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='stock_daily'"
    ).fetchone()
    if not found:
        raise RuntimeError(
            "The target database has no stock_daily table. "
            "Use a copy of your existing RESSET database, not a blank file."
        )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS source_file_state (
            source_file TEXT PRIMARY KEY,
            file_size INTEGER NOT NULL,
            file_mtime_ns INTEGER NOT NULL,
            imported_at TEXT NOT NULL
        )
        """
    )
    conn.commit()


def source_file_is_unchanged(conn, path):
    stat = path.stat()
    row = conn.execute(
        "SELECT file_size, file_mtime_ns FROM source_file_state WHERE source_file = ?",
        (path.name,),
    ).fetchone()
    return bool(
        row
        and int(row[0]) == int(stat.st_size)
        and int(row[1]) == int(stat.st_mtime_ns)
    )


def record_source_file_state(conn, path, imported_at):
    stat = path.stat()
    conn.execute(
        """
        INSERT INTO source_file_state (source_file, file_size, file_mtime_ns, imported_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(source_file) DO UPDATE SET
            file_size=excluded.file_size,
            file_mtime_ns=excluded.file_mtime_ns,
            imported_at=excluded.imported_at
        """,
        (path.name, int(stat.st_size), int(stat.st_mtime_ns), imported_at),
    )


def discover_files(source_dir, years):
    files = [
        path
        for path in source_dir.glob("*.xlsx")
        if not path.name.startswith("~$")
    ]
    if years:
        wanted = {str(year) for year in years}
        files = [path for path in files if path.stem.split("_", 1)[0] in wanted]
    return sorted(files, key=lambda path: path.name)


def source_date_range(zf, sheet_path, shared_strings, columns, cutoff_date):
    rows = iter_sheet_rows(zf, sheet_path, shared_strings)
    next(rows, None)
    source_min = None
    source_max = None
    importable_min = None
    for raw in rows:
        idx = columns["TradingDate"]
        trade_date = parse_date(raw[idx] if idx < len(raw) else None)
        if not trade_date:
            continue
        source_min = min(source_min, trade_date) if source_min else trade_date
        source_max = max(source_max, trade_date) if source_max else trade_date
        if trade_date > cutoff_date:
            importable_min = min(importable_min, trade_date) if importable_min else trade_date
    return source_min, source_max, importable_min


def existing_rows_in_range(conn, start_date, end_date):
    rows = conn.execute(
        """
        SELECT code, trade_date, close, daily_return
        FROM stock_daily
        WHERE trade_date BETWEEN ? AND ?
        """,
        (start_date, end_date),
    ).fetchall()
    return {
        (str(code).zfill(6), trade_date): (parse_float(close), parse_float(daily_return))
        for code, trade_date, close, daily_return in rows
    }


def insert_records(conn, records, imported_at):
    if not records:
        return

    sql = """
        INSERT INTO stock_daily (
            code, name, trade_date, prev_close, open, high, low, close,
            adj_close_1, adj_close_2, volume, amount, turnover_total,
            turnover_float, adj_factor, daily_return, capital_return,
            risk_free_return, listed_state, currency, industry_1, industry_2,
            source_file, source_sheet, imported_at
        )
        VALUES (
            :code, :name, :trade_date, :prev_close, :open, :high, :low, :close,
            :adj_close_1, :adj_close_2, :volume, :amount, :turnover_total,
            :turnover_float, :adj_factor, :daily_return, :capital_return,
            :risk_free_return, :listed_state, :currency, :industry_1, :industry_2,
            :source_file, :source_sheet, :imported_at
        )
        ON CONFLICT(code, trade_date) DO UPDATE SET
            name=excluded.name,
            prev_close=excluded.prev_close,
            open=excluded.open,
            high=excluded.high,
            low=excluded.low,
            close=excluded.close,
            adj_close_1=excluded.adj_close_1,
            adj_close_2=excluded.adj_close_2,
            volume=excluded.volume,
            amount=excluded.amount,
            turnover_total=excluded.turnover_total,
            turnover_float=excluded.turnover_float,
            adj_factor=excluded.adj_factor,
            daily_return=excluded.daily_return,
            capital_return=excluded.capital_return,
            risk_free_return=excluded.risk_free_return,
            listed_state=excluded.listed_state,
            currency=excluded.currency,
            industry_1=excluded.industry_1,
            industry_2=excluded.industry_2,
            source_file=excluded.source_file,
            source_sheet=excluded.source_sheet,
            imported_at=excluded.imported_at
    """
    conn.executemany(sql, records)

    meta_sql = """
        INSERT INTO stock_meta (
            code, name, latest_trade_date, latest_close,
            listed_state, currency, industry_1, industry_2
        )
        VALUES (
            :code, :name, :trade_date, :close,
            :listed_state, :currency, :industry_1, :industry_2
        )
        ON CONFLICT(code) DO UPDATE SET
            name=CASE
                WHEN excluded.latest_trade_date >= stock_meta.latest_trade_date
                THEN excluded.name ELSE stock_meta.name END,
            latest_trade_date=MAX(stock_meta.latest_trade_date, excluded.latest_trade_date),
            latest_close=CASE
                WHEN excluded.latest_trade_date >= stock_meta.latest_trade_date
                THEN excluded.latest_close ELSE stock_meta.latest_close END,
            listed_state=CASE
                WHEN excluded.latest_trade_date >= stock_meta.latest_trade_date
                THEN excluded.listed_state ELSE stock_meta.listed_state END,
            currency=CASE
                WHEN excluded.latest_trade_date >= stock_meta.latest_trade_date
                THEN excluded.currency ELSE stock_meta.currency END,
            industry_1=CASE
                WHEN excluded.latest_trade_date >= stock_meta.latest_trade_date
                THEN excluded.industry_1 ELSE stock_meta.industry_1 END,
            industry_2=CASE
                WHEN excluded.latest_trade_date >= stock_meta.latest_trade_date
                THEN excluded.industry_2 ELSE stock_meta.industry_2 END
    """
    conn.executemany(meta_sql, records)


def process_file(conn, path, args):
    with zipfile.ZipFile(path) as zf:
        shared = load_shared_strings(zf)
        sheets = workbook_sheets(zf)
        if len(sheets) != 1:
            raise ValueError(
                "Expected one worksheet in the forward quotation export; found: "
                + ", ".join(name for name, _ in sheets)
            )

        sheet_name, sheet_path = sheets[0]
        rows = iter_sheet_rows(zf, sheet_path, shared)
        header = next(rows, None)
        if header is None:
            raise ValueError("The source workbook is empty.")

        columns = header_map(header)
        source_min, source_max, importable_min = source_date_range(
            zf,
            sheet_path,
            shared,
            columns,
            args.cutoff_date,
        )

        print(f"\nWorkbook: {path.name}")
        print(f"Sheet: {sheet_name}")
        print(f"Recognized columns: {', '.join(columns)}")
        print(f"Mode: {'APPLY (will write)' if args.apply else 'INSPECT ONLY (no changes)'}")
        print("Source coverage")
        print(f"  source_min_date: {source_min}")
        print(f"  source_max_date: {source_max}")
        print(f"  first_date_after_cutoff: {importable_min}")

        if importable_min is None:
            print("  No rows are later than the configured cutoff; nothing to import.")
            return Counter()

        existing = latest_database_meta_before(conn, importable_min)
        if not existing:
            raise ValueError(
                f"No valid database closes found before {importable_min}."
            )

        anchor_dates = [item["anchor_date"] for item in existing.values()]
        database_max = conn.execute(
            "SELECT MAX(trade_date) FROM stock_daily"
        ).fetchone()[0]
        print("Incremental anchor")
        print(f"  database_max_date_before_import: {database_max}")
        print(f"  anchor_rule: latest close strictly before {importable_min}")
        print(f"  codes_with_usable_anchor: {len(existing):,}")
        print(f"  anchor_date_range: {min(anchor_dates)} to {max(anchor_dates)}")

        overlap = existing_rows_in_range(conn, importable_min, source_max)
        imported_at = now_iso()
        counters = Counter()
        seen_dates = set()
        records = []
        last_close = {
            code: item["last_close"]
            for code, item in existing.items()
        }
        last_source_date = {}

        rows = iter_sheet_rows(zf, sheet_path, shared)
        next(rows, None)
        for raw in rows:
            counters["source_rows"] += 1

            date_idx = columns["TradingDate"]
            trade_date = parse_date(raw[date_idx] if date_idx < len(raw) else None)
            if not trade_date:
                counters["non_data_rows"] += 1
                continue
            if trade_date <= args.cutoff_date:
                counters["at_or_before_cutoff"] += 1
                continue

            code_idx = columns["Symbol"]
            code = parse_code(raw[code_idx] if code_idx < len(raw) else None)
            if not code:
                counters["bad_code"] += 1
                continue

            previous_source_date = last_source_date.get(code)
            if previous_source_date is not None and trade_date <= previous_source_date:
                raise ValueError(
                    f"Rows for code {code} are not strictly date-ascending: "
                    f"{previous_source_date} then {trade_date}."
                )
            last_source_date[code] = trade_date

            state_idx = columns["StateCode"]
            state = str(raw[state_idx] if state_idx < len(raw) else "").strip()
            if state == "2":
                counters["market_holiday_rows_skipped"] += 1
                continue

            if code not in existing:
                counters["unknown_or_new_code_skipped"] += 1
                continue

            close_idx = columns["ClosePrice"]
            return_idx = columns["ChangeRatio"]
            close_forward = parse_float(
                raw[close_idx] if close_idx < len(raw) else None
            )
            change_ratio = parse_float(
                raw[return_idx] if return_idx < len(raw) else None
            )
            if (
                close_forward is None
                or close_forward <= 0
                or change_ratio is None
                or change_ratio <= -0.99
            ):
                counters["missing_or_invalid_price"] += 1
                continue

            prior_close = last_close[code]
            synthetic_close = prior_close * (1.0 + change_ratio)
            scale = synthetic_close / close_forward

            def scaled_price(header_name):
                idx = columns[header_name]
                value = parse_float(raw[idx] if idx < len(raw) else None)
                return value * scale if value is not None and value > 0 else synthetic_close

            turnover_idx = columns["TurnoverRate1"]
            volume_idx = columns["Volume"]
            amount_idx = columns["Amount"]
            turnover = parse_float(raw[turnover_idx] if turnover_idx < len(raw) else None)
            volume = parse_float(raw[volume_idx] if volume_idx < len(raw) else None)
            amount = parse_float(raw[amount_idx] if amount_idx < len(raw) else None)

            meta = existing[code]
            record = {
                "code": code,
                "name": meta["name"],
                "trade_date": trade_date,
                "prev_close": prior_close,
                "open": scaled_price("OpenPrice"),
                "high": scaled_price("HighPrice"),
                "low": scaled_price("LowPrice"),
                "close": synthetic_close,
                "adj_close_1": synthetic_close,
                "adj_close_2": synthetic_close,
                "volume": volume,
                "amount": amount,
                "turnover_total": turnover,
                "turnover_float": turnover,
                "adj_factor": scale,
                "daily_return": change_ratio,
                "capital_return": change_ratio,
                "risk_free_return": None,
                "listed_state": meta["listed_state"],
                "currency": meta["currency"],
                "industry_1": meta["industry_1"],
                "industry_2": meta["industry_2"],
                "source_file": path.name,
                "source_sheet": sheet_name,
                "imported_at": imported_at,
            }

            old = overlap.get((code, trade_date))
            if old is not None:
                counters["overlap_rows"] += 1
                old_close, old_return = old
                close_tolerance = max(abs(old_close or 0.0), 1.0) * 1e-8
                if old_close is None or abs(synthetic_close - old_close) > close_tolerance:
                    counters["overlap_close_mismatch"] += 1
                if old_return is not None and abs(change_ratio - old_return) > 1e-10:
                    counters["overlap_return_mismatch"] += 1

            last_close[code] = synthetic_close
            seen_dates.add(trade_date)
            counters["eligible_rows"] += 1
            records.append(record)

            if args.apply and len(records) >= args.batch_size:
                insert_records(conn, records, imported_at)
                counters["written_rows"] += len(records)
                records.clear()

        if args.apply:
            insert_records(conn, records, imported_at)
            counters["written_rows"] += len(records)
            record_source_file_state(conn, path, imported_at)
            conn.commit()

        print("Import counters")
        for key in [
            "source_rows",
            "non_data_rows",
            "at_or_before_cutoff",
            "market_holiday_rows_skipped",
            "unknown_or_new_code_skipped",
            "bad_code",
            "missing_or_invalid_price",
            "eligible_rows",
            "overlap_rows",
            "overlap_close_mismatch",
            "overlap_return_mismatch",
            "written_rows",
        ]:
            print(f"  {key}: {counters[key]:,}")
        if seen_dates:
            print(f"  first_insertable_date: {min(seen_dates)}")
            print(f"  last_insertable_date: {max(seen_dates)}")
        return counters


def parse_args():
    parser = argparse.ArgumentParser(
        description="Append weekly forward-adjusted daily quotes to an existing SQLite database."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--source-xlsx",
        type=Path,
        help="One TRD_FwardQuotation.xlsx file.",
    )
    source.add_argument(
        "--source-dir",
        type=Path,
        help="Directory containing weekly TRD_FwardQuotation .xlsx files.",
    )
    parser.add_argument(
        "--database",
        required=True,
        type=Path,
        help="Existing stock-history SQLite database.",
    )
    parser.add_argument(
        "--cutoff-date",
        default="2026-03-31",
        help="Only dates strictly after this date are appended.",
    )
    parser.add_argument(
        "--years",
        nargs="*",
        help="Optional filename-year filter used with --source-dir.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually write to the target database. Without this flag the script only inspects.",
    )
    parser.add_argument(
        "--force-reimport",
        action="store_true",
        help="Re-read unchanged files when using --apply.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10_000,
        help="SQLite upsert batch size.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.database.exists():
        raise SystemExit(
            f"Target database does not exist: {args.database}"
        )

    if args.source_xlsx:
        if not args.source_xlsx.exists():
            raise SystemExit(f"Source xlsx does not exist: {args.source_xlsx}")
        files = [args.source_xlsx]
    else:
        if not args.source_dir.exists():
            raise SystemExit(f"Source directory does not exist: {args.source_dir}")
        files = discover_files(args.source_dir, args.years)
        if not files:
            raise SystemExit(f"No matching .xlsx files found in {args.source_dir}")

    print("Input files")
    for path in files:
        print(f"  {path.name} ({path.stat().st_size / 1024 / 1024:.1f} MB)")

    conn = sqlite3.connect(args.database)
    try:
        ensure_tables_exist(conn)
        totals = Counter()
        for path in files:
            if (
                args.apply
                and not args.force_reimport
                and source_file_is_unchanged(conn, path)
            ):
                print(f"Skipping unchanged file: {path.name}")
                continue
            try:
                totals.update(process_file(conn, path, args))
            except Exception:
                conn.rollback()
                raise

        result = conn.execute(
            """
            SELECT MIN(trade_date), MAX(trade_date),
                   COUNT(DISTINCT trade_date), COUNT(*)
            FROM stock_daily
            """
        ).fetchone()
        print("\nTarget database range")
        print(f"  min / max / trading_dates / rows: {result}")
        print("\nAll-file totals")
        for key in [
            "eligible_rows",
            "overlap_rows",
            "overlap_close_mismatch",
            "overlap_return_mismatch",
            "written_rows",
        ]:
            print(f"  {key}: {totals[key]:,}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
