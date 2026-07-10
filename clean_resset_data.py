import argparse
import csv
import re
import sqlite3
import sys
import zipfile
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from xml.etree.ElementTree import iterparse


REL_NS = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"

OUTPUT_COLUMNS = [
    "code",
    "name",
    "trade_date",
    "prev_close",
    "open",
    "high",
    "low",
    "close",
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
    "listed_state",
    "currency",
    "industry_1",
    "industry_2",
    "source_file",
    "source_sheet",
]

FIELD_BY_SUFFIX = {
    "Stkcd": "code",
    "Lstknm": "name",
    "Date": "trade_date",
    "PrevClPr": "prev_close",
    "Oppr": "open",
    "Hipr": "high",
    "Lopr": "low",
    "Clpr": "close",
    "AdjClpr1": "adj_close_1",
    "AdjClpr2": "adj_close_2",
    "Trdvol": "volume",
    "Trdsum": "amount",
    "DFulTurnR": "turnover_total",
    "DTrdTurnR": "turnover_float",
    "Mcfacpr": "adj_factor",
    "Dret": "daily_return",
    "Daret": "capital_return",
    "DRfRet": "risk_free_return",
    "Listedstate": "listed_state",
    "Qttncurrency": "currency",
    "Csrciccd1": "industry_1",
    "Csrciccd2": "industry_2",
}

NUMERIC_FIELDS = {
    "prev_close",
    "open",
    "high",
    "low",
    "close",
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
}

ESSENTIAL_FIELDS = ["code", "trade_date", "close", "daily_return", "currency", "listed_state"]


def local_name(tag):
    if "}" in tag:
        return tag.rsplit("}", 1)[1]
    return tag


def column_index(cell_ref):
    letters = ""
    for char in cell_ref:
        if char.isalpha():
            letters += char.upper()
        else:
            break

    value = 0
    for char in letters:
        value = value * 26 + ord(char) - ord("A") + 1
    return value - 1


def blank(value):
    return value is None or str(value).strip() == ""


def normalize_suffix(header):
    text = str(header or "").strip()
    if "_" in text:
        return text.rsplit("_", 1)[-1]
    return re.sub(r"[^0-9a-zA-Z]+", "", text)


def parse_code(value):
    if blank(value):
        return None
    text = str(value).strip()
    if re.fullmatch(r"\d+\.0", text):
        text = str(int(float(text)))
    text = re.sub(r"\D", "", text)
    if not text:
        return None
    return text.zfill(6)


def parse_float(value):
    if blank(value):
        return None
    text = str(value).strip().replace(",", "")
    try:
        return float(text)
    except ValueError:
        return None


def parse_date(value):
    if blank(value):
        return None

    text = str(value).strip()
    if re.fullmatch(r"\d+(\.\d+)?", text):
        number = float(text)
        if 20000 <= number <= 70000:
            return (datetime(1899, 12, 30) + timedelta(days=int(number))).date().isoformat()

    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y%m%d"):
        candidate = text[:10] if fmt != "%Y%m%d" else text[:8]
        try:
            return datetime.strptime(candidate, fmt).date().isoformat()
        except ValueError:
            pass
    return None


def clean_text(value):
    if blank(value):
        return None
    return str(value).strip()


def load_shared_strings(zf):
    if "xl/sharedStrings.xml" not in zf.namelist():
        return []

    values = []
    with zf.open("xl/sharedStrings.xml") as handle:
        text_parts = []
        in_si = False
        for event, elem in iterparse(handle, events=("start", "end")):
            name = local_name(elem.tag)
            if event == "start" and name == "si":
                in_si = True
                text_parts = []
            elif event == "end" and in_si and name == "t":
                text_parts.append(elem.text or "")
            elif event == "end" and name == "si":
                values.append("".join(text_parts))
                in_si = False
                elem.clear()
            elif event == "end":
                elem.clear()
    return values


def workbook_sheets(zf):
    rels = {}
    with zf.open("xl/_rels/workbook.xml.rels") as handle:
        for _, elem in iterparse(handle, events=("end",)):
            if local_name(elem.tag) == "Relationship":
                rels[elem.attrib["Id"]] = elem.attrib["Target"]
            elem.clear()

    sheets = []
    with zf.open("xl/workbook.xml") as handle:
        for _, elem in iterparse(handle, events=("end",)):
            if local_name(elem.tag) == "sheet":
                rel_id = elem.attrib.get(REL_NS + "id")
                target = rels.get(rel_id)
                if target:
                    if target.startswith("/"):
                        target = target.lstrip("/")
                    elif not target.startswith("xl/"):
                        target = "xl/" + target
                sheets.append({"name": elem.attrib.get("name", ""), "path": target})
            elem.clear()
    return sheets


def cell_value(cell, shared_strings):
    cell_type = cell.attrib.get("t")

    if cell_type == "inlineStr":
        parts = []
        for child in cell.iter():
            if local_name(child.tag) == "t" and child.text:
                parts.append(child.text)
        return "".join(parts)

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
        row_values = {}
        for event, elem in iterparse(handle, events=("start", "end")):
            name = local_name(elem.tag)
            if event == "start" and name == "row":
                row_values = {}
            elif event == "end" and name == "c":
                ref = elem.attrib.get("r", "")
                if ref:
                    row_values[column_index(ref)] = cell_value(elem, shared_strings)
                elem.clear()
            elif event == "end" and name == "row":
                if row_values:
                    max_col = max(row_values)
                    yield [row_values.get(i) for i in range(max_col + 1)]
                else:
                    yield []
                elem.clear()


def build_column_map(header):
    column_map = {}
    for idx, value in enumerate(header):
        suffix = normalize_suffix(value)
        field = FIELD_BY_SUFFIX.get(suffix)
        if field and field not in column_map:
            column_map[field] = idx

    missing = [field for field in ESSENTIAL_FIELDS if field not in column_map]
    if missing:
        raise ValueError(f"missing required columns: {', '.join(missing)}")
    return column_map


def clean_record(row, column_map, source_file, source_sheet):
    record = {column: None for column in OUTPUT_COLUMNS}
    record["source_file"] = source_file
    record["source_sheet"] = source_sheet

    for field, idx in column_map.items():
        value = row[idx] if idx < len(row) else None
        if field == "code":
            record[field] = parse_code(value)
        elif field == "trade_date":
            record[field] = parse_date(value)
        elif field in NUMERIC_FIELDS:
            record[field] = parse_float(value)
        else:
            record[field] = clean_text(value)

    return record


def is_a_share(record, include_non_cny):
    if include_non_cny:
        return True
    if record["currency"] != "CNY":
        return False
    code = record["code"] or ""
    return not (code.startswith("200") or code.startswith("900"))


def is_special_treatment(record):
    state = (record.get("listed_state") or "").strip().upper()
    if state != "NORM":
        return True

    name = (record.get("name") or "").strip().upper()
    return name.startswith("*ST") or name.startswith("ST") or name.startswith("SST")


def rejection_reason(record, include_st, include_non_cny):
    if any(record.get(field) is None for field in ESSENTIAL_FIELDS):
        return "missing_essential"
    if not is_a_share(record, include_non_cny):
        return "non_cny_or_b_share"
    if not include_st and is_special_treatment(record):
        return "not_normal_listing"
    if record["daily_return"] is None:
        return "missing_return"
    return None


def connect_database(path, reset=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    if reset and path.exists():
        path.unlink()

    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS stock_daily (
            code TEXT NOT NULL,
            name TEXT,
            trade_date TEXT NOT NULL,
            prev_close REAL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
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
            listed_state TEXT,
            currency TEXT,
            industry_1 TEXT,
            industry_2 TEXT,
            source_file TEXT,
            source_sheet TEXT,
            imported_at TEXT NOT NULL,
            PRIMARY KEY (code, trade_date)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS stock_meta (
            code TEXT PRIMARY KEY,
            name TEXT,
            latest_trade_date TEXT,
            latest_close REAL,
            listed_state TEXT,
            currency TEXT,
            industry_1 TEXT,
            industry_2 TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS import_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_file TEXT NOT NULL,
            source_sheet TEXT NOT NULL,
            started_at TEXT NOT NULL,
            finished_at TEXT NOT NULL,
            read_rows INTEGER NOT NULL,
            accepted_rows INTEGER NOT NULL,
            rejected_rows INTEGER NOT NULL,
            missing_essential INTEGER NOT NULL,
            non_cny_or_b_share INTEGER NOT NULL,
            not_normal_listing INTEGER NOT NULL
        )
        """
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
    conn.execute(
        """
        CREATE VIEW IF NOT EXISTS markowitz_returns AS
        SELECT code, name, trade_date, daily_return, close, adj_close_1, adj_close_2,
               amount, volume, industry_1, industry_2
        FROM stock_daily
        WHERE daily_return IS NOT NULL
        """
    )
    conn.execute(
        """
        CREATE VIEW IF NOT EXISTS latest_prices AS
        SELECT d.*
        FROM stock_daily d
        JOIN (
            SELECT code, MAX(trade_date) AS latest_trade_date
            FROM stock_daily
            GROUP BY code
        ) x
        ON d.code = x.code AND d.trade_date = x.latest_trade_date
        """
    )
    conn.commit()
    return conn


def insert_batch(conn, batch, imported_at):
    if not batch:
        return

    columns = OUTPUT_COLUMNS + ["imported_at"]
    placeholders = ", ".join(["?"] * len(columns))
    sql = f"""
        INSERT INTO stock_daily ({", ".join(columns)})
        VALUES ({placeholders})
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
    rows = [tuple(record[column] for column in OUTPUT_COLUMNS) + (imported_at,) for record in batch]
    conn.executemany(sql, rows)

    meta_rows = [
        (
            record["code"],
            record["name"],
            record["trade_date"],
            record["close"],
            record["listed_state"],
            record["currency"],
            record["industry_1"],
            record["industry_2"],
        )
        for record in batch
    ]
    conn.executemany(
        """
        INSERT INTO stock_meta (
            code, name, latest_trade_date, latest_close,
            listed_state, currency, industry_1, industry_2
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(code) DO UPDATE SET
            name=CASE
                WHEN excluded.latest_trade_date >= stock_meta.latest_trade_date THEN excluded.name
                ELSE stock_meta.name
            END,
            latest_trade_date=MAX(stock_meta.latest_trade_date, excluded.latest_trade_date),
            latest_close=CASE
                WHEN excluded.latest_trade_date >= stock_meta.latest_trade_date THEN excluded.latest_close
                ELSE stock_meta.latest_close
            END,
            listed_state=CASE
                WHEN excluded.latest_trade_date >= stock_meta.latest_trade_date THEN excluded.listed_state
                ELSE stock_meta.listed_state
            END,
            currency=CASE
                WHEN excluded.latest_trade_date >= stock_meta.latest_trade_date THEN excluded.currency
                ELSE stock_meta.currency
            END,
            industry_1=CASE
                WHEN excluded.latest_trade_date >= stock_meta.latest_trade_date THEN excluded.industry_1
                ELSE stock_meta.industry_1
            END,
            industry_2=CASE
                WHEN excluded.latest_trade_date >= stock_meta.latest_trade_date THEN excluded.industry_2
                ELSE stock_meta.industry_2
            END
        """,
        meta_rows,
    )


def discover_files(source_dir, years):
    files = [path for path in source_dir.glob("*.xlsx") if not path.name.startswith("~$")]
    if years:
        wanted = {str(year) for year in years}
        files = [path for path in files if path.stem.split("_", 1)[0] in wanted]

    def sort_key(path):
        match = re.match(r"(\d{4})(?:_(\d+))?", path.stem)
        if match:
            return (int(match.group(1)), int(match.group(2) or 0), path.name)
        return (9999, 9999, path.name)

    return sorted(files, key=sort_key)


def source_file_is_unchanged(conn, path):
    stat = path.stat()
    row = conn.execute(
        "SELECT file_size, file_mtime_ns FROM source_file_state WHERE source_file = ?",
        (path.name,),
    ).fetchone()
    return bool(row and int(row[0]) == int(stat.st_size) and int(row[1]) == int(stat.st_mtime_ns))


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
    conn.commit()


def process_sheet(conn, zf, path, sheet, shared_strings, args, imported_at):
    started_at = now_iso()
    counters = Counter()
    batch = []

    rows = iter_sheet_rows(zf, sheet["path"], shared_strings)
    header = None
    for row in rows:
        if any(not blank(value) for value in row):
            header = row
            break
    if header is None:
        return counters

    column_map = build_column_map(header)

    for row in rows:
        if not any(not blank(value) for value in row):
            continue

        counters["read_rows"] += 1
        record = clean_record(row, column_map, path.name, sheet["name"])
        reason = rejection_reason(record, args.include_st, args.include_non_cny)
        if reason:
            counters[reason] += 1
            counters["rejected_rows"] += 1
            continue

        batch.append(record)
        counters["accepted_rows"] += 1

        if args.max_rows and counters["read_rows"] >= args.max_rows:
            break

        if len(batch) >= args.batch_size:
            insert_batch(conn, batch, imported_at)
            conn.commit()
            batch.clear()

    insert_batch(conn, batch, imported_at)
    conn.commit()

    finished_at = now_iso()
    conn.execute(
        """
        INSERT INTO import_log (
            source_file, source_sheet, started_at, finished_at,
            read_rows, accepted_rows, rejected_rows,
            missing_essential, non_cny_or_b_share, not_normal_listing
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            path.name,
            sheet["name"],
            started_at,
            finished_at,
            counters["read_rows"],
            counters["accepted_rows"],
            counters["rejected_rows"],
            counters["missing_essential"],
            counters["non_cny_or_b_share"],
            counters["not_normal_listing"],
        ),
    )
    conn.commit()
    return counters


def process_file(conn, path, args, imported_at):
    print(f"Processing {path.name}", flush=True)
    totals = Counter()
    with zipfile.ZipFile(path) as zf:
        shared_strings = load_shared_strings(zf)
        for sheet in workbook_sheets(zf):
            sheet_totals = process_sheet(conn, zf, path, sheet, shared_strings, args, imported_at)
            totals.update(sheet_totals)
            print(
                "  "
                + f"{sheet['name']}: read={sheet_totals['read_rows']:,}, "
                + f"accepted={sheet_totals['accepted_rows']:,}, "
                + f"rejected={sheet_totals['rejected_rows']:,}",
                flush=True,
            )
    return totals


def export_csv(conn, csv_path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    cursor = conn.execute(
        """
        SELECT code, name, trade_date, close, adj_close_1, adj_close_2,
               daily_return, volume, amount, listed_state, currency,
               industry_1, industry_2
        FROM stock_daily
        ORDER BY trade_date, code
        """
    )
    headers = [description[0] for description in cursor.description]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(cursor)


def print_database_summary(conn):
    queries = [
        ("clean_rows", "SELECT COUNT(*) FROM stock_daily"),
        ("stocks", "SELECT COUNT(DISTINCT code) FROM stock_daily"),
        ("date_min", "SELECT MIN(trade_date) FROM stock_daily"),
        ("date_max", "SELECT MAX(trade_date) FROM stock_daily"),
        ("latest_price_rows", "SELECT COUNT(*) FROM latest_prices"),
    ]
    print("\nDatabase summary")
    for label, sql in queries:
        value = conn.execute(sql).fetchone()[0]
        print(f"  {label}: {value}")

    print("\nSample rows")
    for row in conn.execute(
        """
        SELECT code, name, trade_date, close, daily_return, amount, industry_1, industry_2
        FROM stock_daily
        ORDER BY trade_date, code
        LIMIT 5
        """
    ):
        print("  " + ", ".join("" if value is None else str(value) for value in row))


def now_iso():
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Clean RESSET DRESSTK Excel exports into a SQLite database for portfolio analysis."
    )
    parser.add_argument(
        "--source-dir",
        default=Path("data") / "raw" / "market_data",
        type=Path,
        help="Directory containing RESSET .xlsx files.",
    )
    parser.add_argument(
        "--database",
        default=Path("data") / "processed" / "stock_daily.sqlite",
        type=Path,
        help="Output SQLite database path.",
    )
    parser.add_argument(
        "--years",
        nargs="*",
        help="Optional list of years to import, for example: --years 2024 2025 2026",
    )
    parser.add_argument(
        "--include-st",
        action="store_true",
        help="Keep ST/*ST/SST rows. Default keeps only Listedstate == Norm.",
    )
    parser.add_argument(
        "--include-non-cny",
        action="store_true",
        help="Keep non-CNY and B-share rows. Default keeps CNY A-share-like rows only.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete the output database before importing.",
    )
    parser.add_argument(
        "--force-reimport",
        action="store_true",
        help="Re-read unchanged source files. By default, matching size and mtime are skipped.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10_000,
        help="SQLite insert batch size.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        help="Debug option: stop after this many read rows per sheet.",
    )
    parser.add_argument(
        "--export-csv",
        action="store_true",
        help="Also export a compact UTF-8 CSV beside the SQLite database.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv or sys.argv[1:])
    source_dir = args.source_dir
    database = args.database

    if not source_dir.exists():
        raise SystemExit(f"Source directory does not exist: {source_dir}")

    files = discover_files(source_dir, args.years)
    if not files:
        raise SystemExit(f"No .xlsx files found in {source_dir}")

    print("Input files")
    for path in files:
        print(f"  {path.name} ({path.stat().st_size / 1024 / 1024:.1f} MB)")

    imported_at = now_iso()
    conn = connect_database(database, reset=args.reset)
    totals = Counter()

    try:
        for path in files:
            if not args.force_reimport and source_file_is_unchanged(conn, path):
                print(f"Skipping unchanged file: {path.name}", flush=True)
                continue
            totals.update(process_file(conn, path, args, imported_at))
            record_source_file_state(conn, path, imported_at)
        print_database_summary(conn)

        if args.export_csv:
            csv_path = database.with_suffix(".csv")
            export_csv(conn, csv_path)
            print(f"\nCSV exported: {csv_path}")
    finally:
        conn.close()

    print("\nImport totals")
    for key in [
        "read_rows",
        "accepted_rows",
        "rejected_rows",
        "missing_essential",
        "non_cny_or_b_share",
        "not_normal_listing",
    ]:
        print(f"  {key}: {totals[key]:,}")

    print(f"\nSQLite database: {database}")


if __name__ == "__main__":
    main()
