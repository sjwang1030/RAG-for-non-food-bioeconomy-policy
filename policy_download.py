import argparse
import csv
import re
import unicodedata
import xml.etree.ElementTree as ET
from pathlib import Path
from zipfile import ZipFile
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE_PATH = BASE_DIR / "faolex_source"
DEFAULT_OUTPUT_DIR = BASE_DIR / "faolex_policies"
DEFAULT_LOG_PATH = DEFAULT_OUTPUT_DIR / "download_log.csv"
DEFAULT_CATALOG_PATH = DEFAULT_OUTPUT_DIR / "policy_catalog.csv"
DEFAULT_SUMMARY_PATH = DEFAULT_OUTPUT_DIR / "download_summary.csv"

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

SECTOR_RULES = {
    "bioenergy": [
        "bioenergy", "bio-energy", "biofuel", "bio-fuel", "biogas",
        "biomethane", "biomass energy", "biodiesel", "ethanol"
    ],
    "bio-based materials": [
        "bio-based material", "bio-based materials", "biomaterial", "biomaterials",
        "wood-based", "fiber-based", "fibre-based", "timber", "bioplastic", "bioplastics"
    ],
    "biochemicals": [
        "biochemical", "biochemicals", "bio-based chemical", "bio-based chemicals"
    ],
    "industrial biotechnology": [
        "industrial biotechnology", "biotechnology", "enzyme", "enzymes", "fermentation", "bioprocess"
    ],
    "biorefineries": [
        "biorefinery", "biorefineries"
    ],
    "biomass residue utilization": [
        "residue", "valorization", "organic residue", "crop residue", "forestry residue",
        "waste biomass", "biowaste", "bio-waste"
    ],
}


def build_session():
    session = requests.Session()
    retry = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update({"User-Agent": USER_AGENT})
    return session


def sanitize_filename(name, max_length=180):
    cleaned = unicodedata.normalize("NFKC", name or "").strip()
    cleaned = re.sub(r'[<>:"/\\|?*]+', "_", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = cleaned.rstrip(". ")

    if not cleaned:
        cleaned = "policy"

    if len(cleaned) > max_length:
        cleaned = cleaned[:max_length].rstrip()

    return cleaned


def build_output_filename(policy_id, faolex_no, title):
    policy_id_part = sanitize_filename(policy_id or "POL-UNKNOWN", max_length=24)
    faolex_part = sanitize_filename(faolex_no or "NO-FAOLEX", max_length=64)
    title_part = sanitize_filename(title or "policy", max_length=120)
    return f"{policy_id_part}_{faolex_part}_{title_part}.pdf"


def normalize_text(text):
    text = unicodedata.normalize("NFKC", text or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def canonical_url(url: str) -> str:
    parsed = urlparse(url)
    return parsed._replace(fragment="").geturl()


def merge_unique_values(old_value, new_value, sep="; "):
    old_items = [x.strip() for x in str(old_value or "").split(sep) if x.strip()]
    new_items = [x.strip() for x in str(new_value or "").split(sep) if x.strip()]
    merged = []
    seen = set()
    for item in old_items + new_items:
        key = item.lower()
        if key not in seen:
            seen.add(key)
            merged.append(item)
    return sep.join(merged)


def get_namespace(tag: str) -> str:
    if tag.startswith("{"):
        return tag[1:].split("}", 1)[0]
    return ""


def local_name(tag: str) -> str:
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def read_xml(zip_file, member_name):
    with zip_file.open(member_name) as handle:
        return ET.fromstring(handle.read())


def find_first_child_by_localname(parent, child_name):
    for child in list(parent):
        if local_name(child.tag) == child_name:
            return child
    return None


def find_all_descendants_by_localname(parent, child_name):
    return [el for el in parent.iter() if local_name(el.tag) == child_name]


def load_shared_strings(zip_file):
    try:
        root = read_xml(zip_file, "xl/sharedStrings.xml")
    except KeyError:
        return []

    values = []
    for si in find_all_descendants_by_localname(root, "si"):
        parts = []
        for text_node in si.iter():
            if local_name(text_node.tag) == "t":
                parts.append(text_node.text or "")
        values.append("".join(parts))
    return values


def get_cell_value(cell, shared_strings):
    value_node = None
    for child in list(cell):
        if local_name(child.tag) == "v":
            value_node = child
            break

    if value_node is None:
        return None

    raw = value_node.text or ""
    if cell.attrib.get("t") == "s":
        return shared_strings[int(raw)]
    return raw


def find_sheet_path(zip_file):
    workbook = read_xml(zip_file, "xl/workbook.xml")
    workbook_rels = read_xml(zip_file, "xl/_rels/workbook.xml.rels")

    sheets_node = find_first_child_by_localname(workbook, "sheets")
    if sheets_node is None:
        raise ValueError("No sheets node found in workbook.")

    first_sheet = None
    for child in list(sheets_node):
        if local_name(child.tag) == "sheet":
            first_sheet = child
            break

    if first_sheet is None:
        raise ValueError("No worksheet found in workbook.")

    rel_id = None
    for attr_key, attr_val in first_sheet.attrib.items():
        if attr_key.endswith("}id") or attr_key == "id":
            rel_id = attr_val
            break

    if not rel_id:
        raise ValueError("Worksheet relationship id not found.")

    for rel in workbook_rels.iter():
        if local_name(rel.tag) == "Relationship" and rel.attrib.get("Id") == rel_id:
            target = rel.attrib.get("Target", "")
            if target.startswith("xl/"):
                return target
            return f"xl/{target}"

    raise ValueError("Worksheet target not found in workbook relationships.")


def load_sheet_hyperlinks(zip_file, sheet_path):
    sheet = read_xml(zip_file, sheet_path)
    rels_path = str(Path(sheet_path).parent / "_rels" / f"{Path(sheet_path).name}.rels").replace("\\", "/")

    relationships = {}
    try:
        rels_root = read_xml(zip_file, rels_path)
        for rel in rels_root.iter():
            if local_name(rel.tag) == "Relationship":
                relationships[rel.attrib.get("Id")] = rel.attrib.get("Target")
    except KeyError:
        pass

    hyperlinks = {}
    for hyperlink in find_all_descendants_by_localname(sheet, "hyperlink"):
        ref = hyperlink.attrib.get("ref")
        rel_id = None
        for attr_key, attr_val in hyperlink.attrib.items():
            if attr_key.endswith("}id") or attr_key == "id":
                rel_id = attr_val
                break

        if ref and rel_id and rel_id in relationships:
            hyperlinks[ref] = relationships[rel_id]

    return sheet, hyperlinks


def find_columns(sheet, shared_strings, target_headers):
    normalized_targets = {k: v.strip().lower() for k, v in target_headers.items()}

    for row in find_all_descendants_by_localname(sheet, "row"):
        found = {}
        for cell in list(row):
            if local_name(cell.tag) != "c":
                continue

            value = get_cell_value(cell, shared_strings)
            if not isinstance(value, str):
                continue

            value_norm = value.strip().lower()
            for logical_name, header_text in normalized_targets.items():
                if value_norm == header_text:
                    cell_ref = cell.attrib.get("r", "")
                    found[logical_name] = re.sub(r"\d", "", cell_ref)

        if "title" in found:
            return found, int(row.attrib["r"])

    raise ValueError("Required header 'Title' not found in worksheet.")


def get_row_cell_by_column(row, column_letter):
    if not column_letter:
        return None
    target_pattern = re.compile(rf"^{re.escape(column_letter)}\d+$")
    for cell in list(row):
        if local_name(cell.tag) != "c":
            continue
        if target_pattern.match(cell.attrib.get("r", "")):
            return cell
    return None


def cell_value_from_row(row, column_letter, shared_strings):
    cell = get_row_cell_by_column(row, column_letter)
    if cell is None:
        return ""
    value = get_cell_value(cell, shared_strings)
    return "" if value is None else str(value).strip()


def normalize_repealed_status(value):
    if value is None:
        return "active"
    text = str(value).strip()
    if text == "":
        return "active"
    return "repealed"


def infer_keyword_from_filename(xlsx_path: Path) -> str:
    return xlsx_path.stem


def infer_non_food_sector(title, keywords, primary_subject):
    text = normalize_text(" | ".join([title or "", keywords or "", primary_subject or ""]))
    matched = []

    for sector, patterns in SECTOR_RULES.items():
        for pattern in patterns:
            if pattern in text:
                matched.append(sector)
                break

    if not matched:
        return "unspecified"
    if len(matched) == 1:
        return matched[0]
    return "cross-sectoral"


def extract_year(date_text: str) -> str:
    """
    Extract a 4-digit year from strings like:
    - 30 October 2020
    - 2020-10-30
    - 01 January 0001
    Returns "" if missing/invalid.
    Treats year 0001 as missing.
    """
    text = str(date_text or "").strip()
    if not text:
        return ""

    match = re.search(r"(\d{4})", text)
    if not match:
        return ""

    year = match.group(1)
    if year == "0001":
        return ""
    return year


def derive_policy_year(original_date_text: str, amended_date_text: str) -> str:
    """
    Prefer amendment year; if missing, use original year.
    """
    amendment_year = extract_year(amended_date_text)
    original_year = extract_year(original_date_text)

    if amendment_year:
        return amendment_year
    if original_year:
        return original_year
    return ""


def in_scope_2000_2025(policy_year: str) -> str:
    if not policy_year:
        return ""
    try:
        y = int(policy_year)
    except ValueError:
        return ""
    return "yes" if 2000 <= y <= 2025 else "no"


def load_policies_from_xlsx(xlsx_path):
    print(f"  Parsing workbook: {xlsx_path.name}")
    policies = []

    with ZipFile(xlsx_path) as zip_file:
        shared_strings = load_shared_strings(zip_file)
        sheet_path = find_sheet_path(zip_file)
        sheet, hyperlink_map = load_sheet_hyperlinks(zip_file, sheet_path)

        column_map, header_row = find_columns(
            sheet,
            shared_strings,
            {
                "jurisdiction": "Jurisdiction",
                "faolex_no": "FAOLEX No.",
                "type_of_text": "Type of text",
                "title": "Title",
                "date_of_text": "Date of text",
                "last_amended_date": "Last amended date",
                "repealed": "Repealed",
                "primary_subject": "Primary Subject",
                "keywords": "Keywords",
            },
        )

        keyword = infer_keyword_from_filename(xlsx_path)

        for row in find_all_descendants_by_localname(sheet, "row"):
            row_number = int(row.attrib["r"])
            if row_number <= header_row:
                continue

            title_col = column_map["title"]
            title_cell = get_row_cell_by_column(row, title_col)
            if title_cell is None:
                continue

            title_ref = title_cell.attrib.get("r", "")
            title = get_cell_value(title_cell, shared_strings)
            pdf_url = hyperlink_map.get(title_ref)

            if not title or not pdf_url:
                continue

            jurisdiction = cell_value_from_row(row, column_map.get("jurisdiction"), shared_strings)
            faolex_no = cell_value_from_row(row, column_map.get("faolex_no"), shared_strings)
            type_of_text = cell_value_from_row(row, column_map.get("type_of_text"), shared_strings)
            date_of_text = cell_value_from_row(row, column_map.get("date_of_text"), shared_strings)
            last_amended_date = cell_value_from_row(row, column_map.get("last_amended_date"), shared_strings)
            repealed_raw = cell_value_from_row(row, column_map.get("repealed"), shared_strings)
            primary_subject = cell_value_from_row(row, column_map.get("primary_subject"), shared_strings)
            keywords = cell_value_from_row(row, column_map.get("keywords"), shared_strings)

            policy_status = normalize_repealed_status(repealed_raw)
            sector = infer_non_food_sector(str(title), keywords, primary_subject)

            original_year = extract_year(date_of_text)
            amendment_year = extract_year(last_amended_date)
            policy_year = derive_policy_year(date_of_text, last_amended_date)
            scope_flag = in_scope_2000_2025(policy_year)

            policies.append(
                {
                    "source_xlsx": xlsx_path.name,
                    "source_keyword": keyword,
                    "row_number": row_number,
                    "faolex_no": faolex_no,
                    "jurisdiction": jurisdiction,
                    "type_of_text": type_of_text,
                    "title": str(title).strip(),
                    "date_of_text": date_of_text,
                    "last_amended_date": last_amended_date,
                    "original_year": original_year,
                    "amendment_year": amendment_year,
                    "policy_year": policy_year,
                    "in_scope_2000_2025": scope_flag,
                    "repealed_raw": repealed_raw,
                    "policy_status": policy_status,
                    "primary_subject": primary_subject,
                    "keywords": keywords,
                    "non_food_bio_based_sector": sector,
                    "pdf_url": pdf_url,
                }
            )

    if not policies:
        raise ValueError(f"No Title hyperlinks found in: {xlsx_path}")

    return policies


def resolve_xlsx_files(source_path):
    source_path = source_path.expanduser().resolve()

    if not source_path.exists():
        raise FileNotFoundError(f"Source path not found: {source_path}")

    if source_path.is_file():
        if source_path.suffix.lower() != ".xlsx":
            raise ValueError(f"Source file is not an xlsx: {source_path}")
        return [source_path]

    files = sorted(
        path for path in source_path.glob("*.xlsx")
        if path.is_file() and not path.name.startswith("~$")
    )

    if not files:
        raise ValueError(f"No xlsx files found in: {source_path}")

    return files


def download_file(session, url, output_path):
    with session.get(url, stream=True, timeout=180) as response:
        response.raise_for_status()

        with open(output_path, "wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 128):
                if chunk:
                    handle.write(chunk)


def build_output_path(output_dir, policy_id, title, faolex_no=""):
    return output_dir / build_output_filename(policy_id, faolex_no, title)


def resolve_existing_output_path(output_dir, policy_id, title, faolex_no=""):
    new_path = build_output_path(output_dir, policy_id, title, faolex_no)
    stem = sanitize_filename(title)
    legacy_path = output_dir / f"{stem}.pdf"

    if faolex_no:
        legacy_faolex_path = output_dir / f"{stem}__{sanitize_filename(faolex_no, max_length=50)}.pdf"
    else:
        legacy_faolex_path = legacy_path

    if new_path.exists() and new_path.stat().st_size > 0:
        return new_path, True

    if legacy_faolex_path.exists() and legacy_faolex_path.stat().st_size > 0:
        return legacy_faolex_path, True

    if legacy_path.exists() and legacy_path.stat().st_size > 0:
        return legacy_path, True

    return new_path, False


def load_existing_catalog(catalog_path):
    if not catalog_path.exists():
        return []

    with open(catalog_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def build_catalog_index(records):
    index = {}
    for row in records:
        faolex_no = (row.get("faolex_no") or "").strip()
        pdf_url = (row.get("pdf_url") or "").strip()

        if faolex_no:
            key = f"faolex::{faolex_no}"
        else:
            key = f"url::{canonical_url(pdf_url)}"

        index[key] = row
    return index


def make_policy_key(policy):
    if policy["faolex_no"]:
        return f"faolex::{policy['faolex_no']}"
    return f"url::{canonical_url(policy['pdf_url'])}"


def write_log(log_path, records):
    fieldnames = [
        "source_xlsx",
        "source_keyword",
        "row_number",
        "faolex_no",
        "title",
        "pdf_url",
        "policy_status",
        "policy_year",
        "output_file",
        "status",
        "error",
    ]

    with open(log_path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def write_catalog(catalog_path, records):
    fieldnames = [
        "policy_id",
        "source_xlsx",
        "source_keyword",
        "row_number",
        "faolex_no",
        "jurisdiction",
        "type_of_text",
        "title",
        "date_of_text",
        "last_amended_date",
        "original_year",
        "amendment_year",
        "policy_year",
        "in_scope_2000_2025",
        "repealed_raw",
        "policy_status",
        "primary_subject",
        "keywords",
        "non_food_bio_based_sector",
        "pdf_url",
        "file_name",
        "file_path",
        "download_status",
        "error",
    ]

    with open(catalog_path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def write_summary(summary_path, summary_rows):
    fieldnames = [
        "source_xlsx",
        "source_keyword",
        "raw_records",
        "new_records_added",
        "existing_records_updated",
        "downloaded",
        "skipped_existing_file",
        "failed",
        "parse_error",
    ]

    with open(summary_path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)


def main():
    parser = argparse.ArgumentParser(
        description="Incrementally download FAOLEX PDFs from Title hyperlinks and maintain a policy catalog."
    )
    parser.add_argument("--source", default=str(DEFAULT_SOURCE_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--log-file", default=str(DEFAULT_LOG_PATH))
    parser.add_argument("--catalog-file", default=str(DEFAULT_CATALOG_PATH))
    parser.add_argument("--summary-file", default=str(DEFAULT_SUMMARY_PATH))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--fresh-run",
        action="store_true",
        help="Ignore the existing catalog and rebuild policy IDs from scratch.",
    )
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Download again even if the target PDF file already exists.",
    )
    args = parser.parse_args()

    source_path = Path(args.source).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    log_path = Path(args.log_file).expanduser().resolve()
    catalog_path = Path(args.catalog_file).expanduser().resolve()
    summary_path = Path(args.summary_file).expanduser().resolve()

    output_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    session = build_session()
    xlsx_files = resolve_xlsx_files(source_path)

    existing_catalog = [] if args.fresh_run else load_existing_catalog(catalog_path)
    catalog_index = build_catalog_index(existing_catalog)

    print(f"Found {len(xlsx_files)} xlsx file(s) in: {source_path}")
    print(f"Loaded existing catalog records: {len(existing_catalog)}")

    log_records = []
    summary_rows = []

    downloaded_total = 0
    skipped_existing_total = 0
    failed_total = 0
    processed = 0
    next_policy_id = len(existing_catalog) + 1

    for file_index, xlsx_path in enumerate(xlsx_files, start=1):
        print(f"\n=== XLSX {file_index}/{len(xlsx_files)}: {xlsx_path.name} ===")

        try:
            policies = load_policies_from_xlsx(xlsx_path)
        except Exception as exc:
            error_msg = str(exc)

            print(f"  Failed to parse xlsx: {xlsx_path.name}")
            print(f"  Error: {error_msg}")

            summary_rows.append(
                {
                    "source_xlsx": xlsx_path.name,
                    "source_keyword": xlsx_path.stem,
                    "raw_records": 0,
                    "new_records_added": 0,
                    "existing_records_updated": 0,
                    "downloaded": 0,
                    "skipped_existing_file": 0,
                    "failed": 1,
                    "parse_error": error_msg,
                }
            )

            log_records.append(
                {
                    "source_xlsx": xlsx_path.name,
                    "source_keyword": xlsx_path.stem,
                    "row_number": "",
                    "faolex_no": "",
                    "title": "",
                    "pdf_url": "",
                    "policy_status": "",
                    "policy_year": "",
                    "output_file": "",
                    "status": "xlsx_parse_failed",
                    "error": error_msg,
                }
            )
            continue

        print(f"Loaded {len(policies)} policy records")

        file_new = 0
        file_updated = 0
        file_downloaded = 0
        file_skipped_existing = 0
        file_failed = 0

        for policy in policies:
            if args.limit > 0 and processed >= args.limit:
                break

            processed += 1
            key = make_policy_key(policy)
            existing = catalog_index.get(key)

            if existing:
                policy_id = existing.get("policy_id") or f"POL-{next_policy_id:06d}"
            else:
                policy_id = f"POL-{next_policy_id:06d}"

            if args.overwrite_existing:
                output_path = build_output_path(output_dir, policy_id, policy["title"], policy["faolex_no"])
                already_exists = output_path.exists() and output_path.stat().st_size > 0
            else:
                output_path, already_exists = resolve_existing_output_path(
                    output_dir, policy_id, policy["title"], policy["faolex_no"]
                )

            if existing and not args.overwrite_existing:
                existing["source_keyword"] = merge_unique_values(existing.get("source_keyword", ""), policy["source_keyword"])
                existing["source_xlsx"] = merge_unique_values(existing.get("source_xlsx", ""), policy["source_xlsx"])

                for field in [
                    "jurisdiction", "type_of_text", "date_of_text", "last_amended_date",
                    "original_year", "amendment_year", "policy_year", "in_scope_2000_2025",
                    "repealed_raw", "policy_status", "primary_subject", "keywords",
                    "non_food_bio_based_sector", "pdf_url"
                ]:
                    if not str(existing.get(field, "")).strip() and str(policy.get(field, "")).strip():
                        existing[field] = policy[field]

                existing_file_path = str(existing.get("file_path", "")).strip()
                should_sync_existing_path = (
                    already_exists and (
                        not existing_file_path
                        or existing_file_path != str(output_path)
                        or not Path(existing_file_path).exists()
                    )
                )
                if should_sync_existing_path:
                    existing["file_name"] = output_path.name
                    existing["file_path"] = str(output_path)
                    existing["download_status"] = "skipped_existing"

                if already_exists:
                    file_updated += 1
                    log_records.append(
                        {
                            "source_xlsx": policy["source_xlsx"],
                            "source_keyword": policy["source_keyword"],
                            "row_number": policy["row_number"],
                            "faolex_no": policy["faolex_no"],
                            "title": policy["title"],
                            "pdf_url": policy["pdf_url"],
                            "policy_status": policy["policy_status"],
                            "policy_year": policy["policy_year"],
                            "output_file": existing.get("file_path", ""),
                            "status": "existing_catalog_record",
                            "error": "",
                        }
                    )
                    continue

            if existing:
                record = existing
                record["source_keyword"] = merge_unique_values(record.get("source_keyword", ""), policy["source_keyword"])
                record["source_xlsx"] = merge_unique_values(record.get("source_xlsx", ""), policy["source_xlsx"])
            else:
                record = {
                    "policy_id": policy_id,
                    "source_xlsx": policy["source_xlsx"],
                    "source_keyword": policy["source_keyword"],
                    "row_number": policy["row_number"],
                    "faolex_no": policy["faolex_no"],
                    "jurisdiction": policy["jurisdiction"],
                    "type_of_text": policy["type_of_text"],
                    "title": policy["title"],
                    "date_of_text": policy["date_of_text"],
                    "last_amended_date": policy["last_amended_date"],
                    "original_year": policy["original_year"],
                    "amendment_year": policy["amendment_year"],
                    "policy_year": policy["policy_year"],
                    "in_scope_2000_2025": policy["in_scope_2000_2025"],
                    "repealed_raw": policy["repealed_raw"],
                    "policy_status": policy["policy_status"],
                    "primary_subject": policy["primary_subject"],
                    "keywords": policy["keywords"],
                    "non_food_bio_based_sector": policy["non_food_bio_based_sector"],
                    "pdf_url": policy["pdf_url"],
                    "file_name": "",
                    "file_path": "",
                    "download_status": "",
                    "error": "",
                }

            for field in [
                "jurisdiction", "type_of_text", "title", "date_of_text", "last_amended_date",
                "original_year", "amendment_year", "policy_year", "in_scope_2000_2025",
                "repealed_raw", "policy_status", "primary_subject", "keywords",
                "non_food_bio_based_sector", "pdf_url"
            ]:
                record[field] = policy[field]

            if already_exists and not args.overwrite_existing:
                record["file_name"] = output_path.name
                record["file_path"] = str(output_path)
                record["download_status"] = "skipped_existing"
                file_skipped_existing += 1
                skipped_existing_total += 1
                print(f"[{processed}] Skipped existing file: {output_path.name}")
            else:
                try:
                    download_file(session, policy["pdf_url"], output_path)
                    record["file_name"] = output_path.name
                    record["file_path"] = str(output_path)
                    record["download_status"] = "downloaded"
                    file_downloaded += 1
                    downloaded_total += 1
                    print(f"[{processed}] Downloaded: {output_path.name}")
                except Exception as exc:
                    record["file_name"] = output_path.name
                    record["file_path"] = str(output_path)
                    record["download_status"] = "failed"
                    record["error"] = str(exc)
                    file_failed += 1
                    failed_total += 1
                    output_path.unlink(missing_ok=True)
                    print(f"[{processed}] Download failed: {policy['title']} | {exc}")

            if existing:
                file_updated += 1
            else:
                existing_catalog.append(record)
                catalog_index[key] = record
                next_policy_id += 1
                file_new += 1

            log_records.append(
                {
                    "source_xlsx": policy["source_xlsx"],
                    "source_keyword": policy["source_keyword"],
                    "row_number": policy["row_number"],
                    "faolex_no": policy["faolex_no"],
                    "title": policy["title"],
                    "pdf_url": policy["pdf_url"],
                    "policy_status": policy["policy_status"],
                    "policy_year": policy["policy_year"],
                    "output_file": record["file_path"],
                    "status": record["download_status"],
                    "error": record["error"],
                }
            )

        summary_rows.append(
            {
                "source_xlsx": xlsx_path.name,
                "source_keyword": xlsx_path.stem,
                "raw_records": len(policies),
                "new_records_added": file_new,
                "existing_records_updated": file_updated,
                "downloaded": file_downloaded,
                "skipped_existing_file": file_skipped_existing,
                "failed": file_failed,
                "parse_error": "",
            }
        )

        if args.limit > 0 and processed >= args.limit:
            break

    write_log(log_path, log_records)
    write_catalog(catalog_path, existing_catalog)
    write_summary(summary_path, summary_rows)

    print("\nFinished")
    print(f"Downloaded this run: {downloaded_total}")
    print(f"Skipped existing files this run: {skipped_existing_total}")
    print(f"Failed this run: {failed_total}")
    print(f"Catalog saved: {catalog_path}")
    print(f"Run log saved: {log_path}")
    print(f"Summary saved: {summary_path}")


if __name__ == "__main__":
    main()
