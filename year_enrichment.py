import csv
import re
import time
from datetime import datetime, timedelta
from pathlib import Path

import requests


BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = BASE_DIR / "faolex_policies" / "policy_catalog.csv"
OUTPUT_DIR = BASE_DIR / "outputs" / "year_enrichment"
OUTPUT_FILE = OUTPUT_DIR / "policy_catalog_year_enriched.csv"
API_URL = "https://fao-faolex-prod.appspot.com/api/query"

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Content-Type": "application/json",
}

REQUEST_OPTIONS = {
    "searchApplicationId": "searchapplications/1be285f8874b8c6bfaabf84aa9d0c1be"
}

FACET_OPTIONS = [
    {"operatorName": "facetyears", "numFacetBuckets": 100},
    {"operatorName": "countries", "numFacetBuckets": 100},
    {"operatorName": "documentlanguages", "numFacetBuckets": 100},
    {"operatorName": "geographicalareas", "numFacetBuckets": 100},
    {"operatorName": "keywords", "numFacetBuckets": 100},
    {"operatorName": "mainareas", "numFacetBuckets": 100},
    {"operatorName": "subjectselections", "numFacetBuckets": 100},
    {"operatorName": "territorialsubdivisions", "numFacetBuckets": 100},
    {"operatorName": "typeoftexts", "numFacetBuckets": 100},
]

SORT_OPTIONS = {
    "operatorName": "byyear",
    "sortOrder": "DESCENDING",
}

SLEEP_TIME = 0.8


def extract_year(text):
    if not text:
        return ""

    normalized = normalize_date_value(text)
    match = re.search(r"(\d{4})", str(normalized))
    if not match:
        return ""

    year = match.group(1)
    if year == "0001":
        return ""

    return year


def derive_policy_year(original, amended):
    if amended:
        return amended
    return original


def excel_serial_to_iso(value):
    text = str(value or "").strip()
    if not text:
        return ""

    if not re.fullmatch(r"\d+(\.0+)?", text):
        return ""

    try:
        serial = int(float(text))
    except Exception:
        return ""

    if not (20000 <= serial <= 60000):
        return ""

    base_date = datetime(1899, 12, 30)
    converted = base_date + timedelta(days=serial)
    return converted.strftime("%Y-%m-%d")


def normalize_date_value(value):
    text = str(value or "").strip()
    if not text:
        return ""

    excel_date = excel_serial_to_iso(text)
    if excel_date:
        return excel_date

    return text


def parse_year(value):
    value = str(value or "").strip()
    if not value:
        return None

    if not re.fullmatch(r"\d{4}", value):
        return None

    year = int(value)
    if 1000 <= year <= 2100:
        return year

    return None


def in_scope(year):
    parsed = parse_year(year)
    if parsed is None:
        return ""
    return "yes" if 2000 <= parsed <= 2025 else "no"


def needs_web_fetch(policy_year):
    parsed = parse_year(policy_year)
    if parsed is None:
        return True
    return not (2000 <= parsed <= 2025)


def build_payload(faolex_no):
    return {
        "query": f'faolexid:("{faolex_no}")',
        "start": 0,
        "requestOptions": REQUEST_OPTIONS,
        "sortOptions": SORT_OPTIONS,
        "facetOptions": FACET_OPTIONS,
    }


def get_field_values(fields, field_name):
    for field in fields:
        if field.get("name") != field_name:
            continue

        text_values = field.get("textValues", {}).get("values", [])
        if text_values:
            return [str(v).strip() for v in text_values if str(v).strip()]

        integer_values = field.get("integerValues", {}).get("values", [])
        if integer_values:
            return [str(v).strip() for v in integer_values if str(v).strip()]

    return []


def fetch_faolex_metadata(faolex_no):
    payload = build_payload(faolex_no)

    try:
        response = requests.post(API_URL, headers=HEADERS, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
    except Exception as exc:
        print("Error:", faolex_no, exc)
        return None, None

    results = data.get("results", [])
    if not results:
        return None, None

    metadata = results[0].get("metadata", {})
    fields = metadata.get("fields", [])

    original_values = get_field_values(fields, "dateOfOriginalText")
    consolidation_values = get_field_values(fields, "dateOfConsolidation")
    original_year_values = get_field_values(fields, "originalYear")
    year_values = get_field_values(fields, "year")
    all_year_values = get_field_values(fields, "allYear")

    original = original_values[0] if original_values else ""
    amended = consolidation_values[0] if consolidation_values else ""

    if not original and original_year_values:
        original = f"{original_year_values[0]}-01-01"

    if not amended and year_values:
        amended = f"{year_values[0]}-01-01"

    if not original and all_year_values:
        oldest = min(all_year_values, key=lambda x: int(x))
        original = f"{oldest}-01-01"

    if not amended and all_year_values:
        newest = max(all_year_values, key=lambda x: int(x))
        amended = f"{newest}-01-01"

    return original, amended


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(INPUT_FILE, encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    total = len(rows)
    checked = 0
    updated = 0

    for index, row in enumerate(rows):
        row["date_of_text"] = normalize_date_value(row.get("date_of_text", ""))
        row["last_amended_date"] = normalize_date_value(row.get("last_amended_date", ""))
        row["original_year"] = extract_year(row.get("date_of_text", ""))
        row["amendment_year"] = extract_year(row.get("last_amended_date", ""))

        if row["original_year"] or row["amendment_year"]:
            row["policy_year"] = derive_policy_year(row["original_year"], row["amendment_year"])
            row["in_scope_2000_2025"] = in_scope(row["policy_year"])

        policy_year = row.get("policy_year", "")
        faolex_no = row.get("faolex_no", "")

        if not faolex_no:
            continue

        if not needs_web_fetch(policy_year):
            continue

        checked += 1
        print(f"[{index + 1}/{total}] Checking {faolex_no}")

        original, amended = fetch_faolex_metadata(faolex_no)
        if not original and not amended:
            continue

        original_year = extract_year(original)
        amendment_year = extract_year(amended)
        policy_year = derive_policy_year(original_year, amendment_year)
        scope = in_scope(policy_year)

        row["original_year"] = original_year
        row["amendment_year"] = amendment_year
        row["policy_year"] = policy_year
        row["in_scope_2000_2025"] = scope

        updated += 1
        time.sleep(SLEEP_TIME)

    print("\nChecked:", checked)
    print("Updated:", updated)

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print("Saved to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
