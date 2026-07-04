import csv
import json
import os
import re
import unicodedata
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = BASE_DIR / "outputs" / "year_enrichment" / "policy_catalog_year_enriched.csv"
OUTPUT_DIR = BASE_DIR / "outputs" / "jurisdiction_filter"
OUTPUT_FILE = OUTPUT_DIR / "policy_catalog_jurisdiction_filtered.csv"
SUMMARY_FILE = OUTPUT_DIR / "jurisdiction_filter_summary.json"


load_dotenv(BASE_DIR / ".env")
client = OpenAI()

MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")


SUBNATIONAL_KEYWORDS = [
    "province",
    "state",
    "region",
    "territory",
    "county",
    "municipal",
    "city",
    "prefecture",
    "district",
    "department",
    "canton",
    "local level",
    "local government",
    "municipality",
    "governorate",
    "parish",
]

MACRO_REGIONAL = [
    "European Union",
    "EU",
    "East African Community",
    "EAC",
    "African Union",
    "ASEAN",
    "OECD",
    "APEC",
    "Mercosur",
    "CARICOM",
]

MACRO_REGIONAL_KEYWORDS = [
    "commission",
    "organization",
    "organisation",
    "community",
    "union",
    "council",
    "secretariat",
    "conference",
    "tuna commission",
]

SOVEREIGN_QUALIFIERS = [
    "kingdom of",
    "kingdom of the",
    "republic of",
    "federal republic of",
    "federative republic of",
    "federated states of",
    "islamic republic of",
    "plurinational state of",
    "state of",
    "union of",
    "commonwealth of",
    "democratic republic of",
    "people's republic of",
]


COUNTRY_BASES = [
    "algeria",
    "azerbaijan",
    "australia",
    "belarus",
    "belgium",
    "benin",
    "bolivia",
    "bosnia and herzegovina",
    "bulgaria",
    "burkina faso",
    "canada",
    "cabo verde",
    "chile",
    "china",
    "colombia",
    "croatia",
    "côte d'ivoire",
    "cote d'ivoire",
    "denmark",
    "democratic people's republic of korea",
    "democratic republic of the congo",
    "estonia",
    "finland",
    "georgia",
    "greece",
    "ghana",
    "india",
    "iran",
    "kenya",
    "kyrgyzstan",
    "lao people's democratic republic",
    "latvia",
    "mali",
    "malaysia",
    "montenegro",
    "netherlands",
    "north macedonia",
    "micronesia",
    "philippines",
    "united kingdom of great britain and northern ireland",
    "united republic of tanzania",
    "united states",
    "united states of america",
    "usa",
    "mexico",
    "brazil",
    "argentina",
    "germany",
    "france",
    "italy",
    "pakistan",
    "peru",
    "romania",
    "russian federation",
    "saudi arabia",
    "serbia",
    "slovakia",
    "spain",
    "japan",
    "indonesia",
    "south africa",
    "timor-leste",
    "togo",
    "türkiye",
    "united kingdom",
    "ukraine",
    "uk",
    "venezuela",
    "viet nam",
    "switzerland",
    "zambia",
]

SUBNATIONAL_BRACKET_NAMES = [
    "new brunswick",
    "northern ireland",
    "vlaanderen",
    "victoria",
    "makueni",
    "queensland",
]

NON_SOVEREIGN_BASE_HINTS = [
    "zanzibar",
    "bonaire",
    "sint eustatius",
    "saba",
]

SPECIAL_JURISDICTION_KEYWORDS = [
    "unscr",
    "unmik",
    "gibraltar",
    "northern mariana islands",
    "hong kong",
    "macao",
    "macau",
    "tokelau",
    "guam",
    "puerto rico",
    "falkland islands",
    "bermuda",
    "greenland",
]


def is_blank(value):
    return value is None or str(value).strip() == ""


def contains_special_jurisdiction(value):
    value = str(value or "").strip().lower()
    return any(keyword in value for keyword in SPECIAL_JURISDICTION_KEYWORDS)


def classify_special_jurisdiction(value):
    value = str(value or "").strip().lower()
    if not value:
        return None

    if "hong kong" in value:
        return ("subnational", False) if "china" in value else ("unclear", False)

    if "macao" in value or "macau" in value:
        return ("subnational", False) if "china" in value else ("unclear", False)

    if "greenland" in value:
        return ("subnational", False) if "denmark" in value else ("unclear", False)

    if "tokelau" in value:
        return ("subnational", False) if "new zealand" in value else ("unclear", False)

    if "gibraltar" in value or "bermuda" in value:
        return ("subnational", False) if ("uk" in value or "united kingdom" in value) else ("unclear", False)

    if "guam" in value or "puerto rico" in value or "northern mariana islands" in value:
        return ("subnational", False) if ("usa" in value or "united states" in value) else ("unclear", False)

    if "cook islands" in value:
        return "unclear", False

    if "kosovo" in value or "somaliland" in value:
        return "unclear", False

    return None


def is_sovereign_like_part(part):
    part = str(part or "").strip().lower()
    if not part:
        return False

    if part in COUNTRY_BASES:
        return True

    return any(qualifier in part for qualifier in SOVEREIGN_QUALIFIERS)


def is_macro_regional_combo_part(part):
    part = str(part or "").strip().lower()
    if not part:
        return False

    if is_sovereign_like_part(part):
        return True

    if any(region_name_matches(part, region_name) for region_name in MACRO_REGIONAL):
        return True

    return any(keyword in part for keyword in MACRO_REGIONAL_KEYWORDS)


def region_name_matches(value, region_name):
    value = str(value or "").strip().lower()
    region = str(region_name or "").strip().lower()
    if not value or not region:
        return False

    if len(region) <= 4 and re.fullmatch(r"[a-z]+", region):
        return re.search(rf"\b{re.escape(region)}\b", value) is not None

    return region in value


def rule_filter(jurisdiction):
    if is_blank(jurisdiction):
        return "unclear", None

    value = str(jurisdiction).strip().lower()

    special_result = classify_special_jurisdiction(value)
    if special_result is not None:
        return special_result

    if ";" in value:
        parts = [p.strip() for p in value.split(";") if p.strip()]
        if len(parts) >= 2 and any(classify_special_jurisdiction(part) is not None for part in parts):
            return "unclear", None
        if len(parts) >= 2 and all(is_macro_regional_combo_part(part) for part in parts):
            return "macro_regional", True
        if len(parts) >= 2:
            return "unclear", None

    if "," in value:
        parts = [p.strip() for p in value.split(",") if p.strip()]
        if len(parts) >= 2 and any(classify_special_jurisdiction(part) is not None for part in parts):
            return "unclear", None
        if len(parts) >= 2 and all(is_macro_regional_combo_part(part) for part in parts):
            return "macro_regional", True
        if len(parts) >= 2:
            return "unclear", None

    if contains_special_jurisdiction(value):
        if bracket_match := re.match(r"^(.+?)\s*\(([^)]+)\)\s*$", value):
            bracket_name = bracket_match.group(2).strip()
            if bracket_name in COUNTRY_BASES or is_sovereign_like_part(bracket_name):
                return "subnational", False
        return "unclear", False

    if value in COUNTRY_BASES:
        return "national", True

    for region_name in MACRO_REGIONAL:
        if region_name_matches(value, region_name):
            return "macro_regional", True

    for keyword in MACRO_REGIONAL_KEYWORDS:
        if keyword in value:
            return "macro_regional", True

    # Country followed by a bracketed territorial unit, e.g. Australia (Queensland)
    bracket_match = re.match(r"^(.+?)\s*\(([^)]+)\)\s*$", value)
    if bracket_match:
        base_name = bracket_match.group(1).strip()
        bracket_name = bracket_match.group(2).strip()

        if "regional organization" in bracket_name:
            return "macro_regional", True

        if any(hint in base_name for hint in NON_SOVEREIGN_BASE_HINTS):
            return "subnational", False

        for keyword in SUBNATIONAL_KEYWORDS:
            if keyword in bracket_name:
                return "subnational", False

        if bracket_name in SUBNATIONAL_BRACKET_NAMES:
            return "subnational", False

        if base_name in COUNTRY_BASES:
            for qualifier in SOVEREIGN_QUALIFIERS:
                if qualifier in bracket_name:
                    return "national", True
            return "subnational", False

        if is_sovereign_like_part(base_name):
            for qualifier in SOVEREIGN_QUALIFIERS:
                if qualifier in bracket_name:
                    return "national", True

        if bracket_name in COUNTRY_BASES:
            return "subnational", False

        for qualifier in SOVEREIGN_QUALIFIERS:
            if qualifier in bracket_name:
                return "subnational", False

        return "unclear", None

    for keyword in SUBNATIONAL_KEYWORDS:
        if re.search(rf"\b{re.escape(keyword)}\b", value):
            return "subnational", False

    return "unclear", None


JF_MACRO_REGIONAL = [
    "European Union",
    "EU",
    "East African Community",
    "EAC",
    "African Union",
    "ASEAN",
    "OECD",
    "APEC",
    "Mercosur",
    "CARICOM",
    "Andean Community",
    "Amazon Cooperation Treaty Organization",
    "Association of Southeast Asian Nations",
    "ASCOBANS",
    "AU",
]

JF_MACRO_REGIONAL_KEYWORDS = [
    "commission",
    "organization",
    "organisation",
    "community",
    "union",
    "council",
    "secretariat",
    "conference",
    "tuna commission",
    "regional organization",
]

JF_SOVEREIGN_QUALIFIERS = [
    "kingdom of",
    "kingdom of the",
    "republic of",
    "federal republic of",
    "federative republic of",
    "federated states of",
    "islamic republic of",
    "plurinational state of",
    "bolivarian republic of",
    "state of",
    "union of",
    "commonwealth of",
    "democratic republic of",
    "people's republic of",
    "united republic of",
]

JF_COUNTRY_BASES = [
    "afghanistan", "albania", "algeria", "andorra", "angola", "antigua and barbuda",
    "argentina", "armenia", "australia", "austria", "azerbaijan", "bahamas", "bahrain",
    "bangladesh", "barbados", "belarus", "belgium", "belize", "benin", "bhutan", "bolivia",
    "bosnia and herzegovina", "botswana", "brazil", "brunei darussalam", "bulgaria",
    "burkina faso", "burundi", "cabo verde", "cambodia", "cameroon", "canada",
    "central african republic", "chad", "chile", "china", "colombia", "comoros", "congo",
    "costa rica", "cote d'ivoire", "croatia", "cuba", "cyprus", "czech republic",
    "czechia", "democratic people's republic of korea", "democratic republic of the congo",
    "denmark", "djibouti", "dominica", "dominican republic", "ecuador", "egypt",
    "el salvador", "equatorial guinea", "eritrea", "estonia", "eswatini", "ethiopia",
    "fiji", "finland", "france", "gabon", "gambia", "georgia", "germany", "ghana",
    "greece", "grenada", "guatemala", "guinea", "guinea-bissau", "guyana", "haiti",
    "honduras", "hungary", "iceland", "india", "indonesia", "iran", "iraq", "ireland",
    "israel", "italy", "jamaica", "japan", "jordan", "kazakhstan", "kenya", "kiribati",
    "kuwait", "kyrgyzstan", "lao people's democratic republic", "latvia", "lebanon",
    "lesotho", "liberia", "libya", "lithuania", "luxembourg", "madagascar", "malawi",
    "malaysia", "maldives", "mali", "malta", "marshall islands", "mauritania",
    "mauritius", "mexico", "micronesia", "moldova", "mongolia", "montenegro", "morocco",
    "mozambique", "myanmar", "namibia", "nauru", "nepal", "netherlands", "new zealand",
    "nicaragua", "niger", "nigeria", "north macedonia", "norway", "oman", "pakistan",
    "panama", "papua new guinea", "paraguay", "peru", "philippines", "poland", "portugal",
    "qatar", "republic of korea", "republic of moldova", "romania", "russian federation",
    "rwanda", "saint lucia", "saint vincent and the grenadines", "samoa",
    "sao tome and principe", "saudi arabia", "senegal", "serbia", "seychelles",
    "sierra leone", "singapore", "slovakia", "slovenia", "solomon islands", "somalia",
    "south africa", "south sudan", "spain", "sri lanka", "sudan", "suriname", "sweden",
    "switzerland", "tajikistan", "thailand", "timor-leste", "togo", "tonga",
    "trinidad and tobago", "tunisia", "turkiye", "turkmenistan", "tuvalu", "uganda", "uk",
    "ukraine", "united kingdom", "united kingdom of great britain and northern ireland",
    "united republic of tanzania", "united states", "united states of america", "uruguay",
    "usa", "uzbekistan", "vanuatu", "venezuela", "viet nam", "yemen", "zambia",
    "zimbabwe",
]

JF_SOVEREIGN_COMMA_FORMS = [
    "bolivia, plurinational state of",
    "iran, islamic republic of",
    "korea, republic of",
    "micronesia, federated states of",
    "moldova, republic of",
    "tanzania, united republic of",
    "venezuela, bolivarian republic of",
]

JF_SUBNATIONAL_BRACKET_NAMES = [
    "new brunswick",
    "northern ireland",
    "vlaanderen",
    "victoria",
    "makueni",
    "queensland",
    "vorarlberg",
    "basel-landschaft",
    "meghalaya",
    "osun",
    "amhara",
]

JF_SPECIAL_JURISDICTION_KEYWORDS = [
    "american samoa",
    "aruba",
    "bermuda",
    "british virgin islands",
    "cayman islands",
    "curacao",
    "falkland islands",
    "faroe islands",
    "french polynesia",
    "gibraltar",
    "greenland",
    "guam",
    "kosovo",
    "new caledonia",
    "northern mariana islands",
    "puerto rico",
    "saint helena",
    "somaliland",
    "tokelau",
    "unmik",
    "unscr",
    "u.s. virgin islands",
    "virgin islands",
    "western sahara",
]

JF_SPECIAL_NATIONAL_KEYWORDS = [
    "bolivia (plurinational state of)",
    "cook islands",
    "palestine",
]

JF_SPECIAL_UNCLEAR_KEYWORDS = [
    "niue",
]

JF_SPECIAL_SUBNATIONAL_KEYWORDS = [
    "taiwan",
    "hong kong",
    "macao",
    "macau",
    "zanzibar",
    "andaman and nicobar islands",
    "arunachal pradesh",
    "jammu and kashmir",
    "canary islands",
    "madeira islands",
]

JF_PAREN_RE = re.compile(r"^(.+?)\s*\(([^)]+)\)\s*$")


def jf_normalize_text(value):
    text = str(value or "").strip()
    if not text:
        return ""

    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


JF_COUNTRY_BASE_SET = {jf_normalize_text(item) for item in JF_COUNTRY_BASES}
JF_SOVEREIGN_COMMA_SET = {jf_normalize_text(item) for item in JF_SOVEREIGN_COMMA_FORMS}
JF_SUBNATIONAL_BRACKET_SET = {jf_normalize_text(item) for item in JF_SUBNATIONAL_BRACKET_NAMES}
JF_SPECIAL_NATIONAL_SET = {jf_normalize_text(item) for item in JF_SPECIAL_NATIONAL_KEYWORDS}


def jf_region_name_matches(value, region_name):
    value_norm = jf_normalize_text(value)
    region_norm = jf_normalize_text(region_name)
    if not value_norm or not region_norm:
        return False

    if len(region_norm) <= 4 and re.fullmatch(r"[a-z]+", region_norm):
        return re.search(rf"\b{re.escape(region_norm)}\b", value_norm) is not None

    return region_norm in value_norm


def jf_contains_special_jurisdiction(value):
    value_norm = jf_normalize_text(value)
    return any(keyword in value_norm for keyword in JF_SPECIAL_JURISDICTION_KEYWORDS)


def jf_contains_special_national(value):
    value_norm = jf_normalize_text(value)
    return value_norm in JF_SPECIAL_NATIONAL_SET


def jf_contains_special_unclear(value):
    value_norm = jf_normalize_text(value)
    return any(keyword in value_norm for keyword in JF_SPECIAL_UNCLEAR_KEYWORDS)


def jf_contains_special_subnational(value):
    value_norm = jf_normalize_text(value)
    return any(keyword in value_norm for keyword in JF_SPECIAL_SUBNATIONAL_KEYWORDS)


def jf_contains_subnational_keyword(value):
    value_norm = jf_normalize_text(value)
    return any(re.search(rf"\b{re.escape(keyword)}\b", value_norm) for keyword in SUBNATIONAL_KEYWORDS)


def jf_is_country_with_formal_parenthetical(value):
    value_norm = jf_normalize_text(value)
    match = JF_PAREN_RE.match(value_norm)
    if not match:
        return False

    base_name = match.group(1).strip()
    bracket_name = match.group(2).strip()

    # Prevent forms like "State of Washington" from being promoted to national.
    if jf_contains_subnational_keyword(bracket_name) or bracket_name in JF_SUBNATIONAL_BRACKET_SET:
        return False

    return base_name in JF_COUNTRY_BASE_SET and any(
        qualifier in bracket_name for qualifier in JF_SOVEREIGN_QUALIFIERS
    )


def jf_is_sovereign_like_part(part):
    part_norm = jf_normalize_text(part)
    if not part_norm:
        return False

    if part_norm in JF_COUNTRY_BASE_SET or part_norm in JF_SOVEREIGN_COMMA_SET:
        return True

    if jf_is_country_with_formal_parenthetical(part_norm):
        return True

    return any(qualifier in part_norm for qualifier in JF_SOVEREIGN_QUALIFIERS)


def jf_is_macro_regional_combo_part(part):
    part_norm = jf_normalize_text(part)
    if not part_norm:
        return False

    if jf_is_sovereign_like_part(part_norm):
        return True

    if any(jf_region_name_matches(part_norm, region_name) for region_name in JF_MACRO_REGIONAL):
        return True

    return any(keyword in part_norm for keyword in JF_MACRO_REGIONAL_KEYWORDS)


def jf_classify_multi_part(value, separator):
    parts = [jf_normalize_text(part) for part in value.split(separator) if jf_normalize_text(part)]
    if len(parts) < 2:
        return None

    if any(jf_contains_special_subnational(part) for part in parts):
        return "subnational", False

    if any(jf_contains_special_national(part) for part in parts):
        if all(jf_is_macro_regional_combo_part(part) or jf_contains_special_national(part) for part in parts):
            return "macro_regional", True
        return "national", True

    if any(jf_contains_special_unclear(part) for part in parts):
        return "unclear", False

    if any(jf_contains_special_jurisdiction(part) for part in parts):
        return "unclear", False

    if all(jf_is_macro_regional_combo_part(part) for part in parts):
        return "macro_regional", True

    return "unclear", None


def embedded_rule_filter(jurisdiction):
    if jf_normalize_text(jurisdiction) == "":
        return "unclear", None

    value = jf_normalize_text(jurisdiction)

    if jf_contains_special_subnational(value):
        return "subnational", False

    if jf_contains_special_national(value):
        return "national", True

    if jf_contains_special_unclear(value):
        return "unclear", False

    if jf_contains_special_jurisdiction(value):
        return "unclear", False

    if (
        value in JF_COUNTRY_BASE_SET
        or value in JF_SOVEREIGN_COMMA_SET
        or jf_is_country_with_formal_parenthetical(value)
    ):
        return "national", True

    for region_name in JF_MACRO_REGIONAL:
        if jf_region_name_matches(value, region_name):
            return "macro_regional", True

    for keyword in JF_MACRO_REGIONAL_KEYWORDS:
        if keyword in value:
            return "macro_regional", True

    bracket_match = JF_PAREN_RE.match(value)
    if bracket_match:
        base_name = bracket_match.group(1).strip()
        bracket_name = bracket_match.group(2).strip()

        if "regional organization" in bracket_name:
            return "macro_regional", True

        if jf_contains_special_jurisdiction(base_name) or jf_contains_special_jurisdiction(bracket_name):
            return "unclear", False

        if jf_contains_subnational_keyword(bracket_name) or bracket_name in JF_SUBNATIONAL_BRACKET_SET:
            return "subnational", False

        if base_name in JF_COUNTRY_BASE_SET:
            if any(qualifier in bracket_name for qualifier in JF_SOVEREIGN_QUALIFIERS):
                return "national", True
            if bracket_name in JF_COUNTRY_BASE_SET or jf_is_sovereign_like_part(bracket_name):
                return "unclear", False
            return "subnational", False

        if bracket_name in JF_COUNTRY_BASE_SET:
            return "unclear", False

        if any(qualifier in bracket_name for qualifier in JF_SOVEREIGN_QUALIFIERS):
            return "unclear", False

        return "unclear", None

    if ";" in value:
        result = jf_classify_multi_part(value, ";")
        if result is not None:
            return result

    if "," in value and value not in JF_SOVEREIGN_COMMA_SET:
        result = jf_classify_multi_part(value, ",")
        if result is not None:
            return result

    if jf_contains_subnational_keyword(value):
        return "subnational", False

    return "unclear", None


rule_filter = embedded_rule_filter


def try_parse_json(text):
    if not text:
        return None

    text = text.strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"(\{.*\})", text, flags=re.S)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            pass

    return None


def normalize_llm_result(result):
    if not isinstance(result, dict):
        return {"level": "unclear", "keep": False}

    level = str(result.get("level", "unclear")).strip().lower()
    keep = result.get("keep")

    if level not in {"national", "macro_regional", "subnational", "unclear"}:
        level = "unclear"

    if isinstance(keep, str):
        keep = keep.strip().lower() == "true"
    elif not isinstance(keep, bool):
        keep = False

    return {"level": level, "keep": keep}


def llm_filter(jurisdiction):
    prompt = f"""
Determine jurisdiction level.

Keep only:
- national
- macro-regional

Exclude:
- province
- state
- city
- subnational

Additional rules:
- If the jurisdiction is a list of multiple sovereign countries separated by ";" or commas,
  classify as macro_regional and keep = true.
- If the text is a sovereign country with a formal state qualifier in parentheses,
  such as "Plurinational State of", "Kingdom of the", or "Federated States of",
  classify as national and keep = true.
- If the text clearly names a province, state, county, or other internal territorial unit,
  classify as subnational and keep = false.
- If the text is a territory, protectorate, disputed entity, UN-administered entity,
  or another special international legal status that is not clearly national or macro-regional,
  classify as unclear and keep = false.

Jurisdiction:
{jurisdiction}

Return JSON:

{{
  "level": "national | macro_regional | subnational | unclear",
  "keep": true
}}
"""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a policy classification expert."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        content = response.choices[0].message.content or ""
        parsed = try_parse_json(content)
        return normalize_llm_result(parsed)
    except Exception:
        return {"level": "unclear", "keep": False}


def safe_write_csv(path, fieldnames, rows):
    try:
        with open(path, "w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return path
    except PermissionError:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fallback_path = path.with_name(f"{path.stem}_{timestamp}{path.suffix}")
        with open(fallback_path, "w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return fallback_path


def safe_write_json(path, payload):
    try:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
        return path
    except PermissionError:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fallback_path = path.with_name(f"{path.stem}_{timestamp}{path.suffix}")
        with open(fallback_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
        return fallback_path


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(INPUT_FILE, "r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    print("Before filtering:", len(rows))

    level_counts = {
        "national": 0,
        "macro_regional": 0,
        "subnational": 0,
        "unclear": 0,
    }
    method_counts = {
        "rule": 0,
        "llm": 0,
    }

    for row in rows:
        jurisdiction = row.get("jurisdiction", "")
        level, keep = rule_filter(jurisdiction)

        if keep is None:
            result = llm_filter(jurisdiction)
            level = result.get("level", "unclear")
            keep = result.get("keep", False)
            reason = "llm"
        else:
            reason = "rule"

        row["jurisdiction_level"] = level
        row["jurisdiction_keep"] = keep
        row["jurisdiction_method"] = reason

        level_counts[level] = level_counts.get(level, 0) + 1
        method_counts[reason] = method_counts.get(reason, 0) + 1

    kept_count = sum(1 for row in rows if row.get("jurisdiction_keep") is True)
    print("After filtering:", kept_count)

    fieldnames = list(rows[0].keys())
    for extra in ["jurisdiction_level", "jurisdiction_keep", "jurisdiction_method"]:
        if extra not in fieldnames:
            fieldnames.append(extra)

    saved_output_path = safe_write_csv(OUTPUT_FILE, fieldnames, rows)

    summary = {
        "before_filtering": len(rows),
        "after_filtering": kept_count,
        "level_counts": level_counts,
        "method_counts": method_counts,
        "kept_count": kept_count,
        "excluded_count": sum(1 for row in rows if row.get("jurisdiction_keep") is not True),
    }

    saved_summary_path = safe_write_json(SUMMARY_FILE, summary)

    print("Saved:", saved_output_path)
    print("Summary saved:", saved_summary_path)
    print("Level summary:", level_counts)
    print("Method summary:", method_counts)


if __name__ == "__main__":
    main()
