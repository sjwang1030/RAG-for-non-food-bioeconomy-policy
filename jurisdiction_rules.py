import re
import unicodedata


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
    "Andean Community",
    "Amazon Cooperation Treaty Organization",
    "Association of Southeast Asian Nations",
    "ASCOBANS",
    "AU",
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
    "regional organization",
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
    "bolivarian republic of",
    "state of",
    "union of",
    "commonwealth of",
    "democratic republic of",
    "people's republic of",
    "united republic of",
]

COUNTRY_BASES = [
    "afghanistan",
    "albania",
    "algeria",
    "andorra",
    "angola",
    "antigua and barbuda",
    "argentina",
    "armenia",
    "australia",
    "austria",
    "azerbaijan",
    "bahamas",
    "bahrain",
    "bangladesh",
    "barbados",
    "belarus",
    "belgium",
    "belize",
    "benin",
    "bhutan",
    "bolivia",
    "bolivia (plurinational state of)",
    "bosnia and herzegovina",
    "botswana",
    "brazil",
    "brunei darussalam",
    "bulgaria",
    "burkina faso",
    "burundi",
    "cabo verde",
    "cambodia",
    "cameroon",
    "canada",
    "central african republic",
    "chad",
    "chile",
    "china",
    "colombia",
    "comoros",
    "congo",
    "costa rica",
    "cote d'ivoire",
    "croatia",
    "cuba",
    "cyprus",
    "czech republic",
    "czechia",
    "democratic people's republic of korea",
    "democratic republic of the congo",
    "denmark",
    "djibouti",
    "dominica",
    "dominican republic",
    "ecuador",
    "egypt",
    "el salvador",
    "equatorial guinea",
    "eritrea",
    "estonia",
    "eswatini",
    "ethiopia",
    "fiji",
    "finland",
    "france",
    "gabon",
    "gambia",
    "georgia",
    "germany",
    "ghana",
    "greece",
    "grenada",
    "guatemala",
    "guinea",
    "guinea-bissau",
    "guyana",
    "haiti",
    "honduras",
    "hungary",
    "iceland",
    "india",
    "indonesia",
    "iran",
    "iraq",
    "ireland",
    "israel",
    "italy",
    "jamaica",
    "japan",
    "jordan",
    "kazakhstan",
    "kenya",
    "kiribati",
    "kuwait",
    "kyrgyzstan",
    "lao people's democratic republic",
    "latvia",
    "lebanon",
    "lesotho",
    "liberia",
    "libya",
    "lithuania",
    "luxembourg",
    "madagascar",
    "malawi",
    "malaysia",
    "maldives",
    "mali",
    "malta",
    "marshall islands",
    "mauritania",
    "mauritius",
    "mexico",
    "micronesia",
    "moldova",
    "mongolia",
    "montenegro",
    "morocco",
    "mozambique",
    "myanmar",
    "namibia",
    "nauru",
    "nepal",
    "netherlands",
    "new zealand",
    "nicaragua",
    "niger",
    "nigeria",
    "north macedonia",
    "norway",
    "oman",
    "pakistan",
    "panama",
    "papua new guinea",
    "paraguay",
    "peru",
    "philippines",
    "poland",
    "portugal",
    "qatar",
    "republic of korea",
    "republic of moldova",
    "romania",
    "russian federation",
    "rwanda",
    "saint lucia",
    "saint vincent and the grenadines",
    "samoa",
    "sao tome and principe",
    "saudi arabia",
    "senegal",
    "serbia",
    "seychelles",
    "sierra leone",
    "singapore",
    "slovakia",
    "slovenia",
    "solomon islands",
    "somalia",
    "south africa",
    "south sudan",
    "spain",
    "sri lanka",
    "sudan",
    "suriname",
    "sweden",
    "switzerland",
    "tajikistan",
    "thailand",
    "timor-leste",
    "togo",
    "tonga",
    "trinidad and tobago",
    "tunisia",
    "turkiye",
    "turkmenistan",
    "tuvalu",
    "uganda",
    "uk",
    "ukraine",
    "united kingdom",
    "united kingdom of great britain and northern ireland",
    "united republic of tanzania",
    "united states",
    "united states of america",
    "uruguay",
    "usa",
    "uzbekistan",
    "vanuatu",
    "venezuela",
    "viet nam",
    "yemen",
    "zambia",
    "zimbabwe",
]

SOVEREIGN_COMMA_FORMS = [
    "bolivia, plurinational state of",
    "iran, islamic republic of",
    "korea, republic of",
    "micronesia, federated states of",
    "moldova, republic of",
    "tanzania, united republic of",
    "venezuela, bolivarian republic of",
]

SUBNATIONAL_BRACKET_NAMES = [
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

SPECIAL_JURISDICTION_KEYWORDS = [
    "american samoa",
    "aruba",
    "bermuda",
    "british virgin islands",
    "cayman islands",
    "cook islands",
    "curacao",
    "falkland islands",
    "faroe islands",
    "french polynesia",
    "gibraltar",
    "greenland",
    "guam",
    "hong kong",
    "kosovo",
    "macao",
    "macau",
    "new caledonia",
    "niue",
    "northern mariana islands",
    "puerto rico",
    "somaliland",
    "tokelau",
    "unmik",
    "unscr",
    "u.s. virgin islands",
    "virgin islands",
    "western sahara",
]

SPECIAL_NATIONAL_KEYWORDS = [
    "palestine",
]

SPECIAL_SUBNATIONAL_KEYWORDS = [
    "taiwan",
    "hong kong",
    "macao",
    "macau",
]

PAREN_RE = re.compile(r"^(.+?)\s*\(([^)]+)\)\s*$")


def normalize_text(value):
    text = str(value or "").strip()
    if not text:
        return ""

    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


COUNTRY_BASE_SET = {normalize_text(item) for item in COUNTRY_BASES}
SOVEREIGN_COMMA_SET = {normalize_text(item) for item in SOVEREIGN_COMMA_FORMS}
SUBNATIONAL_BRACKET_SET = {normalize_text(item) for item in SUBNATIONAL_BRACKET_NAMES}


def is_blank(value):
    return normalize_text(value) == ""


def region_name_matches(value, region_name):
    value_norm = normalize_text(value)
    region_norm = normalize_text(region_name)
    if not value_norm or not region_norm:
        return False

    if len(region_norm) <= 4 and re.fullmatch(r"[a-z]+", region_norm):
        return re.search(rf"\b{re.escape(region_norm)}\b", value_norm) is not None

    return region_norm in value_norm


def contains_special_jurisdiction(value):
    value_norm = normalize_text(value)
    return any(keyword in value_norm for keyword in SPECIAL_JURISDICTION_KEYWORDS)


def contains_special_national(value):
    value_norm = normalize_text(value)
    return any(keyword in value_norm for keyword in SPECIAL_NATIONAL_KEYWORDS)


def contains_special_subnational(value):
    value_norm = normalize_text(value)
    return any(keyword in value_norm for keyword in SPECIAL_SUBNATIONAL_KEYWORDS)


def contains_subnational_keyword(value):
    value_norm = normalize_text(value)
    return any(re.search(rf"\b{re.escape(keyword)}\b", value_norm) for keyword in SUBNATIONAL_KEYWORDS)


def is_country_with_formal_parenthetical(value):
    value_norm = normalize_text(value)
    match = PAREN_RE.match(value_norm)
    if not match:
        return False

    base_name = match.group(1).strip()
    bracket_name = match.group(2).strip()
    return base_name in COUNTRY_BASE_SET and any(
        qualifier in bracket_name for qualifier in SOVEREIGN_QUALIFIERS
    )


def is_sovereign_like_part(part):
    part_norm = normalize_text(part)
    if not part_norm:
        return False

    if part_norm in COUNTRY_BASE_SET or part_norm in SOVEREIGN_COMMA_SET:
        return True

    if is_country_with_formal_parenthetical(part_norm):
        return True

    return any(qualifier in part_norm for qualifier in SOVEREIGN_QUALIFIERS)


def is_macro_regional_combo_part(part):
    part_norm = normalize_text(part)
    if not part_norm:
        return False

    if is_sovereign_like_part(part_norm):
        return True

    if any(region_name_matches(part_norm, region_name) for region_name in MACRO_REGIONAL):
        return True

    return any(keyword in part_norm for keyword in MACRO_REGIONAL_KEYWORDS)


def classify_multi_part(value, separator):
    parts = [normalize_text(part) for part in value.split(separator) if normalize_text(part)]
    if len(parts) < 2:
        return None

    if any(contains_special_subnational(part) for part in parts):
        return "subnational", False

    if any(contains_special_national(part) for part in parts):
        if all(is_macro_regional_combo_part(part) or contains_special_national(part) for part in parts):
            return "macro_regional", True
        return "national", True

    if any(contains_special_jurisdiction(part) for part in parts):
        return "unclear", False

    if all(is_macro_regional_combo_part(part) for part in parts):
        return "macro_regional", True

    return "unclear", None


def rule_filter(jurisdiction):
    if is_blank(jurisdiction):
        return "unclear", None

    value = normalize_text(jurisdiction)

    if contains_special_subnational(value):
        return "subnational", False

    if contains_special_national(value):
        return "national", True

    if contains_special_jurisdiction(value):
        return "unclear", False

    if value in COUNTRY_BASE_SET or value in SOVEREIGN_COMMA_SET or is_country_with_formal_parenthetical(value):
        return "national", True

    for region_name in MACRO_REGIONAL:
        if region_name_matches(value, region_name):
            return "macro_regional", True

    for keyword in MACRO_REGIONAL_KEYWORDS:
        if keyword in value:
            return "macro_regional", True

    bracket_match = PAREN_RE.match(value)
    if bracket_match:
        base_name = bracket_match.group(1).strip()
        bracket_name = bracket_match.group(2).strip()

        if "regional organization" in bracket_name:
            return "macro_regional", True

        if contains_special_jurisdiction(base_name) or contains_special_jurisdiction(bracket_name):
            return "unclear", False

        if contains_subnational_keyword(bracket_name) or bracket_name in SUBNATIONAL_BRACKET_SET:
            return "subnational", False

        if base_name in COUNTRY_BASE_SET:
            if any(qualifier in bracket_name for qualifier in SOVEREIGN_QUALIFIERS):
                return "national", True
            if bracket_name in COUNTRY_BASE_SET or is_sovereign_like_part(bracket_name):
                return "unclear", False
            return "subnational", False

        if bracket_name in COUNTRY_BASE_SET:
            return "unclear", False

        if any(qualifier in bracket_name for qualifier in SOVEREIGN_QUALIFIERS):
            return "unclear", False

        return "unclear", None

    if ";" in value:
        result = classify_multi_part(value, ";")
        if result is not None:
            return result

    if "," in value and value not in SOVEREIGN_COMMA_SET:
        result = classify_multi_part(value, ",")
        if result is not None:
            return result

    if contains_subnational_keyword(value):
        return "subnational", False

    return "unclear", None
