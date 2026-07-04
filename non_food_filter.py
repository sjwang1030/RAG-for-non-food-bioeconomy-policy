import csv
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List

import requests
from dotenv import load_dotenv
from pypdf import PdfReader
from tqdm import tqdm

from non_food_prompt_shared import (
    NON_FOOD_PROMPT_VERSION,
    NON_FOOD_SYSTEM_PROMPT,
    build_non_food_prompt,
)


BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env", override=True)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not found in .env")

logging.getLogger("pypdf").setLevel(logging.ERROR)

DEFAULT_MODEL = "gpt-5.4-mini"
MODEL_NAME = os.getenv("OPENAI_MODEL", DEFAULT_MODEL)
CLASSIFICATION_SEED = 42
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_TIMEOUT = int(os.getenv("OPENAI_TIMEOUT", "300"))
OPENAI_MAX_RETRIES = int(os.getenv("OPENAI_MAX_RETRIES", "4"))
OPENAI_RETRY_BACKOFF = float(os.getenv("OPENAI_RETRY_BACKOFF", "2.0"))
NON_FOOD_TEST_LIMIT = int(os.getenv("NON_FOOD_TEST_LIMIT", "0"))

CATALOG_PATH = BASE_DIR / "outputs" / "jurisdiction_filter" / "policy_catalog_jurisdiction_filtered.csv"
OUTPUT_DIR = BASE_DIR / "outputs" / "non_food_filter"
OUTPUT_PATH = OUTPUT_DIR / "policy_catalog_non_food_filtered_v14.csv"
OUTPUT_ALL_PATH = OUTPUT_DIR / "policy_catalog_non_food_all_results_v14.csv"
OUTPUT_REVIEW_QUEUE_PATH = OUTPUT_DIR / "policy_catalog_non_food_review_queue_v14.csv"
OUTPUT_TECHNICAL_PATH = OUTPUT_DIR / "policy_catalog_non_food_technical_failures_v14.csv"
GOLD_SET_PATH = BASE_DIR / "src" / "non_food_gold_set.csv"
GOLD_SET_AUDIT_PATH = OUTPUT_DIR / "non_food_gold_set_audit.csv"
PDF_DIR = BASE_DIR / "faolex_policies"
RESUME_FROM_EXISTING = False
CHECKPOINT_EVERY = 1000
MAX_FRONT_PAGES = 4
MAX_KEYWORD_PAGES = 12
MAX_CHARS = 40000


def clean_bool(value):
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"true", "1", "yes", "y"}


def load_existing_output(path):
    if not path.exists():
        return {}

    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    existing = {}
    for row in rows:
        policy_id = row.get("policy_id", "")
        category = row.get("category", "").strip()
        prompt_version = row.get("prompt_version", "")
        if policy_id and category and prompt_version == NON_FOOD_PROMPT_VERSION:
            existing[policy_id] = {
                "policy_id": policy_id,
                "file_path": row.get("file_path", ""),
                "category": category,
                "confidence": row.get("confidence", ""),
                "reason": row.get("reason", ""),
                "model_reason": row.get("model_reason", ""),
                "standardized_reason": row.get("standardized_reason", ""),
                "evidence": row.get("evidence", ""),
                "basis": row.get("basis", ""),
                "matched_terms": row.get("matched_terms", ""),
                "prompt_version": prompt_version,
                "requested_model": row.get("requested_model", ""),
                "response_model": row.get("response_model", ""),
                "system_fingerprint": row.get("system_fingerprint", ""),
                "title_scope_cleaning_decision": row.get("title_scope_cleaning_decision", ""),
                "title_scope_cleaning_rule": row.get("title_scope_cleaning_rule", ""),
                "title_scope_cleaning_terms": row.get("title_scope_cleaning_terms", ""),
            }
    return existing


def load_existing_results():
    existing = {}
    for path in [OUTPUT_ALL_PATH, OUTPUT_REVIEW_QUEUE_PATH, OUTPUT_TECHNICAL_PATH, OUTPUT_PATH]:
        for policy_id, row in load_existing_output(path).items():
            if policy_id not in existing:
                existing[policy_id] = row
    return existing


def load_gold_set(path: Path = GOLD_SET_PATH) -> List[Dict[str, str]]:
    if not path.exists():
        return []

    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def normalize_gold_expected(value: str) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in {"contains", "contains_non_food", "true"}:
        return "contains_non_food"
    if normalized in {"no", "no_non_food", "false"}:
        return "no_non_food"
    return normalized


def build_gold_set_audit(rows: List[Dict[str, Any]], gold_rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    row_map = {str(row.get("policy_id", "")).strip(): row for row in rows}
    audit_rows = []

    for gold in gold_rows:
        policy_id = str(gold.get("policy_id", "")).strip()
        actual = row_map.get(policy_id, {})
        expected_category = normalize_gold_expected(gold.get("expected_category", ""))
        actual_category = str(actual.get("category", "")).strip()

        audit_rows.append(
            {
                "policy_id": policy_id,
                "label": str(gold.get("label", "")).strip(),
                "expected_category": expected_category,
                "actual_category": actual_category,
                "pass": str(actual_category == expected_category).lower(),
                "confidence": str(actual.get("confidence", "")).strip(),
                "basis": str(actual.get("basis", "")).strip(),
                "matched_terms": str(actual.get("matched_terms", "")).strip(),
                "reason": str(actual.get("reason", "")).strip(),
                "prompt_version": str(actual.get("prompt_version", "")).strip(),
            }
        )

    return audit_rows


def save_gold_set_audit(rows: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    gold_rows = load_gold_set()
    if not gold_rows:
        return []

    audit_rows = build_gold_set_audit(rows, gold_rows)
    fieldnames = [
        "policy_id",
        "label",
        "expected_category",
        "actual_category",
        "pass",
        "confidence",
        "basis",
        "matched_terms",
        "reason",
        "prompt_version",
    ]
    write_output_csv(GOLD_SET_AUDIT_PATH, fieldnames, audit_rows)
    return audit_rows


def resolve_pdf_path(row):
    existing = row.get("file_path", "")
    if existing and os.path.exists(existing):
        return existing

    policy_id = str(row.get("policy_id", "")).strip()
    faolex_no = str(row.get("faolex_no", "")).strip().lower()
    title = str(row.get("title", "")).strip().lower()
    title_tokens = [t for t in re.split(r"\W+", title) if len(t) >= 4][:6]

    if policy_id:
        matches = list(PDF_DIR.glob(f"{policy_id}_*.pdf"))
        if matches:
            return str(matches[0])

    for pdf_file in PDF_DIR.glob("*.pdf"):
        name = pdf_file.name.lower()
        if faolex_no and faolex_no in name:
            return str(pdf_file)
        overlap = sum(1 for token in title_tokens if token in name)
        if overlap >= 3:
            return str(pdf_file)

    return ""


KEY_TERMS = [
    "biomass", "bio-based", "biobased", "bioeconomy", "bio-economy",
    "biofuel", "biofuels", "biogas", "biomethane", "bioenergy",
    "bioethanol", "biodiesel", "sustainable aviation fuel", "sustainable aviation fuels",
    "biomass-based fuel", "biomass-based fuels", "renewable fuel of biological origin",
    "advanced biofuels", "renewable fuels", "transport fuel", "transport fuels",
    "aviation fuels", "maritime fuels", "bioliquids", "biomass fuels",
    "renewable energy", "renewable energy directive", "red ii", "red iii",
    "bioplastic", "bioplastics", "biochemical", "biochemicals",
    "industrial biotechnology", "biomanufacturing", "biorefinery", "biorefineries",
    "organic waste", "agricultural residues", "forestry residues", "wood residues",
    "sawdust", "black liquor", "bagasse", "straw", "crop residues",
    "biodegradable", "feedstock", "renewable carbon", "biomaterial",
    "biomaterials", "bio-based materials", "bio-based products", "bio-based chemical",
    "wood-based", "wood-based materials", "timber construction", "mass timber",
    "pulp", "paper", "cellulose", "lignin", "hemicellulose",
    "anaerobic digestion", "fermentation", "enzymes", "cascading use",
    "valorization", "valorisation", "renewable biological resources",
    "biological resources", "biogenic carbon", "biogenic co2",
]

STRONG_EVIDENCE_TERMS = [
    "biofuel", "biofuels", "biogas", "biomethane", "bioethanol", "biodiesel",
    "sustainable aviation fuel", "renewable fuel of biological origin",
    "bio-based products", "bio-based materials", "wood-based materials",
    "bioplastic", "bioplastics", "biochemical", "biochemicals",
    "industrial biotechnology", "biomanufacturing", "biorefinery", "biorefineries",
    "anaerobic digestion", "fermentation", "biomass residue", "agricultural residues",
    "forestry residues", "feedstock", "valorization", "valorisation",
    "renewable biological resources", "cascading use", "mass timber",
]

POLICY_ACTION_TERMS = [
    "support", "promote", "promotion", "incentive", "subsidy", "subsidies",
    "grant", "fund", "funding", "programme", "program", "strategy", "action plan",
    "roadmap", "target", "mandate", "obligation", "requirement", "certification",
    "standard", "procurement", "quota", "deployment",
]

EXCLUSION_CONTEXT_TERMS = [
    "food security", "nutrition", "food production", "feeding", "feed use",
    "fishery management", "catch limit", "biodiversity conservation",
]

TITLE_SCOPE_SUBNATIONAL_TERMS = [
    "province",
    "provincial",
    "island",
    "district",
    "city",
    "urban planning",
    "rural residential planning",
    "mekong delta key economic region",
]

TITLE_SCOPE_SUBNATIONAL_REVIEW_TERMS = [
    "national",
    "nationwide",
    "countrywide",
    "regional development",
    "economic region",
    "key economic region",
    "socio economic development",
    "socioeconomic development",
    "spatial planning",
]

TITLE_SCOPE_FOSSIL_ENERGY_TERMS = [
    "petroleum",
    "petrol",
    "oil reserves",
    "oil reserve",
    "oil reserves agency",
    "oil and gas",
    "coal",
    "natural gas",
    "lpg",
    "specified fuels",
    "atomic energy",
]

TITLE_SCOPE_SITE_CONSERVATION_TERMS = [
    "conservation of wild birds",
    "wild birds",
    "special protection area",
    "special area of conservation",
]

TITLE_SCOPE_PROCEDURAL_TERMS = [
    "agriculture appeals act",
    "amendment of schedule",
]

TITLE_SCOPE_PLANNING_TERMS = [
    "planning and development regulations",
    "national planning framework",
    "maritime area planning",
]

TITLE_SCOPE_FOOD_AGRI_TERMS = [
    "food 2030",
    "food wise",
    "food vision",
    "food system",
    "food systems",
    "food policy",
    "food strategy",
    "food programme",
    "food program",
    "national food",
    "food safety",
    "safeguarding food",
    "rice industry",
    "livestock",
    "animal husbandry",
    "animal feed",
    "animal by-product",
    "animal by-products",
    "poultry",
    "cattle",
    "dairy",
    "swine",
    "fodder",
    "dried fodder",
    "feed material",
    "feed materials",
    "feed use",
    "fertilizer management",
    "agricultural production development",
    "forest seeds",
    "crop protection",
    "plant protection",
    "pesticide",
    "fisheries",
    "fishery",
    "fisheries domain",
    "aquaculture",
    "fish stock",
    "catch limit",
    "food security",
    "biodiversity conservation",
    "species conservation",
    "habitat conservation",
]

TITLE_SCOPE_GENERAL_LAW_TERMS = [
    "land law",
    "electricity law",
    "national master plan",
    "environmental protection tax",
    "general building code",
    "general water drainage and wastewater treatment",
    "pops",
    "hazardous pollutants",
    "raised bears",
]

COUNTING_GROUP_PATTERNS = [
    (
        re.compile(r"\b(european communities|european union)\s+renewable energy and biofuel sustainability criteria\b"),
        "renewable_energy_biofuel_sustainability_framework",
    ),
    (
        re.compile(r"\bbiofuel sustainability criteria\b"),
        "biofuel_sustainability_framework",
    ),
    (
        re.compile(r"\bgreenhouse gas emission reductions calculation methods and reporting requirements\b"),
        "greenhouse_gas_reduction_reporting_framework",
    ),
    (
        re.compile(r"\bsustainable energy act\b.*\brenewable energy\b"),
        "sustainable_energy_renewable_functions_framework",
    ),
    (
        re.compile(r"\b(european communities|european union)\s+renewable energy\b"),
        "renewable_energy_framework",
    ),
]

TITLE_SCOPE_OVERRIDE_STRONG_TERMS = [
    "biofuel",
    "biofuels",
    "biogas",
    "biomethane",
    "bioethanol",
    "biodiesel",
    "biomass energy",
    "bio-based materials",
    "bio-based products",
    "bioplastics",
    "biochemicals",
    "industrial biotechnology",
    "biomanufacturing",
    "biorefinery",
    "biorefineries",
    "wood-based materials",
    "timber construction",
    "mass timber",
]

TITLE_SCOPE_OVERRIDE_OBJECT_TERMS = [
    "biomass",
    "bio-based",
    "biobased",
    "bioeconomy",
    "bio-economy",
    "renewable biological resources",
    "biological resources",
    "organic waste",
    "agricultural residues",
    "forestry residues",
    "wood residues",
    "crop residues",
    "feedstock",
    "cellulose",
    "lignin",
]

TITLE_SCOPE_OVERRIDE_CORE_BIOMASS_TERMS = [
    "biomass",
    "bio-based",
    "biobased",
    "bioeconomy",
    "bio-economy",
    "agricultural residues",
    "forestry residues",
    "wood residues",
    "crop residues",
    "cellulose",
    "lignin",
]

TITLE_SCOPE_OVERRIDE_EXPLICIT_NONFOOD_USE_TERMS = [
    "bioenergy",
    "biofuel",
    "biofuels",
    "biogas",
    "biomethane",
    "bioethanol",
    "biodiesel",
    "biorefinery",
    "biorefineries",
    "bio-based material",
    "bio-based materials",
    "bio-based product",
    "bio-based products",
    "bio-based chemical",
    "bio-based chemicals",
    "biochemical",
    "biochemicals",
    "non-food use",
    "non food use",
    "non-food purpose",
    "non food purpose",
    "industrial biotechnology",
    "biomanufacturing",
    "biomass-based fuel",
    "biomass-based energy",
    "energy production",
    "renewable fuel",
    "renewable fuels",
    "valorization for energy",
    "valorisation for energy",
]

OVERRIDE_EXCLUSION_SIGNALS = [
    "food security",
    "food production",
    "food system",
    "animal feed",
    "feed use",
    "fodder",
    "fish",
    "fishery",
    "fisheries",
    "aquaculture",
    "livestock",
    "poultry",
    "nutrition",
    "biodiversity conservation",
    "habitat",
    "catch limit",
    "animal by-product",
]

PROTECTED_AREA_CONTEXT_TERMS = [
    "wild birds",
    "habitats",
    "special protection area",
    "special area of conservation",
    "protected area",
    "conservation objectives",
]

PROTECTED_AREA_BIOENERGY_ACTIVITY_TERMS = [
    "bioenergy crops",
    "multi-annual bioenergy crops",
    "planting of trees",
    "tree planting",
]

PROTECTED_AREA_RESTRICTION_TERMS = [
    "prior written consent",
    "written consent of the minister",
    "requiring consent",
    "operations or activities requiring consent",
    "requires consent",
    "consent of the minister",
    "schedule 4",
    "arc 29",
    "restricted activity",
    "prohibited activity",
]

RETROSPECTIVE_REPORTING_TERMS = [
    "past five years",
    "over the past",
    "during the past",
    "during the previous",
    "in recent years",
    "has increased",
    "have increased",
    "has reached",
    "have reached",
    "had reached",
    "increased financial support",
    "rose to",
    "grew to",
    "totaled",
    "accounted for",
    "as of the end of",
    "by the end of",
    "progress made",
    "achievements",
    "was built",
    "were built",
    "was installed",
    "were installed",
    "has been built",
    "have been built",
    "capacity reached",
]

REPORTING_DOCUMENT_TERMS = [
    "white paper",
    "report",
    "progress",
    "review",
    "communication",
    "policies and actions",
]

ASPIRATIONAL_DOCUMENT_TERMS = [
    "strategy",
    "vision",
    "communication",
    "white paper",
    "roadmap",
]

ASPIRATIONAL_FALSE_POSITIVE_TERMS = [
    "untapped potential",
    "potential to",
    "have the potential to",
    "offer opportunities",
    "opportunities for",
    "should grasp opportunities",
]

FORWARD_LOOKING_GOVERNANCE_TERMS = [
    "target",
    "targets",
    "by 2020",
    "by 2025",
    "by 2030",
    "will",
    "plan",
    "five-year",
    "priority task",
    "priority tasks",
    "engineering",
    "construction project",
    "construction projects",
    "demonstration",
    "demonstration program",
    "demonstration programmes",
    "demonstration project",
    "research and development",
    "r&d",
    "technology development",
    "promote",
    "promoting",
    "develop",
    "development of",
    "build",
    "build out",
]

STRONG_FORWARD_LOOKING_CONTEXT_TERMS = [
    "target",
    "targets",
    "shall",
    "must",
    "will",
    "plan",
    "plans",
    "program",
    "programme",
    "strategy",
    "roadmap",
    "five-year",
    "by 2020",
    "by 2025",
    "by 2030",
    "objective",
    "objectives",
    "goal",
    "goals",
    "规划",
    "方案",
    "战略",
    "目标",
    "将",
]

PLAN_DOCUMENT_TERMS = [
    "plan",
    "program",
    "programme",
    "strategy",
    "roadmap",
    "five-year",
    "规划",
    "方案",
    "战略",
]

NONFOOD_PLAN_OBJECT_TERMS = [
    "biomass energy",
    "bioenergy",
    "biogas",
    "biofuel",
    "biofuels",
    "biomethane",
    "agricultural residues",
    "forestry residues",
    "crop residues",
    "straw",
    "organic waste",
    "resource utilization",
    "resource recovery",
    "bio-based materials",
    "生物能源",
    "沼气",
    "秸秆",
    "资源化利用",
    "畜禽养殖废弃物",
    "生物制造",
    "生物经济",
]

NONFOOD_PLAN_ACTION_TERMS = [
    "project",
    "projects",
    "program",
    "programme",
    "demonstration",
    "research and development",
    "r&d",
    "technology development",
    "construction",
    "engineering",
    "deployment",
    "建设",
    "工程",
    "示范",
    "研发",
    "推广",
    "技术",
]

NONFOOD_PLAN_QUANT_TARGET_TERMS = [
    "by 2020",
    "by 2025",
    "by 2030",
    "shall reach",
    "will reach",
    "reach above",
    "account for",
    "达到",
    "以上",
    "利用率",
    "目标",
]

PLANNING_FALSE_POSITIVE_TITLE_TERMS = [
    "planning",
    "territory",
    "energy performance of a building",
    "building",
    "spatial",
    "methodology",
    "calculation",
]

PLANNING_FALSE_POSITIVE_MATCH_TERMS = [
    "biomass growing",
    "biogas cogeneration units",
    "wind power stations",
    "wood processing",
    "biomass heating boiler",
    "boiler or installation",
    "wood",
]

PLANNING_FALSE_POSITIVE_CONTEXT_TERMS = [
    "permitted agricultural activities",
    "large engineering structures",
    "planning requirements",
    "energy performance",
    "methodology",
    "calculation",
    "coefficient",
    "parameter",
    "zone",
    "territory",
    "installation",
    "boiler",
]

OPERATIVE_PROVISION_TERMS = [
    "shall",
    "must",
    "required to",
    "requirement",
    "requirements",
    "obligation",
    "obligations",
    "target",
    "targets",
    "mandate",
    "mandates",
    "subsidy",
    "subsidies",
    "grant",
    "grants",
    "quota",
    "certification",
    "standard",
    "standards",
    "support scheme",
    "article ",
    "section ",
    "regulation ",
    "regulations ",
]


def normalize_text(s: str) -> str:
    s = s.replace("\u00a0", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def normalize_match_text(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(s or "").lower()).strip()


def find_matched_terms(text: str, terms: List[str]) -> List[str]:
    normalized = normalize_match_text(text)
    matched = []
    for term in terms:
        normalized_term = normalize_match_text(term)
        if normalized_term and re.search(rf"(?<!\w){re.escape(normalized_term)}(?!\w)", normalized):
            matched.append(term)
    return matched


def has_national_regional_strategy_context(title: str) -> bool:
    return bool(find_matched_terms(title, TITLE_SCOPE_SUBNATIONAL_REVIEW_TERMS))


def extract_override_contexts(text: str) -> List[str]:
    lower = str(text or "").lower()
    parts = re.split(r"(?<=[\.\?!;:])\s+|\n+", lower)
    return [part.strip() for part in parts if part.strip()]


def sentence_has_exclusion_signal(sentence: str) -> bool:
    return any(term in sentence for term in OVERRIDE_EXCLUSION_SIGNALS)


def has_clear_non_food_biomass_override(text: str) -> bool:
    lower = str(text or "").lower()

    if any(term in lower for term in TITLE_SCOPE_OVERRIDE_STRONG_TERMS):
        return True

    for context in extract_override_contexts(lower):
        has_nonfood_label = "non-food" in context or "non food" in context
        has_object = any(term in context for term in TITLE_SCOPE_OVERRIDE_OBJECT_TERMS)
        if has_nonfood_label and has_object and not sentence_has_exclusion_signal(context):
            return True

        has_core_object = any(term in context for term in TITLE_SCOPE_OVERRIDE_CORE_BIOMASS_TERMS)
        has_explicit_use = any(term in context for term in TITLE_SCOPE_OVERRIDE_EXPLICIT_NONFOOD_USE_TERMS)
        if has_core_object and has_explicit_use and not sentence_has_exclusion_signal(context):
            return True

    return False


def has_protected_area_bioenergy_false_positive(
    title: str,
    text: str,
    matched_terms: List[str],
) -> bool:
    title_lower = str(title or "").lower()
    text_lower = str(text or "").lower()
    matched_lower = " | ".join(str(term or "").lower() for term in matched_terms)
    combined = " || ".join(part for part in [title_lower, text_lower, matched_lower] if part)

    has_conservation_context = any(term in combined for term in PROTECTED_AREA_CONTEXT_TERMS)
    has_bioenergy_activity = any(term in combined for term in PROTECTED_AREA_BIOENERGY_ACTIVITY_TERMS)
    has_restriction_context = any(term in combined for term in PROTECTED_AREA_RESTRICTION_TERMS)

    return has_conservation_context and has_bioenergy_activity and has_restriction_context


def has_retrospective_reporting_false_positive(
    title: str,
    text: str,
    matched_terms: List[str],
) -> bool:
    title_lower = str(title or "").lower()
    text_lower = str(text or "").lower()
    matched_lower = [str(term or "").lower() for term in matched_terms if str(term or "").strip()]

    contexts = extract_override_contexts(text_lower)
    if matched_lower:
        focus_contexts = [
            context
            for context in contexts
            if any(term in context for term in matched_lower)
        ]
    else:
        focus_contexts = contexts

    if not focus_contexts:
        focus_contexts = contexts

    retrospective_focus = any(
        any(retro in context for retro in RETROSPECTIVE_REPORTING_TERMS)
        and not any(op in context for op in OPERATIVE_PROVISION_TERMS)
        for context in focus_contexts
    )
    forward_looking_focus = any(
        any(term in context for term in STRONG_FORWARD_LOOKING_CONTEXT_TERMS)
        for context in focus_contexts
    )
    reporting_document = any(term in title_lower for term in REPORTING_DOCUMENT_TERMS)
    dedicated_clause = bool(re.search(r"\b(article|section|regulation)\s+\d+\b", text_lower))

    return retrospective_focus and not forward_looking_focus and (reporting_document or not dedicated_clause)


def has_aspirational_nonfood_false_positive(
    title: str,
    text: str,
    matched_terms: List[str],
) -> bool:
    title_lower = str(title or "").lower()
    text_lower = str(text or "").lower()
    matched_lower = [str(term or "").lower() for term in matched_terms if str(term or "").strip()]

    contexts = extract_override_contexts(text_lower)
    if matched_lower:
        focus_contexts = [
            context
            for context in contexts
            if any(term in context for term in matched_lower)
        ]
    else:
        focus_contexts = contexts

    broad_document = any(term in title_lower for term in ASPIRATIONAL_DOCUMENT_TERMS)
    aspirational_focus = any(
        any(term in context for term in ASPIRATIONAL_FALSE_POSITIVE_TERMS)
        and not any(op in context for op in OPERATIVE_PROVISION_TERMS)
        for context in focus_contexts
    )
    no_operational_focus = focus_contexts and all(
        not any(op in context for op in OPERATIVE_PROVISION_TERMS)
        for context in focus_contexts
    )

    return bool(broad_document and aspirational_focus and no_operational_focus)


def has_forward_looking_nonfood_plan_override(
    title: str,
    text: str,
    matched_terms: List[str],
) -> bool:
    title_lower = str(title or "").lower()
    text_lower = str(text or "").lower()
    matched_lower = " | ".join(str(term or "").lower() for term in matched_terms)
    combined = " || ".join(part for part in [title_lower, text_lower, matched_lower] if part)

    has_plan_document = any(term in title_lower for term in PLAN_DOCUMENT_TERMS)
    has_nonfood_object = any(term in combined for term in NONFOOD_PLAN_OBJECT_TERMS)
    has_target_signal = any(term in combined for term in NONFOOD_PLAN_QUANT_TARGET_TERMS) or bool(
        re.search(r"(\d+\s*%|\d+\s*％)", combined)
    )
    has_action_signal = any(term in combined for term in NONFOOD_PLAN_ACTION_TERMS)

    return has_plan_document and has_nonfood_object and has_target_signal and (
        has_action_signal or "资源化利用率" in combined or "沼气用户" in combined
    )


def has_planning_list_false_positive(
    title: str,
    text: str,
    matched_terms: List[str],
) -> bool:
    title_lower = str(title or "").lower()
    text_lower = str(text or "").lower()
    matched_lower = " | ".join(str(term or "").lower() for term in matched_terms)
    combined = " || ".join(part for part in [title_lower, text_lower, matched_lower] if part)

    has_title_context = any(term in title_lower for term in PLANNING_FALSE_POSITIVE_TITLE_TERMS)
    has_listed_match = any(term in matched_lower for term in PLANNING_FALSE_POSITIVE_MATCH_TERMS)
    has_list_context = any(term in combined for term in PLANNING_FALSE_POSITIVE_CONTEXT_TERMS)

    return has_title_context and has_listed_match and has_list_context


def apply_post_classification_overrides(
    title: str,
    text: str,
    category: str,
    basis: str,
    matched_terms: List[str],
) -> tuple[str, str]:
    if has_protected_area_bioenergy_false_positive(title, text, matched_terms):
        return "no_non_food", "incidental_or_excluded_mention"

    if has_retrospective_reporting_false_positive(title, text, matched_terms):
        return "no_non_food", "incidental_or_excluded_mention"

    if has_aspirational_nonfood_false_positive(title, text, matched_terms):
        return "no_non_food", "incidental_or_excluded_mention"

    if has_planning_list_false_positive(title, text, matched_terms):
        return "no_non_food", "incidental_or_excluded_mention"

    if category == "contains_non_food" and basis in {
        "incidental_or_excluded_mention",
        "no_relevant_signal",
        "mixed_or_borderline",
    }:
        fallback_basis = "no_relevant_signal" if basis == "no_relevant_signal" else "incidental_or_excluded_mention"
        return "no_non_food", fallback_basis

    if (
        category == "no_non_food"
        and basis in {"no_relevant_signal", "mixed_or_borderline", "incidental_or_excluded_mention"}
        and has_forward_looking_nonfood_plan_override(title, text, matched_terms)
    ):
        return "contains_non_food", "substantive_broad_framework"

    return category, basis


def assess_title_scope(title: str) -> Dict[str, Any]:
    subnational_terms = find_matched_terms(title, TITLE_SCOPE_SUBNATIONAL_TERMS)
    fossil_terms = find_matched_terms(title, TITLE_SCOPE_FOSSIL_ENERGY_TERMS)
    site_conservation_terms = find_matched_terms(title, TITLE_SCOPE_SITE_CONSERVATION_TERMS)
    procedural_terms = find_matched_terms(title, TITLE_SCOPE_PROCEDURAL_TERMS)
    planning_terms = find_matched_terms(title, TITLE_SCOPE_PLANNING_TERMS)
    food_agri_terms = find_matched_terms(title, TITLE_SCOPE_FOOD_AGRI_TERMS)
    general_law_terms = find_matched_terms(title, TITLE_SCOPE_GENERAL_LAW_TERMS)

    if fossil_terms:
        return {
            "decision": "needs_text_check",
            "rule": "possible_fossil_or_general_energy_domain",
            "matched_terms": fossil_terms,
            "reason": (
                "Title suggests a fossil or general energy domain; full text is required to verify "
                "whether it contains biogas, biomethane, biofuel, or other non-food bio-based provisions."
            ),
        }

    if site_conservation_terms:
        return {
            "decision": "needs_text_check",
            "rule": "possible_site_specific_biodiversity_conservation",
            "matched_terms": site_conservation_terms,
            "reason": (
                "Title indicates a site-specific wild-birds or habitats conservation instrument; "
                "full text is required before excluding it from the non-food screen."
            ),
        }

    if procedural_terms:
        return {
            "decision": "needs_text_check",
            "rule": "possible_procedural_or_schedule_amendment",
            "matched_terms": procedural_terms,
            "reason": (
                "Title indicates a procedural or schedule-only amendment; full text is required to verify "
                "whether the instrument also carries substantive non-food bioeconomy provisions."
            ),
        }

    if subnational_terms:
        decision = "review" if has_national_regional_strategy_context(title) else "exclude"
        reason = (
            "Title suggests a national-level regional development strategy that may still need manual review."
            if decision == "review"
            else "Title indicates a subnational scope that should be excluded from the national non-food filter."
        )
        return {
            "decision": decision,
            "rule": "subnational_scope",
            "matched_terms": subnational_terms,
            "reason": reason,
        }

    if planning_terms:
        return {
            "decision": "needs_text_check",
            "rule": "planning_or_spatial_framework",
            "matched_terms": planning_terms,
            "reason": (
                "Title indicates a planning or spatial framework that should count only if the document "
                "contains explicit non-food biomass provisions."
            ),
        }

    if food_agri_terms:
        return {
            "decision": "needs_text_check",
            "rule": "food_feed_agriculture_only",
            "matched_terms": food_agri_terms,
            "reason": "Title indicates a food/feed/agriculture-only policy unless the document clearly contains non-food biomass use.",
        }

    if general_law_terms:
        return {
            "decision": "needs_text_check",
            "rule": "general_legal_institutional_law",
            "matched_terms": general_law_terms,
            "reason": "Title indicates a general legal or institutional law unless the document clearly contains non-food bio-based provisions.",
        }

    return {
        "decision": "pass",
        "rule": "",
        "matched_terms": [],
        "reason": "",
    }


def count_term_hits(text: str, terms: List[str]) -> int:
    return sum(1 for term in terms if term in text)


def page_priority_score(text: str) -> int:
    lower = text.lower()
    strong_hits = count_term_hits(lower, STRONG_EVIDENCE_TERMS)
    keyword_hits = count_term_hits(lower, KEY_TERMS)
    action_hits = count_term_hits(lower, POLICY_ACTION_TERMS)
    exclusion_hits = count_term_hits(lower, EXCLUSION_CONTEXT_TERMS)

    score = (strong_hits * 8) + (keyword_hits * 3) + (action_hits * 2) - exclusion_hits

    if strong_hits and action_hits:
        score += 6

    if "biomass" in lower and any(
        term in lower for term in ["energy", "fuel", "materials", "chemicals", "industrial", "manufacturing"]
    ):
        score += 6

    if any(
        term in lower
        for term in [
            "bio-based products",
            "renewable biological resources",
            "wood-based materials",
            "biorefinery",
            "biogas",
            "biomethane",
            "biofuels",
        ]
    ):
        score += 8

    return score


def extract_evidence_snippets(page_texts: List[tuple[int, str]], max_snippets: int = 6) -> str:
    candidates = []
    seen = set()

    for page_no, text in page_texts:
        parts = re.split(r"(?<=[\.\?!;])\s+|\n+", text)
        for part in parts:
            snippet = normalize_text(part)
            if len(snippet) < 40:
                continue
            score = page_priority_score(snippet)
            if score < 8:
                continue
            normalized = snippet.lower()
            if normalized in seen:
                continue
            seen.add(normalized)
            candidates.append((score, page_no, snippet[:400]))

    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    selected = [f"p.{page_no}: {snippet}" for _, page_no, snippet in candidates[:max_snippets]]
    return " | ".join(selected)


def extract_pdf_text(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        front_blocks = []
        keyword_blocks = []
        seen_pages = set()
        scored_pages = []
        front_page_texts = []

        for i, page in enumerate(reader.pages[:MAX_FRONT_PAGES]):
            txt = normalize_text(page.extract_text() or "")
            if txt:
                front_blocks.append(f"[Front page {i + 1}]\n{txt}")
                seen_pages.add(i)
                front_page_texts.append((i + 1, txt))

        for i, page in enumerate(reader.pages):
            txt = normalize_text(page.extract_text() or "")
            if not txt:
                continue
            score = page_priority_score(txt)
            if score > 0:
                scored_pages.append((score, i + 1, txt))

        scored_pages.sort(key=lambda item: (-item[0], item[1]))
        selected_evidence_pages = []
        for score, page_no, txt in scored_pages:
            page_index = page_no - 1
            if page_index in seen_pages:
                continue
            keyword_blocks.append(f"[Evidence page {page_no} | score={score}]\n{txt}")
            selected_evidence_pages.append((page_no, txt))
            seen_pages.add(page_index)
            if len(keyword_blocks) >= MAX_KEYWORD_PAGES:
                break

        text = "\n\n".join(front_blocks + keyword_blocks).strip()
        evidence = extract_evidence_snippets(selected_evidence_pages)
        if not evidence:
            evidence = extract_evidence_snippets(front_page_texts)
        return {
            "text": text[:MAX_CHARS],
            "evidence": evidence[:2500],
        }

    except Exception as exc:
        return {
            "text": "",
            "evidence": "",
            "technical_error": f"[PDF_READ_ERROR] {exc}",
        }


def post_openai_chat_completion(payload: Dict[str, Any]) -> Dict[str, Any]:
    url = f"{OPENAI_BASE_URL.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
        "Connection": "close",
    }

    last_error = None
    for attempt in range(1, OPENAI_MAX_RETRIES + 1):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=OPENAI_TIMEOUT)

            if response.status_code in {429, 500, 502, 503, 504}:
                raise requests.HTTPError(
                    f"retryable_status={response.status_code} body={response.text[:500]}",
                    response=response,
                )

            response.raise_for_status()
            return response.json()
        except (
            requests.exceptions.SSLError,
            requests.exceptions.Timeout,
            requests.exceptions.ConnectionError,
            requests.exceptions.HTTPError,
            requests.exceptions.JSONDecodeError,
            ValueError,
        ) as exc:
            last_error = exc
            if attempt >= OPENAI_MAX_RETRIES:
                break
            time.sleep(OPENAI_RETRY_BACKOFF * attempt)

    raise RuntimeError(f"openai_request_failed_after_{OPENAI_MAX_RETRIES}_attempts: {last_error}")


def classify_policy(text: str) -> Dict[str, Any]:
    prompt = build_non_food_prompt(text)
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": NON_FOOD_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "top_p": 1,
        "seed": CLASSIFICATION_SEED,
        "response_format": {"type": "json_object"},
    }

    data = {}
    raw_text = ""
    for attempt in range(1, OPENAI_MAX_RETRIES + 1):
        data = post_openai_chat_completion(payload)
        raw_text = data["choices"][0]["message"]["content"] or ""
        try:
            parsed = json.loads(raw_text)
            break
        except Exception:
            match = re.search(r"(\{.*\})", str(raw_text), flags=re.S)
            if match:
                parsed = json.loads(match.group(1))
                break
            if attempt >= OPENAI_MAX_RETRIES:
                raise ValueError(f"Could not parse model output after {OPENAI_MAX_RETRIES} attempts: {raw_text[:1000]}")
            time.sleep(OPENAI_RETRY_BACKOFF * attempt)

    return {
        "parsed": parsed,
        "response_model": data.get("model", "") or "",
        "system_fingerprint": data.get("system_fingerprint", "") or "",
    }


def normalize_category(category: str) -> str:
    value = str(category or "").strip().lower()
    if value == "contains_non_food":
        return "contains_non_food"
    if value == "no_non_food":
        return "no_non_food"
    return "unclear_content"


def normalize_basis(value: str) -> str:
    normalized = str(value or "").strip().lower()
    allowed = {
        "explicit_nonfood_target",
        "substantive_broad_framework",
        "incidental_or_excluded_mention",
        "no_relevant_signal",
        "mixed_or_borderline",
        "insufficient_text_or_ambiguous",
    }
    if normalized in allowed:
        return normalized
    return "mixed_or_borderline"


def normalize_matched_terms(value: Any) -> List[str]:
    if isinstance(value, list):
        raw_items = value
    elif isinstance(value, str):
        raw_items = re.split(r"[|,;]\s*", value)
    else:
        raw_items = []

    cleaned = []
    seen = set()
    for item in raw_items:
        term = normalize_text(str(item or ""))
        if not term:
            continue
        lowered = term.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        cleaned.append(term[:80])
        if len(cleaned) >= 3:
            break
    return cleaned


def deterministic_confidence(category: str, basis: str) -> str:
    if category in {"unclear_content", "unclear_technical"}:
        return "low"

    if category == "contains_non_food":
        if basis == "explicit_nonfood_target":
            return "high"
        if basis == "substantive_broad_framework":
            return "medium"
        return "low"

    if category == "no_non_food":
        if basis in {"incidental_or_excluded_mention", "no_relevant_signal"}:
            return "high"
        if basis == "mixed_or_borderline":
            return "low"
        return "medium"

    return "low"


def deterministic_reason(category: str, basis: str, matched_terms: List[str]) -> str:
    if category == "contains_non_food":
        if basis == "explicit_nonfood_target":
            base = "The text contains a direct non-food bio-based policy object together with a concrete policy measure or governance provision."
        elif basis == "substantive_broad_framework":
            base = "The document is broad, but it contains at least one concrete policy measure explicitly directed at non-food bio-based deployment or governance."
        else:
            base = "The text contains some non-food bio-based policy content, but the evidence is not strongly specific."
    elif category == "no_non_food":
        if basis == "incidental_or_excluded_mention":
            base = "The text contains only incidental, descriptive, or excluded mentions and does not establish a direct substantive non-food bio-based policy link."
        elif basis == "no_relevant_signal":
            base = "The text does not contain a meaningful non-food bio-based policy signal."
        elif basis == "mixed_or_borderline":
            base = "The text contains some potentially relevant signals, but the policy link to non-food bio-based use is not clearly proven."
        else:
            base = "The text does not contain direct substantive non-food bioeconomy policy content."
    else:
        base = "The text is too incomplete, poor in quality, or genuinely unresolved to classify reliably."

    if matched_terms:
        return f"{base} Evidence phrases: {', '.join(matched_terms)}."
    return base


def include_type_value(category: str, basis: str = "") -> str:
    if category != "contains_non_food":
        return ""
    if basis == "explicit_nonfood_target":
        return "core"
    if basis == "substantive_broad_framework":
        return "related"
    return "review"


def include_non_food_value(category: str, basis: str = ""):
    if category == "unclear_content":
        return "review"

    if category == "unclear_technical":
        return ""

    if category == "contains_non_food":
        return "true"

    if category == "no_non_food":
        return "false"

    return ""


def review_status_value(category: str, basis: str = "") -> str:
    if category == "unclear_content":
        return "review_queue"

    if category == "unclear_technical":
        return "technical_failure"

    return "final_sample"


def parse_policy_year(value: Any) -> int:
    match = re.search(r"\b(19|20)\d{2}\b", str(value or ""))
    if not match:
        return -1
    return int(match.group(0))


def build_counting_group_key(jurisdiction: str, title: str) -> str:
    normalized_title = normalize_match_text(title)
    if not normalized_title:
        return ""

    for pattern, key in COUNTING_GROUP_PATTERNS:
        if pattern.search(normalized_title):
            return f"{normalize_match_text(jurisdiction)}::{key}"

    return ""


def counting_sort_key(row: Dict[str, Any]) -> tuple[int, str, str]:
    return (
        parse_policy_year(row.get("policy_year", "")),
        normalize_match_text(row.get("title", "")),
        str(row.get("policy_id", "")),
    )


def apply_structural_counting_rules(rows: List[Dict[str, Any]]) -> None:
    grouped: Dict[str, List[Dict[str, Any]]] = {}

    for row in rows:
        row["counting_group_key"] = ""
        row["counting_keep"] = "false"
        row["counting_decision"] = "exclude_category"
        row["counting_reason"] = "Policy is not counted because it is outside the final contains_non_food sample."

        if row.get("category") != "contains_non_food" or not clean_bool(row.get("jurisdiction_keep")):
            continue

        row["counting_keep"] = "true"
        row["counting_decision"] = "keep_distinct"
        row["counting_reason"] = "Counted as a distinct non-food bioeconomy policy."

        group_key = build_counting_group_key(row.get("jurisdiction", ""), row.get("title", ""))
        if not group_key:
            continue

        row["counting_group_key"] = group_key
        grouped.setdefault(group_key, []).append(row)

    for group_key, group_rows in grouped.items():
        if len(group_rows) <= 1:
            continue

        representative = max(group_rows, key=counting_sort_key)
        representative["counting_keep"] = "true"
        representative["counting_decision"] = "keep_structural_representative"
        representative["counting_reason"] = (
            "Counted as the latest representative instrument within a repeated amendment or transposition chain."
        )

        for row in group_rows:
            if row is representative:
                continue
            row["counting_keep"] = "false"
            row["counting_decision"] = "exclude_structural_duplicate"
            row["counting_reason"] = (
                f"Excluded from policy counts as an earlier member of the same structural policy chain as "
                f"{representative.get('policy_id', '')}."
            )


def build_merged_rows(rows, filtered_rows, results) -> tuple[List[Dict[str, Any]], List[str]]:
    result_index = {item["policy_id"]: item for item in results}

    merged = []
    for row in filtered_rows:
        merged_row = dict(row)
        extra = result_index.get(row.get("policy_id"))
        if extra and extra.get("file_path"):
            merged_row["file_path"] = extra["file_path"]

        if extra:
            merged_row["category"] = extra["category"]
            merged_row["confidence"] = extra["confidence"]
            merged_row["reason"] = extra["reason"]
            merged_row["model_reason"] = extra.get("model_reason", "")
            merged_row["standardized_reason"] = extra.get("standardized_reason", "")
            merged_row["evidence"] = extra.get("evidence", "")
            merged_row["basis"] = extra.get("basis", "")
            merged_row["matched_terms"] = extra.get("matched_terms", "")
            merged_row["prompt_version"] = extra.get("prompt_version", "")
            merged_row["requested_model"] = extra.get("requested_model", "")
            merged_row["response_model"] = extra.get("response_model", "")
            merged_row["system_fingerprint"] = extra.get("system_fingerprint", "")
            merged_row["title_scope_cleaning_decision"] = extra.get("title_scope_cleaning_decision", "")
            merged_row["title_scope_cleaning_rule"] = extra.get("title_scope_cleaning_rule", "")
            merged_row["title_scope_cleaning_terms"] = extra.get("title_scope_cleaning_terms", "")
        else:
            merged_row["category"] = ""
            merged_row["confidence"] = ""
            merged_row["reason"] = ""
            merged_row["model_reason"] = ""
            merged_row["standardized_reason"] = ""
            merged_row["evidence"] = ""
            merged_row["basis"] = ""
            merged_row["matched_terms"] = ""
            merged_row["prompt_version"] = ""
            merged_row["requested_model"] = ""
            merged_row["response_model"] = ""
            merged_row["system_fingerprint"] = ""
            merged_row["title_scope_cleaning_decision"] = ""
            merged_row["title_scope_cleaning_rule"] = ""
            merged_row["title_scope_cleaning_terms"] = ""

        merged_row["review_status"] = review_status_value(merged_row["category"], merged_row.get("basis", ""))
        merged_row["include_non_food"] = include_non_food_value(merged_row["category"], merged_row.get("basis", ""))
        merged_row["include_type"] = include_type_value(merged_row["category"], merged_row.get("basis", ""))
        merged.append(merged_row)

    apply_structural_counting_rules(merged)

    fieldnames = list(rows[0].keys())
    for field in [
        "category",
        "confidence",
        "reason",
        "model_reason",
        "standardized_reason",
        "evidence",
        "basis",
        "matched_terms",
        "prompt_version",
        "requested_model",
        "response_model",
        "system_fingerprint",
        "title_scope_cleaning_decision",
        "title_scope_cleaning_rule",
        "title_scope_cleaning_terms",
        "review_status",
        "include_non_food",
        "include_type",
        "counting_group_key",
        "counting_keep",
        "counting_decision",
        "counting_reason",
    ]:
        if field not in fieldnames:
            fieldnames.append(field)

    return merged, fieldnames


def write_output_csv(path: Path, fieldnames: List[str], rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_output(rows, filtered_rows, results):
    merged, fieldnames = build_merged_rows(rows, filtered_rows, results)

    formal_rows = [row for row in merged if row.get("review_status") == "final_sample"]
    review_rows = [row for row in merged if row.get("review_status") == "review_queue"]
    technical_rows = [row for row in merged if row.get("review_status") == "technical_failure"]

    write_output_csv(OUTPUT_ALL_PATH, fieldnames, merged)
    write_output_csv(OUTPUT_PATH, fieldnames, formal_rows)
    write_output_csv(OUTPUT_REVIEW_QUEUE_PATH, fieldnames, review_rows)
    write_output_csv(OUTPUT_TECHNICAL_PATH, fieldnames, technical_rows)

    return merged, formal_rows, review_rows, technical_rows


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(CATALOG_PATH, "r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    filtered_rows = [row for row in rows if clean_bool(row.get("jurisdiction_keep"))]
    if NON_FOOD_TEST_LIMIT > 0:
        filtered_rows = filtered_rows[:NON_FOOD_TEST_LIMIT]
    existing_results = load_existing_results() if RESUME_FROM_EXISTING else {}
    results = list(existing_results.values())
    missing_pdf = 0
    unreadable_pdf = 0
    skipped_existing = 0
    processed_since_save = 0
    interrupted = False

    print(f"\nTotal policies: {len(rows)}")
    print(f"Policies with jurisdiction_keep=true: {len(filtered_rows)}")
    print(f"Resume from existing: {RESUME_FROM_EXISTING}")
    print(f"Existing processed policies: {len(existing_results)}")

    try:
        for row in tqdm(filtered_rows, total=len(filtered_rows)):
            policy_id = row.get("policy_id")
            if policy_id in existing_results:
                skipped_existing += 1
                continue

            title_scope = assess_title_scope(row.get("title", ""))
            title_scope_decision = title_scope.get("decision", "")
            title_scope_rule = title_scope.get("rule", "")
            title_scope_terms = title_scope.get("matched_terms", [])

            if title_scope_decision == "exclude":
                title_reason = title_scope.get("reason", "")
                result_row = {
                    "policy_id": policy_id,
                    "file_path": row.get("file_path", ""),
                    "category": "no_non_food",
                    "confidence": "high",
                    "reason": title_reason,
                    "model_reason": "",
                    "standardized_reason": title_reason,
                    "evidence": "",
                    "basis": "incidental_or_excluded_mention",
                    "matched_terms": " | ".join(title_scope_terms),
                    "prompt_version": NON_FOOD_PROMPT_VERSION,
                    "requested_model": "title_scope_cleaning",
                    "response_model": "",
                    "system_fingerprint": "",
                    "title_scope_cleaning_decision": title_scope_decision,
                    "title_scope_cleaning_rule": title_scope_rule,
                    "title_scope_cleaning_terms": " | ".join(title_scope_terms),
                }
                results.append(result_row)
                existing_results[policy_id] = result_row
                processed_since_save += 1
                continue

            if title_scope_decision == "review":
                title_reason = title_scope.get("reason", "")
                result_row = {
                    "policy_id": policy_id,
                    "file_path": row.get("file_path", ""),
                    "category": "unclear_content",
                    "confidence": "low",
                    "reason": title_reason,
                    "model_reason": "",
                    "standardized_reason": title_reason,
                    "evidence": "",
                    "basis": "mixed_or_borderline",
                    "matched_terms": " | ".join(title_scope_terms),
                    "prompt_version": NON_FOOD_PROMPT_VERSION,
                    "requested_model": "title_scope_cleaning",
                    "response_model": "",
                    "system_fingerprint": "",
                    "title_scope_cleaning_decision": title_scope_decision,
                    "title_scope_cleaning_rule": title_scope_rule,
                    "title_scope_cleaning_terms": " | ".join(title_scope_terms),
                }
                results.append(result_row)
                existing_results[policy_id] = result_row
                processed_since_save += 1
                continue

            pdf_path = resolve_pdf_path(row)

            if not os.path.exists(pdf_path):
                if title_scope_decision == "needs_text_check":
                    title_reason = title_scope.get("reason", "")
                    standardized_reason = (
                        f"{title_reason} Full text unavailable: missing PDF, so the title cannot be used "
                        "as a final non-food exclusion."
                    ).strip()
                    result_row = {
                        "policy_id": policy_id,
                        "file_path": pdf_path,
                        "category": "unclear_technical",
                        "confidence": "low",
                        "reason": standardized_reason,
                        "model_reason": "",
                        "standardized_reason": standardized_reason,
                        "evidence": "",
                        "basis": "insufficient_text_or_ambiguous",
                        "matched_terms": " | ".join(title_scope_terms),
                        "prompt_version": NON_FOOD_PROMPT_VERSION,
                        "requested_model": MODEL_NAME,
                        "response_model": "",
                        "system_fingerprint": "",
                        "title_scope_cleaning_decision": title_scope_decision,
                        "title_scope_cleaning_rule": title_scope_rule,
                        "title_scope_cleaning_terms": " | ".join(title_scope_terms),
                    }
                else:
                    missing_pdf += 1
                    standardized_reason = "missing_pdf"
                    result_row = {
                        "policy_id": policy_id,
                        "file_path": pdf_path,
                        "category": "unclear_technical",
                        "confidence": "low",
                        "reason": standardized_reason,
                        "model_reason": "",
                        "standardized_reason": standardized_reason,
                        "evidence": "",
                        "basis": "insufficient_text_or_ambiguous",
                        "matched_terms": "",
                        "prompt_version": NON_FOOD_PROMPT_VERSION,
                        "requested_model": MODEL_NAME,
                        "response_model": "",
                        "system_fingerprint": "",
                        "title_scope_cleaning_decision": title_scope_decision,
                        "title_scope_cleaning_rule": title_scope_rule,
                        "title_scope_cleaning_terms": " | ".join(title_scope_terms),
                    }
                results.append(result_row)
                existing_results[policy_id] = result_row
                processed_since_save += 1
            else:
                pdf_extract = extract_pdf_text(pdf_path)
                text = pdf_extract.get("text", "")
                evidence = pdf_extract.get("evidence", "")
                technical_error = pdf_extract.get("technical_error", "")

                if title_scope_decision == "needs_text_check" and not technical_error and len(text) >= 200:
                    if has_clear_non_food_biomass_override(text):
                        title_scope_decision = "pass_override"
                    else:
                        title_scope_decision = "needs_llm_check"

                if technical_error:
                    unreadable_pdf += 1
                    if title_scope_decision == "needs_text_check":
                        category = "unclear_technical"
                        confidence = "low"
                        standardized_reason = (
                            f"{title_scope.get('reason', '')} Full text is required, but PDF extraction failed: "
                            f"{technical_error}"
                        ).strip()
                        reason = standardized_reason
                        model_reason = ""
                        basis = "insufficient_text_or_ambiguous"
                        matched_terms = title_scope_terms
                        response_model = ""
                        system_fingerprint = ""
                    else:
                        category = "unclear_technical"
                        confidence = "low"
                        reason = technical_error
                        model_reason = ""
                        standardized_reason = technical_error
                        basis = "insufficient_text_or_ambiguous"
                        matched_terms = []
                        response_model = ""
                        system_fingerprint = ""
                elif len(text) < 200:
                    unreadable_pdf += 1
                    if title_scope_decision == "needs_text_check":
                        category = "unclear_technical"
                        confidence = "low"
                        standardized_reason = (
                            f"{title_scope.get('reason', '')} Full text is required, but the extracted text "
                            "was too short to classify reliably."
                        ).strip()
                        reason = standardized_reason
                        model_reason = ""
                        basis = "insufficient_text_or_ambiguous"
                        matched_terms = title_scope_terms
                        response_model = ""
                        system_fingerprint = ""
                    else:
                        category = "unclear_technical"
                        confidence = "low"
                        reason = "unreadable_pdf_or_text_too_short"
                        model_reason = ""
                        standardized_reason = "unreadable_pdf_or_text_too_short"
                        basis = "insufficient_text_or_ambiguous"
                        matched_terms = []
                        response_model = ""
                        system_fingerprint = ""
                else:
                    try:
                        result = classify_policy(text)
                        parsed = result.get("parsed", {})
                        category = normalize_category(parsed.get("category", "unclear"))
                        basis = normalize_basis(parsed.get("basis", ""))
                        matched_terms = normalize_matched_terms(parsed.get("matched_terms", []))
                        category, basis = apply_post_classification_overrides(
                            row.get("title", ""),
                            text,
                            category,
                            basis,
                            matched_terms,
                        )
                        confidence = deterministic_confidence(category, basis)
                        model_reason = normalize_text(str(parsed.get("reason", "")))
                        standardized_reason = deterministic_reason(category, basis, matched_terms)
                        reason = model_reason or standardized_reason
                        response_model = result.get("response_model", "")
                        system_fingerprint = result.get("system_fingerprint", "")
                    except Exception as exc:
                        category = "unclear_technical"
                        confidence = "low"
                        reason = f"api_or_parse_error: {exc}"
                        model_reason = ""
                        standardized_reason = reason
                        basis = "insufficient_text_or_ambiguous"
                        matched_terms = []
                        response_model = ""
                        system_fingerprint = ""

                requested_model = (
                    "title_scope_cleaning"
                    if title_scope_decision == "exclude" and category == "no_non_food"
                    else MODEL_NAME
                )
                result_row = {
                    "policy_id": policy_id,
                    "file_path": pdf_path,
                    "category": category,
                    "confidence": confidence,
                    "reason": reason,
                    "model_reason": model_reason,
                    "standardized_reason": standardized_reason,
                    "evidence": evidence,
                    "basis": basis,
                    "matched_terms": " | ".join(matched_terms),
                    "prompt_version": NON_FOOD_PROMPT_VERSION,
                    "requested_model": requested_model,
                    "response_model": response_model,
                    "system_fingerprint": system_fingerprint,
                    "title_scope_cleaning_decision": title_scope_decision,
                    "title_scope_cleaning_rule": title_scope_rule,
                    "title_scope_cleaning_terms": " | ".join(title_scope_terms),
                }
                results.append(result_row)
                existing_results[policy_id] = result_row
                processed_since_save += 1

            if processed_since_save >= CHECKPOINT_EVERY:
                _, formal_rows, review_rows, technical_rows = save_output(rows, filtered_rows, results)
                print(f"\nCheckpoint saved after {len(results)} processed policies.")
                print(
                    f"Formal sample rows: {len(formal_rows)} | "
                    f"Review queue rows: {len(review_rows)} | "
                    f"Technical failure rows: {len(technical_rows)}"
                )
                processed_since_save = 0
    except KeyboardInterrupt:
        interrupted = True
        print("\nInterrupted by user. Saving partial results...")

    merged, formal_rows, review_rows, technical_rows = save_output(rows, filtered_rows, results)

    gold_audit_rows = save_gold_set_audit(merged)

    print("\nSaved non-food outputs:")
    print(OUTPUT_ALL_PATH)
    print(OUTPUT_PATH)
    print(OUTPUT_REVIEW_QUEUE_PATH)
    print(OUTPUT_TECHNICAL_PATH)
    if gold_audit_rows:
        print(GOLD_SET_AUDIT_PATH)
    if interrupted:
        print("Run ended early after interruption; partial results were saved.")

    print("\nSummary:")
    counts = {}
    for row in merged:
        key = row["category"] or "missing"
        counts[key] = counts.get(key, 0) + 1
    print(counts)
    print(f"Formal sample rows: {len(formal_rows)}")
    print(f"Review queue rows: {len(review_rows)}")
    print(f"Technical failure rows: {len(technical_rows)}")
    print(f"Excluded jurisdiction_keep=false: {len(rows) - len(filtered_rows)}")
    print(f"Skipped existing: {skipped_existing}")
    print(f"Missing PDF: {missing_pdf}")
    print(f"Unreadable/too short PDF: {unreadable_pdf}")
    if gold_audit_rows:
        gold_pass = sum(1 for row in gold_audit_rows if row.get("pass") == "true")
        gold_total = len(gold_audit_rows)
        print(f"Gold set check: {gold_pass}/{gold_total} matched expected labels")
        failed = [row for row in gold_audit_rows if row.get("pass") != "true"]
        if failed:
            failed_ids = ", ".join(row["policy_id"] for row in failed)
            print(f"Gold set mismatches: {failed_ids}")


if __name__ == "__main__":
    main()
