# sector_extract_v5_precise_sector.py
# Purpose: retrieval-assisted multi-label sector extraction for confirmed non-food bioeconomy policies.
# Key design: maximize real sector recall while preventing broad framework text from being over-assigned.

import csv
import json
import os
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import chromadb
from dotenv import load_dotenv
from openai import OpenAI


# =========================================================
# Paths / config
# =========================================================
BASE_DIR = Path(__file__).resolve().parent.parent

VECTOR_PATH = BASE_DIR / "vector_db"
COLLECTION_NAME = "policy_db_main_openai"

INPUT_CATALOG = Path(os.getenv("SECTOR_INPUT_CATALOG", str(BASE_DIR / "outputs" / "rag_index_v1_openai" / "policy_catalog_main_for_rag_0512.csv")))

OUTPUT_DIR = Path(os.getenv("SECTOR_OUTPUT_DIR", str(BASE_DIR / "outputs" / "sector_extract_v5")))
OUTPUT_OBSERVATIONS = OUTPUT_DIR / "sector_observations_0515.csv"
OUTPUT_SUMMARY = OUTPUT_DIR / "policy_sector_summary_0515.csv"
OUTPUT_LOG = OUTPUT_DIR / "sector_extract_log_0515.csv"
OUTPUT_REJECTED = OUTPUT_DIR / "sector_rejected_observations_0515.csv"

DEFAULT_MODEL = "gpt-5.4-mini"

TOP_K_CHUNKS = int(os.getenv("SECTOR_TOP_K_CHUNKS", "32"))
MIN_FRONT_CHUNKS = int(os.getenv("SECTOR_MIN_FRONT_CHUNKS", "3"))
MAX_CHARS_PER_BATCH = int(os.getenv("SECTOR_MAX_CHARS", "36000"))
SLEEP_SECONDS = float(os.getenv("SECTOR_SLEEP_SECONDS", "0.5"))
RETRY_TIMES = int(os.getenv("SECTOR_RETRY_TIMES", "4"))

TEST_LIMIT_ENV = os.getenv("SECTOR_TEST_LIMIT", "").strip()
TEST_LIMIT = int(TEST_LIMIT_ENV) if TEST_LIMIT_ENV else None
RESUME_FROM_EXISTING = False

PROCESSABLE_INDEX_STATUSES = {
    "indexed",
    "indexed_short_doc",
    "skipped_existing_policy",
}


# =========================================================
# Env / clients
# =========================================================
load_dotenv(BASE_DIR / ".env", override=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("OPENAI_MODEL", DEFAULT_MODEL)

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not found in .env")

oa_client = OpenAI(api_key=OPENAI_API_KEY)
chroma_client = chromadb.PersistentClient(path=str(VECTOR_PATH))
collection = chroma_client.get_collection(COLLECTION_NAME)


# =========================================================
# Sector classification system
# =========================================================
SECTOR_SYSTEM = {
    "bioenergy": [
        "biofuels",
        "biogas",
        "biomethane",
        "biomass power",
        "renewable fuels",
        "biomass heat",
    ],
    "bio_based_materials": [
        "bioplastics",
        "wood_based_materials",
        "packaging",
        "construction_materials",
        "fiber_materials",
        "pulp_and_paper",
        "biomaterials",
    ],
    "biochemicals": [
        "platform_chemicals",
        "solvents",
        "lubricants",
        "surfactants",
        "specialty_chemicals",
        "bio_based_polymers",
        "bio_based_chemicals",
    ],
    "industrial_biotechnology": [
        "fermentation",
        "enzymes",
        "synthetic_biology",
        "bio_manufacturing",
        "microbial_production",
        "biomanufacturing",
    ],
    "biorefinery": [
        "integrated_biorefinery",
        "cascading_biomass_use",
        "multi_product_conversion",
        "biomass_conversion_platform",
    ],
    "biomass_residue_valorization": [
        "agricultural_residues",
        "forestry_residues",
        "organic_waste",
        "side_streams",
        "waste_biomass",
        "residue_valorization",
    ],
    "biomass_feedstock_recovery_for_nonfood_use": [
        "biomass_collection",
        "biomass_segregation",
        "biomass_recovery",
        "feedstock_preparation",
        "organic_stream_separation",
        "biodegradable_material_recovery",
        "feedstock_mobilization",
    ],
}

ALLOWED_SECTORS = list(SECTOR_SYSTEM.keys())
SPECIAL_LABEL_SEQUENCE = [
    "broad_bioeconomy_unspecified",
    "sector_evidence_not_found_in_context",
    "sector_evidence_below_threshold",
    "unclear",
]
SPECIAL_LABELS = set(SPECIAL_LABEL_SEQUENCE)


# =========================================================
# Helpers
# =========================================================
def clean_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def load_catalog(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def save_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        print(f"[Warning] No rows to save: {path}")
        return

    # Use the union of keys so that newly added diagnostic columns in later rows
    # are not silently dropped or causing DictWriter errors.
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_existing_rows(path: Path) -> List[Dict[str, Any]]:
    if not RESUME_FROM_EXISTING or not path.exists():
        return []
    return load_catalog(path)


def upsert_log_row(log_rows: List[Dict[str, Any]], new_row: Dict[str, Any]) -> None:
    policy_id = clean_str(new_row.get("policy_id"))
    if not policy_id:
        log_rows.append(new_row)
        return

    for idx, row in enumerate(log_rows):
        if clean_str(row.get("policy_id")) == policy_id:
            log_rows[idx] = new_row
            return

    log_rows.append(new_row)


def get_processed_policy_ids(
    observation_rows: List[Dict[str, Any]],
    summary_rows: List[Dict[str, Any]],
    log_rows: List[Dict[str, Any]],
) -> set:
    processed = set()

    for row in observation_rows:
        policy_id = clean_str(row.get("policy_id"))
        if policy_id:
            processed.add(policy_id)

    for row in summary_rows:
        policy_id = clean_str(row.get("policy_id"))
        if policy_id:
            processed.add(policy_id)

    terminal_log_statuses = {"ok", "no_chunks", "empty_context"}
    for row in log_rows:
        policy_id = clean_str(row.get("policy_id"))
        status = clean_str(row.get("status")).lower()
        if policy_id and status in terminal_log_statuses:
            processed.add(policy_id)

    return processed


def filter_catalog_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    filtered = [
        row for row in rows
        if clean_str(row.get("index_status")) in PROCESSABLE_INDEX_STATUSES
    ]
    if TEST_LIMIT is not None:
        filtered = filtered[:TEST_LIMIT]
    return filtered


def normalize_text(s: str) -> str:
    s = str(s or "")
    s = s.replace("\u00a0", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def contains_any(text: str, keywords: List[str]) -> bool:
    lowered = normalize_text(text).lower()
    return any(keyword in lowered for keyword in keywords)


def build_nonfood_screening_context(row: Dict[str, Any]) -> str:
    fields = [
        ("non_food_basis", row.get("basis", "")),
        ("non_food_matched_terms", row.get("matched_terms", "")),
        ("non_food_evidence", row.get("evidence", "")),
        ("non_food_reason", row.get("reason", "")),
        ("non_food_include_type", row.get("include_type", "")),
    ]

    lines = []
    for key, value in fields:
        value = clean_str(value)
        if value:
            lines.append(f"{key}: {value}")

    if not lines:
        return ""

    return "[Non-food screening evidence]\n" + "\n".join(lines)


UPSTREAM_FEEDSTOCK_TERMS = [
    "feedstock",
    "biomass collection",
    "collection",
    "segregation",
    "separation",
    "sorting",
    "recovery",
    "mobilization",
    "mobilisation",
    "preparation",
    "organic stream",
    "biodegradable material recovery",
]

DOWNSTREAM_NONFOOD_USE_TERMS = [
    "non-food",
    "bioenergy",
    "biofuel",
    "biogas",
    "biomethane",
    "biomaterial",
    "biomaterials",
    "bio-based material",
    "bio-based materials",
    "chemical",
    "chemicals",
    "biochemical",
    "biochemicals",
    "industrial use",
    "industrial application",
    "biorefinery",
    "energy",
    "fuel",
    "heat",
    "power",
]

BIOREFINERY_REQUIRED_TERMS = [
    "biorefinery",
    "integrated biorefinery",
    "biomass conversion platform",
    "integrated conversion",
    "multi-product",
    "multiple products",
    "co-product",
    "co-products",
    "cascading biomass use",
    "cascade use",
    "fractionation",
    "convert biomass into fuels and chemicals",
    "convert biomass into materials and chemicals",
]

def sector_gate_status(sector: str, evidence: str, justification: str) -> tuple[bool, str]:
    """
    Keep post-processing language-agnostic.

    Sector admission is decided by the LLM via the prompt. Code-side validation
    should only do lightweight schema checks and retain special labels.
    """
    if sector in SPECIAL_LABELS:
        return True, "special_label"
    return True, "llm_admission_retained"


def passes_sector_specific_rules(sector: str, evidence: str, justification: str) -> bool:
    passed, _ = sector_gate_status(sector, evidence, justification)
    return passed


# =========================================================
# Chunk retrieval
# =========================================================
def chunk_priority_score(doc_text: str, section_title: str) -> int:
    text = (section_title + "\n" + doc_text).lower()
    score = 0

    broad_keywords = [
        "bioeconomy", "bio-based", "biobased", "biomass", "biotechnology",
        "industrial use", "non-food", "renewable carbon", "bioproduct", "bio-product",
        "bio-based resources", "biological resources", "feedstock", "biomaterial",
    ]

    sector_specific_keywords = [
        "biofuel", "biofuels", "biogas", "biomethane", "biomass power",
        "renewable fuels", "biomass heat",
        "bioplastics", "bioplastic", "wood", "wood-based", "packaging",
        "construction material", "construction materials", "fiber", "fibre",
        "pulp", "paper", "biomaterial", "biomaterials",
        "platform chemicals", "solvents", "lubricants", "surfactants",
        "specialty chemicals", "bio-based polymers", "bio-based chemical",
        "bio-based chemicals",
        "fermentation", "enzymes", "synthetic biology",
        "bio manufacturing", "biomanufacturing", "microbial production",
        "industrial biotechnology",
        "biorefinery", "integrated biorefinery", "cascading biomass",
        "multi-product", "conversion platform", "biomass conversion",
        "agricultural residues", "forestry residues", "organic waste",
        "side streams", "side-streams", "waste biomass", "residue valorization",
        "biomass collection", "feedstock", "feedstock recovery", "feedstock preparation",
        "biomass recovery", "biomass segregation", "organic stream",
        "biodegradable stream", "biodegradable materials", "resource recovery",
        "mobilization of biomass", "collection of organic waste",
        "separation of organic waste", "segregation", "recovery",
    ]

    policy_action_keywords = [
        "measure", "measures", "action", "actions", "support", "fund", "subsid",
        "grant", "mandate", "target", "programme", "program", "strategy",
        "implementation", "regulation", "standard", "scheme", "roadmap",
        "promotion", "priority", "development", "collection", "recovery",
        "segregation", "mobilization", "processing",
    ]

    for kw in broad_keywords:
        if kw in text:
            score += 1

    for kw in sector_specific_keywords:
        if kw in text:
            score += 4

    for kw in policy_action_keywords:
        if kw in text:
            score += 2

    if any(x in section_title.lower() for x in [
        "sector", "industry", "application", "applications", "value chain",
        "biomass", "biofuel", "biorefinery", "biotechnology", "materials",
        "chemicals", "waste", "residue", "implementation", "measures",
        "feedstock", "recovery", "collection", "segregation"
    ]):
        score += 6

    return score


def get_policy_chunks(policy_id: str) -> List[Dict[str, Any]]:
    data = collection.get(
        where={"policy_id": policy_id},
        include=["documents", "metadatas"],
    )

    docs = data.get("documents", [])
    metas = data.get("metadatas", [])

    out = []
    for doc, meta in zip(docs, metas):
        out.append(
            {
                "text": normalize_text(doc),
                "section_title": clean_str(meta.get("section_title")),
                "chunk_id": meta.get("chunk_id"),
            }
        )

    # Preserve a few early chunks because strategic documents often state their
    # substantive scope near the beginning, while also prioritizing high-scoring
    # sector-relevant chunks for recall.
    out_by_order = sorted(out, key=lambda x: (x["chunk_id"] if x["chunk_id"] is not None else 999999))
    front_chunks = out_by_order[:MIN_FRONT_CHUNKS]

    ranked_chunks = sorted(
        out,
        key=lambda x: chunk_priority_score(x["text"], x["section_title"]),
        reverse=True,
    )

    selected = []
    seen_ids = set()
    for ch in front_chunks + ranked_chunks:
        chunk_id = ch.get("chunk_id")
        if chunk_id is not None:
            key = ("chunk_id", chunk_id)
        else:
            key = (
                "chunk_fallback",
                clean_str(ch.get("section_title")),
                normalize_text(clean_str(ch.get("text")))[:500],
            )
        if key in seen_ids:
            continue
        selected.append(ch)
        seen_ids.add(key)
        if len(selected) >= TOP_K_CHUNKS:
            break

    return selected


def build_context(chunks: List[Dict[str, Any]], max_chars: int = MAX_CHARS_PER_BATCH) -> str:
    parts = []
    total = 0

    for i, ch in enumerate(chunks, start=1):
        section = clean_str(ch["section_title"]) or "UNKNOWN_SECTION"
        text = clean_str(ch["text"])
        if not text:
            continue

        block = f"[Chunk {i} | Section: {section}]\n{text}\n"
        if total + len(block) > max_chars:
            break

        parts.append(block)
        total += len(block)

    return "\n\n".join(parts).strip()


# =========================================================
# Prompt
# =========================================================
def build_prompt(context: str) -> str:
    sector_json = json.dumps(SECTOR_SYSTEM, ensure_ascii=False, indent=2)

    return f"""
You are coding sectoral focus in non-food bioeconomy policy documents.
Important context:
All input policies have already passed the non-food bioeconomy screening.
Your task is NOT to decide whether the policy belongs to the non-food bioeconomy.
Your task is only to identify whether the policy text shows explicit focus on one or more allowed sectors.

Non-food screening evidence is provided only to explain why the policy entered the corpus and to help locate relevant content. It is NOT sufficient by itself for assigning a specific sector.
For an allowed sector assignment, the evidence span must explicitly identify the sectoral object, product, process, feedstock, end use, or value-chain pathway, and the text must show concrete policy focus on that sector.

Core principle:
The non-food bioeconomy concerns biomass, bio-based biological resources,
or biomass-derived streams that are used, recovered, managed, processed,
or governed for non-food purposes.

Task:
1. Identify explicit evidence for each allowed sector.
2. A policy may map to multiple sectors, but for each policy-sector pair, retain only the single strongest and most representative evidence.
3. Assign each retained observation to exactly one allowed sector.
4. Only code sectors supported by explicit text evidence.
5. If the text does not provide enough explicit evidence for any allowed sector, return an empty observations array.
6. Do not return broad labels, review labels, fallback labels, or uncertainty labels. Return only allowed sectors or an empty observations array.

Important coding rules:
- Code policy content, not general background rhetoric.
- Multiple sectors are allowed within a single policy.
- Do NOT return multiple observations for the same sector merely because it is mentioned several times.
- Repeated mentions within the same sector should be collapsed into one strongest observation.
- Prefer the most representative and sufficiently specific evidence span, not the shortest phrase.
- Do NOT infer sectors not supported by explicit text.
- The policy text and evidence may be in any language. Judge sector membership from meaning, not from English keywords.
- Evidence may be quoted or paraphrased from the original language, but it must stay grounded in the retrieved text.
- Do NOT code a sector merely because the document lists it as part of the broader bioeconomy, names it as a possible output, or mentions it as a future opportunity.
- A sector should be coded only when the text shows concrete policy focus, intervention, implementation, support, regulation, deployment, funding, targets, or other operational treatment of that sector.
- General references to circular economy, recycling, waste management,
  plastics, climate action, renewable resources, or innovation should
  NOT be coded as a specific sector unless the text explicitly links them
  to biomass for non-food purposes.
- Broad bioeconomy strategies, roadmaps, action plans, and innovation strategies often list many possible sectors, products, or value chains. Do NOT code all listed sectors unless the text shows concrete policy focus on each one.
- Phrases such as "outputs include", "opportunities include", "value chains include", "priority areas include", or similar enumerations are not sufficient by themselves.

Sector interpretation rules:

1. Use "bioenergy" only when the text clearly concerns biomass used for
   energy, heat, fuels, gas, or power generation.

2. Use "bio_based_materials" only when the text explicitly identifies material products or material applications, such as bio-based materials, biomaterials, bioplastics, bio-based packaging, wood/fibre-based materials, construction materials, timber construction, pulp, paper, textiles, or other clearly material-oriented bio-based products. The generic phrase "bio-based products" alone is NOT sufficient.

3. Use "biochemicals" only when the text explicitly identifies bio-based chemicals, biochemical products, bio-based polymers, solvents, surfactants, lubricants, specialty chemicals, or related chemical production. Generic "bio-based products" or "bio-based industry" alone is NOT sufficient.

4. Use "industrial_biotechnology" only when the text clearly links biotechnology, fermentation, enzymes, microbial production, synthetic biology, or biomanufacturing to non-food industrial production. General biotechnology research or bioeconomy innovation is NOT sufficient unless the industrial non-food use is explicit.

5. Use "biorefinery" only when the text explicitly mentions biorefinery, biorefineries, integrated biomass conversion, cascading biomass use, biomass fractionation, biomass conversion platforms, or multi-product conversion of biomass into fuels, materials and/or chemicals. Do NOT assign biorefinery merely from terms such as circular bio-based industry, bio-based products, bio-based materials and chemicals, or circular bio-based solutions.

6. Use "biomass_residue_valorization" only when the text clearly concerns
   agricultural, forestry, or organic biomass residues being valorized or used
   for non-food industrial, material, chemical, or energy purposes.

7. Use "biomass_feedstock_recovery_for_nonfood_use" only when the text clearly
   governs the collection, segregation, recovery, preparation, or mobilization
   of biomass, biodegradable biomass-derived materials, or organic biomass streams
   for subsequent non-food use, and the text must show BOTH an upstream
   feedstock-recovery element and a downstream non-food use element, but does not yet fit more specifically into
   bioenergy, materials, chemicals, biotechnology, biorefinery, or residue valorization.

Exclusion rules:
- General waste collection, hazardous waste disposal, general recycling,
  plastic bag bans, sustainable products regulation, circular economy policy,
  or ecodesign should not be coded as an allowed sector unless the text explicitly concerns
  biomass or bio-based resources for non-food purposes.
- "Plastic" alone is NOT enough for bio_based_materials.
- "Organic waste" alone is NOT enough for biomass_residue_valorization.
- "Biotechnology" alone is NOT enough for industrial_biotechnology unless
  the context is non-food industrial use.
- Broad mentions of "bioeconomy", "bio-based products", "renewable biological resources",
  or "biobased industry" without identifiable sector content should not be coded as an allowed sector.
- For biorefinery, require direct evidence of integrated conversion, cascading use,
  or multi-product biomass processing; do not infer biorefinery from generic circular use alone.
- For biomass_feedstock_recovery_for_nonfood_use, require clearly upstream feedstock-oriented governance or recovery content AND an explicit downstream non-food use; do not use it as a catch-all for general biological resource governance.

Calibration examples:
- Example A: A bioeconomy strategy says that opportunities include biofuels, biomaterials, and biochemicals, but does not describe concrete policy measures, targets, funding, regulation, or implementation for those sectors. Return: {{"observations": []}}
- Example B: An innovation strategy says outputs may include bio-energy, biomaterials, and bio-based chemicals. This is a broad listing of possible outputs, not concrete sector focus. Return: {{"observations": []}}
- Example C: A policy creates payments to expand production of advanced biofuels and support biomass-to-energy deployment. Return one observation for "bioenergy".
- Example D: A policy establishes biorefinery facilities converting biomass into fuels and chemicals through integrated multi-product conversion. Return one observation for "biorefinery".

Allowed sectors:
{sector_json}

Return ONLY valid JSON in this format:
{{
  "observations": [
    {{
      "sector": "bioenergy",
      "evidence": "short text excerpt or concise representative span",
      "justification": "brief reason grounded in the text",
      "confidence": 0.0
    }}
  ]
}}

Policy text:
{context}
""".strip()


# =========================================================
# LLM call
# =========================================================
def escape_invalid_backslashes(text: str) -> str:
    return re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", text)


def parse_json_object(text: str) -> Dict[str, Any]:
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError("No JSON object found in model output")

    json_text = match.group(0)
    try:
        return json.loads(json_text)
    except json.JSONDecodeError as e:
        if "Invalid \\escape" not in str(e):
            raise
        repaired = escape_invalid_backslashes(json_text)
        return json.loads(repaired)


def call_llm(prompt: str) -> Dict[str, Any]:
    last_error = None

    for attempt in range(RETRY_TIMES):
        try:
            resp = oa_client.responses.create(
                model=MODEL_NAME,
                input=prompt,
                temperature=0,
            )
            text = resp.output_text.strip()
            return parse_json_object(text)

        except Exception as e:
            last_error = e
            print(f"[LLM retry {attempt + 1}] {e}")
            time.sleep(2 * (attempt + 1))

    raise RuntimeError(f"LLM call failed after retries: {last_error}")


# =========================================================
# Validation / aggregation
# =========================================================
def validate_observations(raw_obs: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    cleaned = []
    rejected = []
    seen = set()

    for obs in raw_obs:
        sector = clean_str(obs.get("sector")).lower()
        evidence = normalize_text(clean_str(obs.get("evidence")))
        justification = normalize_text(clean_str(obs.get("justification")))
        confidence = obs.get("confidence", 0.5)

        if not sector:
            continue

        if sector not in ALLOWED_SECTORS and sector not in SPECIAL_LABELS:
            rejected.append({
                "sector": sector,
                "evidence": evidence,
                "justification": justification,
                "confidence": confidence,
                "reject_reason": "unknown_sector_or_label",
            })
            continue

        if not evidence:
            rejected.append({
                "sector": sector,
                "evidence": evidence,
                "justification": justification,
                "confidence": confidence,
                "reject_reason": "empty_evidence",
            })
            continue

        passed, gate_reason = sector_gate_status(sector, evidence, justification)
        if not passed:
            rejected.append({
                "sector": sector,
                "evidence": evidence,
                "justification": justification,
                "confidence": confidence,
                "reject_reason": gate_reason,
            })
            continue

        try:
            confidence = float(confidence)
        except Exception:
            confidence = 0.5

        confidence = max(0.0, min(1.0, confidence))

        dedup_key = (sector, evidence.lower())
        if dedup_key in seen:
            continue
        seen.add(dedup_key)

        cleaned.append(
            {
                "sector": sector,
                "evidence": evidence,
                "justification": justification,
                "confidence": round(confidence, 3),
                "gate_reason": gate_reason,
            }
        )

    return cleaned, rejected


def collapse_to_best_sector_observations(observations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    best_allowed: Dict[str, Dict[str, Any]] = {}
    best_special: Dict[str, Dict[str, Any]] = {}

    def is_better(obs: Dict[str, Any], current: Dict[str, Any]) -> bool:
        if obs["confidence"] > current["confidence"]:
            return True
        if obs["confidence"] == current["confidence"]:
            if len(obs["evidence"]) > len(current["evidence"]):
                return True
            if len(obs["evidence"]) == len(current["evidence"]) and len(obs["justification"]) > len(current["justification"]):
                return True
        return False

    for obs in observations:
        sector = obs["sector"]

        if sector in ALLOWED_SECTORS:
            current = best_allowed.get(sector)
            if current is None or is_better(obs, current):
                best_allowed[sector] = obs
        elif sector in SPECIAL_LABELS:
            current = best_special.get(sector)
            if current is None or is_better(obs, current):
                best_special[sector] = obs

    collapsed = [best_allowed[s] for s in ALLOWED_SECTORS if s in best_allowed]
    if collapsed:
        return collapsed

    for label in SPECIAL_LABEL_SEQUENCE:
        if label in best_special:
            return [best_special[label]]

    return []



def clip_text(value: Any, max_chars: int = 5000) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False)
    except Exception:
        text = str(value)
    return text[:max_chars]


def make_fallback_observation(
    raw_observations: List[Dict[str, Any]],
    validated_observation_count: int,
    chunks: List[Dict[str, Any]],
    context: str,
    screening_context: str,
) -> List[Dict[str, Any]]:
    """
    Avoid true empty outputs for policies that have passed non-food screening.

    If retrieval failed, use sector_evidence_not_found_in_context.
    If retrieval succeeded but no observation survived the evidence gate, use
    sector_evidence_below_threshold. This is not counted as a sector in the
    composition figures; it is a review/status label.
    """
    retrieved_chunk_count = len(chunks or [])
    has_context = bool(clean_str(context))
    has_screening_context = bool(clean_str(screening_context))
    raw_count = len(raw_observations or [])

    if retrieved_chunk_count == 0 and has_screening_context:
        return [
            {
                "sector": "sector_evidence_not_found_in_context",
                "evidence": "Non-food screening evidence was available, but no retrieved policy chunks were found in the vector database.",
                "justification": "Sector attribution could not be determined because the retrieved policy context was unavailable.",
                "confidence": 1.0,
            }
        ]

    if has_context and has_screening_context:
        if raw_count == 0:
            reason = "The model returned no observations."
        elif validated_observation_count == 0:
            reason = "The model returned observations, but none were retained after lightweight validation."
        else:
            reason = "The model returned observations, but none were retained after sector prioritization."
        return [
            {
                "sector": "sector_evidence_below_threshold",
                "evidence": "Non-food screening context and retrieved policy chunks were available, but no sector-specific observation met the predefined evidence threshold.",
                "justification": (
                    f"{reason} The policy is retained in the non-food bioeconomy corpus, "
                    "but no allowed sector or other special label was retained under the current extraction rules."
                ),
                "confidence": 1.0,
            }
        ]

    if has_context:
        return [
            {
                "sector": "sector_evidence_below_threshold",
                "evidence": "Retrieved policy context was available, but no sector-specific observation met the predefined evidence threshold.",
                "justification": "No allowed sector or special label was retained under the current extraction rules.",
                "confidence": 1.0,
            }
        ]

    return []


# =========================================================
# Main extraction
# =========================================================
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    catalog = load_catalog(INPUT_CATALOG)
    catalog = filter_catalog_rows(catalog)

    all_observations = load_existing_rows(OUTPUT_OBSERVATIONS)
    summary_rows = load_existing_rows(OUTPUT_SUMMARY)
    log_rows = load_existing_rows(OUTPUT_LOG)
    rejected_rows = load_existing_rows(OUTPUT_REJECTED)
    processed_policy_ids = get_processed_policy_ids(all_observations, summary_rows, log_rows)

    print(f"Policies eligible to process: {len(catalog)}")
    print(f"Already processed policies: {len(processed_policy_ids)}")
    print(f"Model: {MODEL_NAME}")
    print(f"Collection: {COLLECTION_NAME}")

    for idx, row in enumerate(catalog, start=1):
        policy_id = clean_str(row.get("policy_id"))
        faolex_no = clean_str(row.get("faolex_no"))
        jurisdiction = clean_str(row.get("jurisdiction"))
        policy_year = clean_str(row.get("policy_year") or row.get("year"))
        title = clean_str(row.get("title"))
        screening_context = build_nonfood_screening_context(row)

        print(f"\n[{idx}/{len(catalog)}] Processing {policy_id} | {title}")

        if policy_id in processed_policy_ids:
            print(f"[Skip processed] {policy_id}")
            continue

        try:
            chunks = get_policy_chunks(policy_id)
            if not chunks:
                if not screening_context:
                    upsert_log_row(
                        log_rows,
                        {
                            "policy_id": policy_id,
                            "status": "no_chunks",
                            "message": "No chunks found in vector DB",
                        }
                    )
                    continue
                context = screening_context
            else:
                context = build_context(chunks)

            if not context:
                if not screening_context:
                    upsert_log_row(
                        log_rows,
                        {
                            "policy_id": policy_id,
                            "status": "empty_context",
                            "message": "No usable context built",
                        }
                    )
                    continue
                context = screening_context

            if screening_context and context != screening_context:
                context = screening_context + "\n\n" + context

            prompt = build_prompt(context)
            raw = call_llm(prompt)
            raw_observations = raw.get("observations", [])
            raw_observation_count = len(raw_observations)

            validated_observations, rejected_observations = validate_observations(raw_observations)
            validated_observation_count = len(validated_observations)
            rejected_observation_count = len(rejected_observations)

            for rej_i, rej in enumerate(rejected_observations, start=1):
                rejected_rows.append({
                    "policy_id": policy_id,
                    "faolex_no": faolex_no,
                    "jurisdiction": jurisdiction,
                    "policy_year": policy_year,
                    "title": title,
                    "rejected_obs_id": f"{policy_id}_rejected_{rej_i}",
                    "sector": rej.get("sector", ""),
                    "evidence": rej.get("evidence", ""),
                    "justification": rej.get("justification", ""),
                    "confidence": rej.get("confidence", ""),
                    "reject_reason": rej.get("reject_reason", ""),
                    "review_model": MODEL_NAME,
                })

            observations = collapse_to_best_sector_observations(validated_observations)

            fallback_label = ""
            if not observations:
                fallback_observations = make_fallback_observation(
                    raw_observations=raw_observations,
                    validated_observation_count=validated_observation_count,
                    chunks=chunks,
                    context=context,
                    screening_context=screening_context,
                )
                if fallback_observations:
                    observations = fallback_observations
                    fallback_label = observations[0]["sector"]

            sectors_for_count = []
            best_sector_map: Dict[str, Dict[str, Any]] = {}
            best_review_obs: Dict[str, Any] | None = None

            for obs_i, obs in enumerate(observations, start=1):
                sector = obs["sector"]

                if sector in ALLOWED_SECTORS:
                    sectors_for_count.append(sector)
                    best_sector_map[sector] = obs
                    all_observations.append(
                        {
                            "policy_id": policy_id,
                            "faolex_no": faolex_no,
                            "jurisdiction": jurisdiction,
                            "policy_year": policy_year,
                            "title": title,
                            "obs_id": f"{policy_id}_sector_{obs_i}",
                            "sector": sector,
                            "evidence": obs["evidence"],
                            "justification": obs["justification"],
                            "confidence": obs["confidence"],
                            "review_model": MODEL_NAME,
                        }
                    )
                else:
                    if best_review_obs is None:
                        best_review_obs = obs
                    rejected_rows.append({
                        "policy_id": policy_id,
                        "faolex_no": faolex_no,
                        "jurisdiction": jurisdiction,
                        "policy_year": policy_year,
                        "title": title,
                        "rejected_obs_id": f"{policy_id}_review_{obs_i}",
                        "sector": sector,
                        "evidence": obs.get("evidence", ""),
                        "justification": obs.get("justification", ""),
                        "confidence": obs.get("confidence", ""),
                        "reject_reason": "review_label_not_in_main_sector_output",
                        "review_model": MODEL_NAME,
                    })

            counter = Counter(sectors_for_count)
            sector_list = ";".join(
                sector for sector in ALLOWED_SECTORS if counter.get(sector, 0) > 0
            )
            sector_count = sum(1 for sector in ALLOWED_SECTORS if counter.get(sector, 0) > 0)
            review_label = best_review_obs["sector"] if best_review_obs else ""

            summary_row = {
                "policy_id": policy_id,
                "faolex_no": faolex_no,
                "jurisdiction": jurisdiction,
                "policy_year": policy_year,
                "title": title,
                "retrieved_chunk_count": len(chunks),
                "context_char_count": len(context),
                "has_nonfood_screening_context": bool(screening_context),
                "raw_observation_count": raw_observation_count,
                "validated_observation_count": validated_observation_count,
                "rejected_observation_count": rejected_observation_count,
                "retained_observation_count": len(sectors_for_count),
                "review_label": review_label,
                "cross_sectoral": sector_count > 1,
                "sector_count": sector_count,
                "sector_list": sector_list,
                "has_no_sector_signal": sector_count == 0,
                "needs_main_review": sector_count == 0,
                "needs_review_any": sector_count == 0,
                "needs_review_no_signal": sector_count == 0,
                "sector_review_status": "sector_identified" if sector_count > 0 else "no_sector_identified",
            }

            for sector in ALLOWED_SECTORS:
                best = best_sector_map.get(sector)
                summary_row[sector] = int(best is not None)
                summary_row[f"{sector}_flag"] = int(best is not None)
                summary_row[f"{sector}_evidence"] = best["evidence"] if best else ""
                summary_row[f"{sector}_justification"] = best["justification"] if best else ""
                summary_row[f"{sector}_confidence"] = best["confidence"] if best else ""

            summary_row["review_evidence"] = best_review_obs["evidence"] if best_review_obs else ""
            summary_row["review_justification"] = best_review_obs["justification"] if best_review_obs else ""
            summary_row["review_confidence"] = best_review_obs["confidence"] if best_review_obs else ""

            summary_rows.append(summary_row)

            upsert_log_row(
                log_rows,
                {
                    "policy_id": policy_id,
                    "status": "ok",
                    "message": (
                        f"raw_observations={raw_observation_count}; "
                        f"validated_observations={validated_observation_count}; "
                        f"rejected_observations={rejected_observation_count}; "
                        f"retained_observations={len(sectors_for_count)}; "
                        f"sector_count={sector_count}; "
                        f"review_label={review_label}; "
                        f"retrieved_chunks={len(chunks)}; context_chars={len(context)}"
                    ),
                    "raw_observation_count": raw_observation_count,
                    "validated_observation_count": validated_observation_count,
                    "rejected_observation_count": rejected_observation_count,
                    "retained_observation_count": len(sectors_for_count),
                    "review_label": review_label,
                    "raw_observations_json": clip_text(raw_observations, 5000),
                }
            )

            time.sleep(SLEEP_SECONDS)

        except Exception as e:
            upsert_log_row(
                log_rows,
                {
                    "policy_id": policy_id,
                    "status": "error",
                    "message": str(e),
                }
            )
            print(f"[Error] {policy_id}: {e}")
        finally:
            save_csv(OUTPUT_OBSERVATIONS, all_observations)
            save_csv(OUTPUT_SUMMARY, summary_rows)
            save_csv(OUTPUT_LOG, log_rows)
            save_csv(OUTPUT_REJECTED, rejected_rows)

        latest_log_row = next(
            (row for row in reversed(log_rows) if clean_str(row.get("policy_id")) == policy_id),
            None,
        )
        latest_status = clean_str(latest_log_row.get("status")).lower() if latest_log_row else ""
        if latest_status and latest_status != "error":
            processed_policy_ids.add(policy_id)

    save_csv(OUTPUT_OBSERVATIONS, all_observations)
    save_csv(OUTPUT_SUMMARY, summary_rows)
    save_csv(OUTPUT_LOG, log_rows)
    save_csv(OUTPUT_REJECTED, rejected_rows)

    print("\n=== Done ===")
    print(f"Observation table: {OUTPUT_OBSERVATIONS}")
    print(f"Summary table: {OUTPUT_SUMMARY}")
    print(f"Log table: {OUTPUT_LOG}")
    print(f"Rejected observations table: {OUTPUT_REJECTED}")


if __name__ == "__main__":
    main()
