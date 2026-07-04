import csv
import json
import os
import re
import time
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
INPUT_CATALOG = BASE_DIR / "outputs" / "rag_index_openai" / "policy_catalog_main_for_rag.csv"

OUTPUT_DIR = BASE_DIR / "outputs" / "sdg_mapping_openai"
OUTPUT_OBSERVATIONS = OUTPUT_DIR / "sdg_observations.csv"
OUTPUT_SUMMARY = OUTPUT_DIR / "policy_sdg_summary.csv"
OUTPUT_LOG = OUTPUT_DIR / "sdg_extract_log.csv"

DEFAULT_MODEL = "gpt-5.4-mini"
TOP_K_CHUNKS = 12
MAX_CHARS_PER_BATCH = 18000
SLEEP_SECONDS = 0.5
RETRY_TIMES = 4

TEST_LIMIT = None
RESUME_FROM_EXISTING = True

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
MODEL_NAME = os.getenv("SDG_MODEL", DEFAULT_MODEL)

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not found in .env")

oa_client = OpenAI(api_key=OPENAI_API_KEY)
chroma_client = chromadb.PersistentClient(path=str(VECTOR_PATH))
collection = chroma_client.get_collection(COLLECTION_NAME)


# =========================================================
# SDG framework (aligned with manuscript + SI)
# =========================================================
SDG_FRAMEWORK = {
    "ESA": {
        "SDG6": {
            "description": "Water resource sustainability",
            "policy_indicators": [
                "water efficiency",
                "wastewater treatment",
                "water pollution control",
                "sustainable water use",
                "water resource protection",
                "reduced runoff",
                "leaching reduction",
            ],
            "bioeconomy_relevance": "Relevant when non-food bioeconomy policies address water demand, water quality, wastewater, runoff, or aquatic impacts associated with biomass production or processing.",
        },
        "SDG12": {
            "description": "Sustainable consumption and production",
            "policy_indicators": [
                "resource efficiency",
                "waste reduction",
                "recycling",
                "circular biomass use",
                "side-stream utilization",
                "responsible production",
                "cascading use",
                "valorization",
            ],
            "bioeconomy_relevance": "Under ESA, SDG12 refers to environmental resource efficiency, circular biomass use, recycling, waste reduction, and responsible production.",
        },
        "SDG13": {
            "description": "Climate change mitigation",
            "policy_indicators": [
                "greenhouse gas reduction",
                "emission mitigation",
                "decarbonization",
                "low-carbon transition",
                "carbon reduction targets",
                "climate strategy",
                "renewable carbon",
                "fossil substitution",
            ],
            "bioeconomy_relevance": "Relevant when policies explicitly connect non-food biomass, bio-based substitution, or renewable carbon systems to climate mitigation or decarbonization.",
        },
        "SDG14": {
            "description": "Marine ecosystem protection",
            "policy_indicators": [
                "marine ecosystem protection",
                "coastal resource protection",
                "marine biomass sustainability",
                "blue bioeconomy safeguards",
                "algae sustainability",
            ],
            "bioeconomy_relevance": "Relevant when policies concern marine biomass, algae, fisheries residues, coastal systems, or marine ecosystem safeguards.",
        },
        "SDG15": {
            "description": "Biodiversity and land protection",
            "policy_indicators": [
                "biodiversity conservation",
                "forest protection",
                "land-use sustainability",
                "ecosystem restoration",
                "sustainable biomass sourcing",
                "habitat protection",
                "deforestation control",
            ],
            "bioeconomy_relevance": "Relevant when policies address forests, ecosystems, land-use change, biodiversity safeguards, or sustainable biomass sourcing.",
        },
    },
    "SIA": {
        "SDG1": {
            "description": "Poverty reduction and resilience",
            "policy_indicators": [
                "poverty reduction",
                "rural income",
                "livelihood support",
                "resilience",
                "inclusive rural development",
                "income generation",
            ],
            "bioeconomy_relevance": "Relevant when policies link non-food bioeconomy development to livelihoods, resilience, or rural income generation.",
        },
        "SDG3": {
            "description": "Health and environmental wellbeing",
            "policy_indicators": [
                "human health",
                "pollution reduction",
                "air quality improvement",
                "safer materials",
                "reduced toxic exposure",
                "environmental wellbeing",
            ],
            "bioeconomy_relevance": "Relevant when policies connect bio-based substitution, cleaner production, safer materials, or pollution reduction to health or environmental wellbeing.",
        },
        "SDG8": {
            "description": "Employment and inclusive growth",
            "policy_indicators": [
                "job creation",
                "green jobs",
                "skills development",
                "inclusive growth",
                "decent work",
                "regional employment",
                "SME development",
                "rural employment",
            ],
            "bioeconomy_relevance": "Relevant when policies emphasize employment, value-chain development, local industry, regional development, or inclusive growth.",
        },
    },
    "EIA": {
        "SDG7": {
            "description": "Clean energy transition",
            "policy_indicators": [
                "renewable energy",
                "clean energy",
                "bioenergy deployment",
                "energy efficiency",
                "renewable fuels",
                "energy transition",
                "biomethane",
                "biofuels",
            ],
            "bioeconomy_relevance": "Relevant when policies promote bioenergy, biomethane, biofuels, or non-fossil energy transitions.",
        },
        "SDG9": {
            "description": "Industrial innovation",
            "policy_indicators": [
                "industrial innovation",
                "research and development",
                "technology demonstration",
                "industrial upgrading",
                "biorefinery development",
                "innovation ecosystem",
                "industrial biotechnology",
                "pilot plants",
                "demonstration plants",
            ],
            "bioeconomy_relevance": "Relevant when policies support industrial biotechnology, demonstration plants, biorefineries, or bio-based industrial innovation.",
        },
        "SDG12": {
            "description": "Sustainable industrial production",
            "policy_indicators": [
                "bio-based materials",
                "industrial circularity",
                "sustainable manufacturing",
                "clean production",
                "industrial resource efficiency",
                "bio-based product substitution",
                "bio-based value chains",
            ],
            "bioeconomy_relevance": "Under EIA, SDG12 refers to sustainable industrial production, bio-based materials, circular industry, and clean manufacturing.",
        },
    },
}

SDG_DIMENSIONS = ["ESA", "SIA", "EIA"]

GENERIC_ALIGNMENT_PATTERNS = [
    r"\bsustainable development\b",
    r"\bgreen transition\b",
    r"\bgreen growth\b",
    r"\bsustainability transition\b",
    r"\beconomic social environmental\b",
    r"\btriple bottom line\b",
    r"\bholistic benefits\b",
    r"\bmultiple benefits\b",
    r"\bbalanced development\b",
]

SUBSTANTIVE_POLICY_SIGNAL_TERMS = [
    "target", "targets", "measure", "measures", "programme", "program",
    "fund", "funding", "budget", "grant", "subsidy", "subsidies",
    "standard", "standards", "regulation", "regulatory", "mandate",
    "requirement", "requirements", "monitoring", "indicator", "indicators",
    "responsible authority", "competent authority", "ministry", "agency",
    "plan", "roadmap", "implementation", "training", "infrastructure",
    "research support", "demonstration", "pilot", "procurement",
]

STRONG_IMPLEMENTATION_SIGNAL_TERMS = [
    "budget", "funding", "grant", "subsidy", "investment",
    "mandatory", "shall", "must", "required", "requirement",
    "target", "targets", "quota", "timeline", "roadmap",
    "indicator", "indicators", "monitoring", "reporting",
    "authority", "ministry", "agency", "responsible", "regulation",
    "standard", "certification", "enforcement", "compliance",
]


# =========================================================
# Helpers
# =========================================================
def clean_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def normalize_text(s: str) -> str:
    s = s.replace("\u00a0", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def load_catalog(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def save_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        print(f"[Warning] No rows to save: {path}")
        return

    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_existing_rows(path: Path) -> List[Dict[str, Any]]:
    if not RESUME_FROM_EXISTING or not path.exists():
        return []
    return load_catalog(path)


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

    for row in log_rows:
        policy_id = clean_str(row.get("policy_id"))
        if policy_id:
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


# =========================================================
# Chunk retrieval
# =========================================================
def chunk_priority_score(doc_text: str, section_title: str) -> int:
    text = (section_title + "\n" + doc_text).lower()
    score = 0

    thematic_keywords = [
        "sustainable", "sustainability", "climate", "mitigation", "adaptation",
        "emission", "energy", "innovation", "industry", "production", "consumption",
        "water", "marine", "biodiversity", "land", "employment", "poverty",
        "health", "wellbeing", "inclusive growth", "circular", "resource efficiency",
        "decarbonization", "renewable", "waste", "recycling", "livelihood",
        "resilience", "green jobs", "clean production", "biorefinery",
        "biotechnology", "biomanufacturing", "bio-based", "biomass",
        "rural development", "stakeholder", "participation", "forest",
        "pollution", "ecosystem", "value chain", "demonstration plant",
        "innovation hub", "industrial transformation", "fossil substitution",
    ]
    policy_keywords = [
        "measure", "measures", "policy", "target", "programme", "program",
        "strategy", "action plan", "implementation", "regulation", "standard",
        "fund", "support", "roadmap", "priority", "mandate", "scheme",
        "subsidy", "investment", "certification", "monitoring", "governance",
    ]

    # First-layer retrieval: use thematic expressions only for broad recall.
    for kw in thematic_keywords:
        if kw in text:
            score += 2

    for kw in policy_keywords:
        if kw in text:
            score += 2

    if any(x in section_title.lower() for x in [
        "sustainability", "climate", "environment", "innovation", "industry",
        "energy", "water", "biodiversity", "measures", "implementation",
        "health", "employment", "circular", "waste", "production", "social",
        "rural", "forest", "biotechnology", "technology",
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
                "text": normalize_text(clean_str(doc)),
                "section_title": clean_str(meta.get("section_title")),
                "chunk_id": meta.get("chunk_id"),
            }
        )

    out.sort(key=lambda x: (x["chunk_id"] if x["chunk_id"] is not None else 999999))
    out = sorted(
        out,
        key=lambda x: chunk_priority_score(x["text"], x["section_title"]),
        reverse=True,
    )
    return out[:TOP_K_CHUNKS]


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
# Prompt (mechanism-driven, aligned with manuscript + SI)
# =========================================================
def build_prompt(context: str) -> str:
    framework = json.dumps(SDG_FRAMEWORK, ensure_ascii=False, indent=2)

    return f"""
You are an expert in sustainability governance and non-food bioeconomy policy analysis.

You must assess sustainability alignment ONLY on the basis of the retrieved policy text below.
Do not infer policy impacts, implementation success, or real-world outcomes beyond what is explicitly supported by the text.

Framework:
{framework}

Task:
Assess sustainability alignment for each of the following three dimensions:
- ESA = Environmental Sustainability Alignment
- SIA = Social Inclusiveness Alignment
- EIA = Economic & Innovation Alignment

Use a THREE-LAYER decision logic:

Layer 1. Retrieval trigger layer:
- Keywords, thematic expressions, and semantic similarity are used only to retrieve potentially relevant policy text.
- They are NOT sufficient on their own to justify SDG alignment.

Layer 2. Admission layer:
- Only admit a dimension as aligned if the text contains traceable substantive policy content that can reasonably be interpreted as corresponding to that SDG dimension.
- Substantive policy content may include one or more of the following:
  1. a clear policy objective,
  2. a concrete measure or intervention,
  3. an identifiable target object or beneficiary,
  4. an implementation or governance mechanism.
- Evidence does NOT need to use official SDG terminology.
- Generic rhetoric such as "support sustainable development", "promote green transition", or "balance economic, social and environmental benefits" is not enough for strong alignment.

Required reasoning pathway:
Step 1. Identify explicit policy objectives stated in the retrieved text.
Step 2. Identify the sustainability mechanisms reflected in those objectives.
Step 3. Map those mechanisms to the allowed SDGs under each dimension.
Step 4. Assign a 0-3 score for each dimension.

Important:
- Do NOT assign SDGs only because a document is about the bioeconomy.
- Do NOT rely on loose keyword overlap alone.
- You must identify a meaningful policy objective or governance mechanism supported by the text.
- Generic green growth, innovation, sustainability, or transition rhetoric without dimension-specific substance should usually score 1, not 2 or 3, and may score 0 if no meaningful substantive policy content is present.
- A score of 2 or 3 requires dimension-specific evidence beyond broad rhetoric.
- A score of 3 requires clearly operationalized alignment, for example quantitative targets, concrete constraints, budgetary commitments, responsible authorities, monitoring arrangements, regulatory requirements, implementation plans, or similarly specific policy mechanisms.
- A score of 2 applies when the text clearly makes the dimension a policy priority and includes meaningful policy content, but the implementation details remain partial or incomplete.
- A score of 1 applies when the text only mentions policy objectives in vague expressions, or when targets are too generic to demonstrate meaningful alignment with the relevant SDG objective.
- When multiple statements occur within the same dimension, score based on the strongest target-based statement.

Dimension-specific guidance:

ESA includes text related to:
- greenhouse gas mitigation
- decarbonization
- resource efficiency
- circular biomass use
- side-stream or residue valorization
- sustainable biomass sourcing
- biodiversity protection
- land-use or forest safeguards
- water protection
- pollution reduction
- marine or coastal ecosystem protection

SIA includes text related to:
- rural development
- income generation
- poverty reduction
- livelihoods
- resilience
- job creation
- skills development
- just transition
- stakeholder participation
- inclusive regional development
- health or environmental wellbeing

EIA includes text related to:
- clean energy transition
- bioenergy deployment
- biofuels / biomethane / renewable fuels
- industrial innovation
- industrial biotechnology
- biorefineries
- pilot or demonstration plants
- R&D and technology development
- industrial upgrading
- value-chain development
- bio-based manufacturing
- sustainable industrial production

Allowed SDGs by dimension:
- ESA: SDG6, SDG12, SDG13, SDG14, SDG15
- SIA: SDG1, SDG3, SDG8
- EIA: SDG7, SDG9, SDG12

Important SDG12 distinction:
- Under ESA, SDG12 refers to resource efficiency, waste reduction, recycling, circular biomass use, and responsible production.
- Under EIA, SDG12 refers to sustainable industrial production, bio-based materials, circular industry, cleaner manufacturing, and bio-based value chains.

Scoring rules:
- 3 = containing clear quantitative targets that align with the relevant SDG objective, with a single type and relevant specific supporting targets (for example industrial, regional, or year-by-year targets)
- 2 = containing quantitative targets that align with the relevant SDG objective but without specific supporting targets; or containing timelines or roadmaps with few quantitative goals
- 1 = only mentioning policy objectives with vague expressions, or targets that are too generic to show meaningful SDG alignment
- 0 = no meaningful alignment signal in the retrieved text

Output requirements:
For EACH dimension return:
1. score
2. sdgs (list of allowed SDGs only)
3. mechanism (short phrase)
4. reason (short explanation grounded in the text)
5. evidence (short, text-grounded quote or paraphrase)

Important output discipline:
- If score = 0, return no SDGs unless the text contains a real but still weak dimension-specific policy signal.
- If score >= 2, the reason and evidence must make clear why the text passed the admission layer.
- Do not give 2 or 3 only because the text is "about sustainability" or "about the bioeconomy".
- Do not use broad semantic resemblance as the sole basis for inclusion.

Return ONLY valid JSON in this format:
{{
  "ESA": {{
    "score": 0,
    "sdgs": [],
    "mechanism": "",
    "reason": "",
    "evidence": ""
  }},
  "SIA": {{
    "score": 0,
    "sdgs": [],
    "mechanism": "",
    "reason": "",
    "evidence": ""
  }},
  "EIA": {{
    "score": 0,
    "sdgs": [],
    "mechanism": "",
    "reason": "",
    "evidence": ""
  }}
}}

Policy text:
{context}
""".strip()


# =========================================================
# LLM / validation
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


def validate_dimension_result(dimension: str, raw: Dict[str, Any]) -> Dict[str, Any]:
    score = raw.get("score", 0)
    try:
        score = int(score)
    except Exception:
        score = 0
    score = max(0, min(3, score))

    sdgs = raw.get("sdgs", [])
    if not isinstance(sdgs, list):
        sdgs = [sdgs]

    allowed_sdgs = set(SDG_FRAMEWORK[dimension].keys())
    cleaned_sdgs = []
    seen = set()
    for sdg in sdgs:
        value = clean_str(sdg).upper()
        if value in allowed_sdgs and value not in seen:
            seen.add(value)
            cleaned_sdgs.append(value)

    mechanism = normalize_text(clean_str(raw.get("mechanism")))
    reason = normalize_text(clean_str(raw.get("reason")))
    evidence = normalize_text(clean_str(raw.get("evidence")))

    combined_text = f"{mechanism}\n{reason}\n{evidence}".strip().lower()
    has_generic_only = any(re.search(pattern, combined_text, flags=re.I) for pattern in GENERIC_ALIGNMENT_PATTERNS)
    has_substantive_policy_signal = any(term in combined_text for term in SUBSTANTIVE_POLICY_SIGNAL_TERMS)
    has_strong_implementation_signal = any(term in combined_text for term in STRONG_IMPLEMENTATION_SIGNAL_TERMS)

    # consistency cleanup
    if score > 0 and not (mechanism or reason or evidence):
        score = 0

    if score > 0 and has_generic_only and not has_substantive_policy_signal:
        score = 0

    if score >= 2 and not has_substantive_policy_signal:
        score = 1 if (mechanism or reason or evidence) else 0

    if score == 3 and not has_strong_implementation_signal:
        score = 2 if has_substantive_policy_signal else 1

    if score == 0:
        cleaned_sdgs = []
        mechanism = ""
        reason = ""
        evidence = ""

    return {
        "score": score,
        "sdgs": cleaned_sdgs,
        "mechanism": mechanism,
        "reason": reason,
        "evidence": evidence,
    }


# =========================================================
# Main
# =========================================================
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    catalog = load_catalog(INPUT_CATALOG)
    catalog = filter_catalog_rows(catalog)

    observation_rows = load_existing_rows(OUTPUT_OBSERVATIONS)
    summary_rows = load_existing_rows(OUTPUT_SUMMARY)
    log_rows = load_existing_rows(OUTPUT_LOG)
    processed_policy_ids = get_processed_policy_ids(observation_rows, summary_rows, log_rows)

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

        print(f"\n[{idx}/{len(catalog)}] Processing {policy_id} | {title}")

        if policy_id in processed_policy_ids:
            print(f"[Skip processed] {policy_id}")
            continue

        try:
            chunks = get_policy_chunks(policy_id)
            if not chunks:
                log_rows.append(
                    {
                        "policy_id": policy_id,
                        "status": "no_chunks",
                        "message": "No chunks found in vector DB",
                    }
                )
                continue

            context = build_context(chunks)
            if not context:
                log_rows.append(
                    {
                        "policy_id": policy_id,
                        "status": "empty_context",
                        "message": "No usable context built",
                    }
                )
                continue

            prompt = build_prompt(context)
            raw = call_llm(prompt)

            validated = {
                dimension: validate_dimension_result(dimension, raw.get(dimension, {}))
                for dimension in SDG_DIMENSIONS
            }

            for dimension in SDG_DIMENSIONS:
                item = validated[dimension]
                observation_rows.append(
                    {
                        "policy_id": policy_id,
                        "faolex_no": faolex_no,
                        "jurisdiction": jurisdiction,
                        "policy_year": policy_year,
                        "title": title,
                        "dimension": dimension,
                        "score": item["score"],
                        "sdgs": ";".join(item["sdgs"]),
                        "mechanism": item["mechanism"],
                        "reason": item["reason"],
                        "evidence": item["evidence"],
                        "review_model": MODEL_NAME,
                    }
                )

            summary_rows.append(
                {
                    "policy_id": policy_id,
                    "faolex_no": faolex_no,
                    "jurisdiction": jurisdiction,
                    "policy_year": policy_year,
                    "title": title,

                    "ESA_score": validated["ESA"]["score"],
                    "ESA_sdgs": ";".join(validated["ESA"]["sdgs"]),
                    "ESA_mechanism": validated["ESA"]["mechanism"],
                    "ESA_reason": validated["ESA"]["reason"],
                    "ESA_evidence": validated["ESA"]["evidence"],

                    "SIA_score": validated["SIA"]["score"],
                    "SIA_sdgs": ";".join(validated["SIA"]["sdgs"]),
                    "SIA_mechanism": validated["SIA"]["mechanism"],
                    "SIA_reason": validated["SIA"]["reason"],
                    "SIA_evidence": validated["SIA"]["evidence"],

                    "EIA_score": validated["EIA"]["score"],
                    "EIA_sdgs": ";".join(validated["EIA"]["sdgs"]),
                    "EIA_mechanism": validated["EIA"]["mechanism"],
                    "EIA_reason": validated["EIA"]["reason"],
                    "EIA_evidence": validated["EIA"]["evidence"],

                    "overall_sdg_signal_count": sum(
                        1 for dimension in SDG_DIMENSIONS if validated[dimension]["score"] > 0
                    ),
                }
            )

            log_rows.append(
                {
                    "policy_id": policy_id,
                    "status": "ok",
                    "message": (
                        f"ESA={validated['ESA']['score']}; "
                        f"SIA={validated['SIA']['score']}; "
                        f"EIA={validated['EIA']['score']}"
                    ),
                }
            )

            time.sleep(SLEEP_SECONDS)

        except Exception as e:
            log_rows.append(
                {
                    "policy_id": policy_id,
                    "status": "error",
                    "message": str(e),
                }
            )
            print(f"[Error] {policy_id}: {e}")

        finally:
            processed_policy_ids.add(policy_id)
            save_csv(OUTPUT_OBSERVATIONS, observation_rows)
            save_csv(OUTPUT_SUMMARY, summary_rows)
            save_csv(OUTPUT_LOG, log_rows)

    save_csv(OUTPUT_OBSERVATIONS, observation_rows)
    save_csv(OUTPUT_SUMMARY, summary_rows)
    save_csv(OUTPUT_LOG, log_rows)

    print("\n=== Done ===")
    print(f"Observation table: {OUTPUT_OBSERVATIONS}")
    print(f"Summary table: {OUTPUT_SUMMARY}")
    print(f"Log table: {OUTPUT_LOG}")


if __name__ == "__main__":
    main()
