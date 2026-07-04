import csv
import os
import re
import time
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import chromadb
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader

try:
    import tiktoken
except ImportError:
    tiktoken = None


# =========================================================
# Paths / config
# =========================================================
BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE_CANDIDATES = [
    BASE_DIR / "outputs" / "non_food_filter" / "policy_catalog_non_food_contains_jurisdiction_true_by_continent.csv",
]
OUTPUT_DIR = BASE_DIR / "outputs" / "rag_index_openai"
OUTPUT_MAIN_CATALOG = OUTPUT_DIR / "policy_catalog_main_for_rag.csv"
OUTPUT_UNCLEAR_CATALOG = OUTPUT_DIR / "policy_catalog_unclear_for_rag.csv"

VECTOR_PATH = BASE_DIR / "vector_db"
MAIN_COLLECTION_NAME = "policy_db_main_openai"
UNCLEAR_COLLECTION_NAME = "policy_db_unclear_openai"

FALLBACK_PDF_DIRS = [
    BASE_DIR / "faolex_policies",
    BASE_DIR / "policies",
    BASE_DIR / "faolex_policies" / "pdfs",
    BASE_DIR / "test_policies",
]

DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"

MAX_SECTION_CHARS = 18000
TARGET_CHUNK_CHARS = 6000
CHUNK_OVERLAP_PARAGRAPHS = 0
MIN_CHUNK_CHARS = 1500
MIN_SECTION_CHARS = 2500
EMBEDDING_MAX_TOKENS = 8192
EMBEDDING_TARGET_TOKENS = 6500
FALLBACK_CHARS_PER_TOKEN = 2.0

RECREATE_COLLECTION = False

TEST_LIMIT_MAIN = None
TEST_LIMIT_UNCLEAR = None

KEEP_FIELDS = [
    "policy_id",
    "faolex_no",
    "jurisdiction",
    "title",
    "policy_year",
    "primary_subject",
    "keywords",
    "non_food_bio_based_sector",
    "pdf_url",
    "category",
    "confidence",
    "reason",
    "jurisdiction_level",
    "jurisdiction_keep",
    "include_non_food",
    "review_status",
    "policy_status",
    "repealed_raw",
    "continent",
    "continent_detail",
    "counting_keep",
    "file_name",
    "file_path",
]


# =========================================================
# Load env / clients
# =========================================================
load_dotenv(BASE_DIR / ".env", override=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = os.getenv("OPENAI_EMBED_MODEL", DEFAULT_EMBEDDING_MODEL)

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not found in .env")

oa_client = OpenAI(api_key=OPENAI_API_KEY)
chroma_client = chromadb.PersistentClient(path=str(VECTOR_PATH))


# =========================================================
# Helpers
# =========================================================
def clean_str(value) -> str:
    if value is None:
        return ""
    return str(value).strip()


def strip_invalid_surrogates(text: str) -> str:
    if not text:
        return ""
    # Some PDFs extract to lone surrogate code points that cannot be UTF-8 encoded.
    return "".join(ch for ch in text if not 0xD800 <= ord(ch) <= 0xDFFF)


def normalize_spaces(text: str) -> str:
    text = strip_invalid_surrogates(text)
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    s = clean_str(value).lower()
    return s in {"true", "1", "yes", "y"}


def normalize_non_food(value) -> str:
    return clean_str(value).lower()


def clean_metadata_value(value, default="unknown"):
    if value is None or value == "":
        return default
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


@lru_cache(maxsize=4)
def get_token_encoder(model_name: str):
    if tiktoken is None:
        return None
    try:
        return tiktoken.encoding_for_model(model_name)
    except Exception:
        return tiktoken.get_encoding("cl100k_base")


def estimate_token_count(text: str) -> int:
    text = clean_str(text)
    if not text:
        return 0

    encoder = get_token_encoder(EMBEDDING_MODEL)
    if encoder is not None:
        try:
            return len(encoder.encode(text))
        except Exception:
            pass

    return max(1, int(len(text) / FALLBACK_CHARS_PER_TOKEN))


def fits_embedding_limit(text: str, max_tokens: int = EMBEDDING_TARGET_TOKENS) -> bool:
    return estimate_token_count(text) <= max_tokens


# =========================================================
# Continent mapping
# =========================================================
MACRO_REGION_TO_CONTINENT = {
    "European Union": "Europe",
    "EU": "Europe",
    "East African Community": "Africa",
    "EAC": "Africa",
    "African Union": "Africa",
    "ASEAN": "Asia",
    "APEC": "Multi-region",
    "OECD": "Multi-region",
    "Mercosur": "South America",
    "CARICOM": "North America",
}

COUNTRY_TO_CONTINENT = {
    "Australia": "Oceania",
    "Belgium": "Europe",
    "Brazil": "South America",
    "Canada": "North America",
    "Cabo Verde": "Africa",
    "China": "Asia",
    "France": "Europe",
    "Germany": "Europe",
    "India": "Asia",
    "Indonesia": "Asia",
    "Italy": "Europe",
    "Japan": "Asia",
    "Kenya": "Africa",
    "Kyrgyzstan": "Asia",
    "Mexico": "North America",
    "Micronesia": "Oceania",
    "South Africa": "Africa",
    "Spain": "Europe",
    "United Kingdom": "Europe",
    "United Kingdom of Great Britain and Northern Ireland": "Europe",
    "United States": "North America",
    "United States of America": "North America",
    "USA": "North America",
    "Zambia": "Africa",
    "Argentina": "South America",
}


def infer_continent(jurisdiction: str) -> str:
    j = clean_str(jurisdiction)
    if not j:
        return "Unknown"

    for k, v in MACRO_REGION_TO_CONTINENT.items():
        if j.lower() == k.lower() or k.lower() in j.lower():
            return v

    if j in COUNTRY_TO_CONTINENT:
        return COUNTRY_TO_CONTINENT[j]

    bracket_match = re.match(r"^(.+?)\s*\(([^)]+)\)\s*$", j)
    if bracket_match:
        base = bracket_match.group(1).strip()
        if base in COUNTRY_TO_CONTINENT:
            return COUNTRY_TO_CONTINENT[base]

    for country, continent in COUNTRY_TO_CONTINENT.items():
        if country.lower() in j.lower():
            return continent

    return "Unknown"


# =========================================================
# CSV
# =========================================================
def load_rows(csv_path: Path) -> List[Dict]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Input file not found: {csv_path}")
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def resolve_input_file() -> Path:
    for path in INPUT_FILE_CANDIDATES:
        if path.exists():
            return path
    searched = "\n".join(str(path) for path in INPUT_FILE_CANDIDATES)
    raise FileNotFoundError(f"No continent-filtered input file found. Searched:\n{searched}")


def normalize_policy_status(value) -> str:
    return clean_str(value).lower()


def is_active_policy(row: Dict) -> bool:
    return normalize_policy_status(row.get("policy_status")) == "active"


def passes_base_filters(row: Dict) -> bool:
    if not clean_bool(row.get("jurisdiction_keep")):
        return False
    if not is_active_policy(row):
        return False
    return True


def filter_main_rows(rows: List[Dict]) -> List[Dict]:
    out = []
    for row in rows:
        if not passes_base_filters(row):
            continue
        if (
            normalize_non_food(row.get("include_non_food")) == "true"
            or normalize_non_food(row.get("category")) == "contains_non_food"
        ):
            out.append(row)
    return out


def filter_unclear_rows(rows: List[Dict]) -> List[Dict]:
    out = []
    for row in rows:
        if passes_base_filters(row) and (
            normalize_non_food(row.get("include_non_food")) == "review"
            or normalize_non_food(row.get("review_status")) == "review_queue"
            or normalize_non_food(row.get("category")) in {"unclear_content", "unclear_technical"}
        ):
            out.append(row)
    return out


def build_final_catalog(rows: List[Dict]) -> List[Dict]:
    final_rows = []
    for row in rows:
        new_row = {}
        for field in KEEP_FIELDS:
            new_row[field] = row.get(field, "")
        if not clean_str(new_row.get("continent")):
            new_row["continent"] = infer_continent(row.get("jurisdiction", ""))
        new_row["index_status"] = "pending"
        new_row["index_note"] = ""
        new_row["indexed_chunk_count"] = 0
        new_row["resolved_pdf_path"] = ""
        final_rows.append(new_row)
    return final_rows


def save_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        print(f"[Warning] No rows to save: {path}")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# =========================================================
# PDF location
# =========================================================
def get_pdf_path(row: Dict) -> Optional[Path]:
    existing = clean_str(row.get("file_path"))
    if existing:
        p = Path(existing)
        if p.exists():
            return p

    if existing:
        name = Path(existing).name
        for base in FALLBACK_PDF_DIRS:
            candidate = base / name
            if candidate.exists():
                return candidate

    faolex_no = clean_str(row.get("faolex_no")).lower()
    title = clean_str(row.get("title")).lower()
    title_tokens = [t for t in re.split(r"\W+", title) if len(t) >= 4][:6]

    for base in FALLBACK_PDF_DIRS:
        if not base.exists():
            continue
        for file in base.glob("*.pdf"):
            fname = file.name.lower()
            if faolex_no and faolex_no in fname:
                return file
            overlap = sum(1 for t in title_tokens if t in fname)
            if overlap >= 3:
                return file

    return None


# =========================================================
# Read PDF
# =========================================================
def read_pdf_pages(file_path: Path) -> List[str]:
    pages: List[str] = []
    pypdf_error = None

    try:
        reader = PdfReader(str(file_path))
        for page in reader.pages:
            text = page.extract_text() or ""
            text = normalize_spaces(text)
            if text:
                pages.append(text)
        if pages:
            return pages
    except Exception as e:
        pypdf_error = e
        print(f"[PyPDF read error] {file_path}: {e}")

    try:
        import fitz  # PyMuPDF

        pages = []
        doc = fitz.open(str(file_path))
        for page in doc:
            text = page.get_text("text") or ""
            text = normalize_spaces(text)
            if text:
                pages.append(text)

        if pages:
            print(f"[PyMuPDF fallback succeeded] {file_path}")
            return pages
    except Exception as e:
        if pypdf_error is not None:
            print(f"[PyMuPDF read error] {file_path}: {e}")
        else:
            print(f"[PDF read error] {file_path}: {e}")

    return pages


def merge_pages(pages: List[str]) -> str:
    return "\n\n".join([p for p in pages if p.strip()])


# =========================================================
# Language detection
# =========================================================
def detect_language(text: str) -> str:
    snippet = text[:4000].strip()
    if not snippet:
        return "unknown"

    try:
        from langdetect import detect
        return detect(snippet)
    except Exception:
        pass

    if re.search(r"[\u4e00-\u9fff]", snippet):
        return "zh"
    if re.search(r"[\u0400-\u04FF]", snippet):
        return "ru"
    if re.search(r"[\u0600-\u06FF]", snippet):
        return "ar"
    if re.search(r"[A-Za-z]", snippet):
        return "en_or_latin"

    return "unknown"


# =========================================================
# Section-aware chunking
# =========================================================
def looks_like_heading(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    if len(s) < 4 or len(s) > 100:
        return False
    if s.endswith("."):
        return False

    words = s.split()
    if len(words) > 12:
        return False

    if re.match(r"^(chapter|section|part|article|annex|appendix)\s+[A-Za-z0-9IVXivx.\-]+([:.\-]|\s).*$", s, re.I):
        return True

    if re.match(r"^(\d+(\.\d+){0,3}|[IVXivx]{1,8}|[A-Z])[:.\-]?\s+[A-Z][^\n]+$", s):
        return True

    if s.isupper() and 1 <= len(words) <= 8:
        return True

    if s == s.title() and 1 <= len(words) <= 8 and len(re.findall(r"[,:;]", s)) <= 1:
        return True

    return False


def split_by_headings(text: str) -> List[Tuple[str, str]]:
    lines = text.splitlines()
    sections: List[Tuple[str, List[str]]] = []

    current_title = "DOCUMENT_START"
    current_lines: List[str] = []

    for idx, line in enumerate(lines):
        prev_blank = idx == 0 or not lines[idx - 1].strip()
        next_blank = idx == len(lines) - 1 or not lines[idx + 1].strip()
        heading_like = looks_like_heading(line)

        if heading_like and (prev_blank or next_blank):
            if current_lines:
                sections.append((current_title, current_lines))
            current_title = line.strip()
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        sections.append((current_title, current_lines))

    cleaned = []
    for title, sec_lines in sections:
        body = normalize_spaces("\n".join(sec_lines))
        if body.strip():
            cleaned.append((title, body))

    return cleaned


def merge_short_sections(
    sections: List[Tuple[str, str]],
    min_section_chars: int = MIN_SECTION_CHARS,
) -> List[Tuple[str, str]]:
    if not sections:
        return sections

    merged = []
    buffer_title, buffer_body = sections[0]

    for title, body in sections[1:]:
        if len(buffer_body) < min_section_chars:
            buffer_body = buffer_body + "\n\n" + body
        else:
            merged.append((buffer_title, buffer_body))
            buffer_title, buffer_body = title, body

    merged.append((buffer_title, buffer_body))
    return merged


def split_oversized_paragraph(paragraph: str, target: int) -> List[str]:
    paragraph = normalize_spaces(paragraph)
    if len(paragraph) <= target:
        return [paragraph] if paragraph else []

    sentences = re.split(r"(?<=[.!?;])\s+", paragraph)
    sentences = [s.strip() for s in sentences if s.strip()]
    if len(sentences) <= 1:
        return [paragraph[i:i + target].strip() for i in range(0, len(paragraph), target)]

    pieces = []
    current = ""
    for sentence in sentences:
        if not current:
            current = sentence
            continue
        if len(current) + 1 + len(sentence) <= target:
            current += " " + sentence
        else:
            pieces.append(current.strip())
            current = sentence

    if current.strip():
        pieces.append(current.strip())

    return pieces


def merge_small_chunks(
    chunks: List[str],
    min_chars: int = MIN_CHUNK_CHARS,
    max_chars: int = TARGET_CHUNK_CHARS,
    max_tokens: int = EMBEDDING_TARGET_TOKENS,
) -> List[str]:
    if not chunks:
        return []

    merged = []
    buffer = chunks[0].strip()

    for chunk in chunks[1:]:
        chunk = chunk.strip()
        if not chunk:
            continue

        candidate = f"{buffer}\n\n{chunk}".strip()
        if (
            len(buffer) < min_chars
            and len(candidate) <= max_chars
            and fits_embedding_limit(candidate, max_tokens=max_tokens)
        ):
            buffer = candidate
        else:
            merged.append(buffer)
            buffer = chunk

    if buffer:
        merged.append(buffer)

    return merged


def split_long_text(
    text: str,
    target: int = TARGET_CHUNK_CHARS,
    overlap_paragraphs: int = CHUNK_OVERLAP_PARAGRAPHS,
    max_tokens: int = EMBEDDING_TARGET_TOKENS,
) -> List[str]:
    text = normalize_spaces(text)
    if len(text) <= target and fits_embedding_limit(text, max_tokens=max_tokens):
        return [text] if text else []

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        paragraphs = [text]

    expanded_paragraphs: List[str] = []
    for para in paragraphs:
        expanded_paragraphs.extend(split_oversized_paragraph(para, target))

    chunks = []
    current_parts: List[str] = []
    current_len = 0

    for para in expanded_paragraphs:
        sep_len = 2 if current_parts else 0
        if current_parts and current_len + sep_len + len(para) > target:
            chunks.append("\n\n".join(current_parts).strip())
            overlap_parts = current_parts[-overlap_paragraphs:] if overlap_paragraphs > 0 else []
            current_parts = overlap_parts + [para]
            current_len = sum(len(part) for part in current_parts) + max(0, 2 * (len(current_parts) - 1))
        else:
            current_parts.append(para)
            current_len += sep_len + len(para)

    if current_parts:
        chunks.append("\n\n".join(current_parts).strip())

    merged_chunks = merge_small_chunks(
        chunks,
        min_chars=MIN_CHUNK_CHARS,
        max_chars=target,
        max_tokens=max_tokens,
    )
    return [c for c in merged_chunks if c.strip() and len(c.strip()) >= MIN_CHUNK_CHARS]


def split_for_embedding_limit(
    text: str,
    target: int = TARGET_CHUNK_CHARS,
    max_tokens: int = EMBEDDING_TARGET_TOKENS,
) -> List[str]:
    text = normalize_spaces(text)
    if not text:
        return []
    if len(text) <= target and fits_embedding_limit(text, max_tokens=max_tokens):
        return [text]

    queue = [text]
    safe_chunks: List[str] = []

    while queue:
        current = queue.pop(0).strip()
        if not current:
            continue

        if len(current) <= target and fits_embedding_limit(current, max_tokens=max_tokens):
            safe_chunks.append(current)
            continue

        next_target = max(MIN_CHUNK_CHARS, min(target, len(current) // 2))
        pieces = split_long_text(
            current,
            target=next_target,
            overlap_paragraphs=CHUNK_OVERLAP_PARAGRAPHS,
            max_tokens=max_tokens,
        )

        if len(pieces) <= 1 and pieces and pieces[0] == current:
            hard_target = max(MIN_CHUNK_CHARS, next_target)
            pieces = [
                current[i:i + hard_target].strip()
                for i in range(0, len(current), hard_target)
                if current[i:i + hard_target].strip()
            ]

        if len(pieces) <= 1 and pieces and pieces[0] == current:
            raise RuntimeError(
                f"Unable to split chunk to embedding-safe size: chars={len(current)}, "
                f"estimated_tokens={estimate_token_count(current)}"
            )

        queue = pieces + queue

    return safe_chunks


def build_structured_chunks(pages: List[str]) -> List[Dict]:
    text = merge_pages(pages)
    if not text.strip():
        return []

    sections = split_by_headings(text)
    sections = merge_short_sections(sections, min_section_chars=MIN_SECTION_CHARS)

    if len(sections) <= 1:
        raw_chunks = split_for_embedding_limit(text)
        return [
            {
                "section_title": "DOCUMENT_BODY",
                "chunk_text": chunk,
                "chunk_order": idx,
            }
            for idx, chunk in enumerate(raw_chunks)
        ]

    out = []
    chunk_order = 0

    for section_title, section_body in sections:
        section_body = normalize_spaces(section_body)
        if len(section_body) > MAX_SECTION_CHARS or not fits_embedding_limit(section_body):
            section_chunks = split_long_text(section_body)
        else:
            section_chunks = [section_body]

        for chunk in section_chunks:
            chunk_text = chunk
            if section_title and section_title != "DOCUMENT_START":
                chunk_text = f"{section_title}\n\n{chunk}".strip()

            safe_chunks = split_for_embedding_limit(chunk_text)
            for safe_chunk in safe_chunks:
                out.append(
                    {
                        "section_title": section_title,
                        "chunk_text": safe_chunk,
                        "chunk_order": chunk_order,
                    }
                )
                chunk_order += 1

    return out


def build_short_doc_chunk(full_text: str) -> List[Dict]:
    text = normalize_spaces(full_text)
    if not text:
        return []
    return [
        {
            "section_title": "SHORT_DOCUMENT",
            "chunk_text": text,
            "chunk_order": 0,
        }
    ]


# =========================================================
# OpenAI embedding
# =========================================================
def get_embedding(text: str) -> List[float]:
    for attempt in range(5):
        try:
            resp = oa_client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=text,
            )
            return resp.data[0].embedding
        except Exception as e:
            error_text = str(e).lower()
            if "maximum context length" in error_text or "max context length" in error_text:
                raise RuntimeError(
                    "OpenAI embedding input is too long even after chunking: "
                    f"chars={len(text)}, estimated_tokens={estimate_token_count(text)}, error={e}"
                )
            if attempt == 4:
                raise RuntimeError(f"OpenAI embedding failed after retries: {e}")
            print(f"[Embedding retry {attempt + 1}] {e}")
            time.sleep(2 * (attempt + 1))

    raise RuntimeError("Unreachable")


# =========================================================
# Chroma helpers
# =========================================================
def get_collection(collection_name: str, recreate: bool = False):
    if recreate:
        try:
            chroma_client.delete_collection(collection_name)
            print(f"Deleted existing collection: {collection_name}")
        except Exception:
            pass
    return chroma_client.get_or_create_collection(name=collection_name)


def get_existing_ids(collection) -> set:
    existing_ids = set()
    try:
        data = collection.get(include=[])
        for _id in data.get("ids", []):
            existing_ids.add(_id)
    except Exception:
        pass
    return existing_ids


def get_existing_policy_ids(existing_ids: set) -> set:
    policy_ids = set()
    for chunk_id in existing_ids:
        if "_" not in chunk_id:
            continue
        policy_id, chunk_suffix = chunk_id.rsplit("_", 1)
        if chunk_suffix.isdigit():
            policy_ids.add(policy_id)
    return policy_ids


# =========================================================
# Indexing
# =========================================================
def index_rows_to_collection(rows: List[Dict], collection_name: str, catalog_output_path: Path):
    if not rows:
        print(f"[Warning] No rows for collection: {collection_name}")
        return

    final_catalog = build_final_catalog(rows)

    collection = get_collection(collection_name, recreate=RECREATE_COLLECTION)
    existing_ids = get_existing_ids(collection)
    existing_policy_ids = get_existing_policy_ids(existing_ids)
    print(f"Existing chunk ids in {collection_name}: {len(existing_ids)}")
    print(f"Existing policy ids in {collection_name}: {len(existing_policy_ids)}")

    total_docs = 0
    total_chunks_added = 0
    missing_pdf = 0
    unreadable_pdf = 0
    skipped_existing_chunks = 0
    skipped_existing_policies = 0
    failed_exception = 0

    for row in final_catalog:
        policy_id = clean_str(row.get("policy_id"))
        title = clean_str(row.get("title"))

        try:
            if policy_id in existing_policy_ids:
                row["index_status"] = "skipped_existing_policy"
                row["index_note"] = "Policy already has chunks in collection"
                print(f"[Skipped existing policy] {policy_id} | {title}")
                skipped_existing_policies += 1
                continue

            pdf_path = get_pdf_path(row)
            row["resolved_pdf_path"] = clean_str(pdf_path) if pdf_path else ""

            if not pdf_path:
                row["index_status"] = "missing_pdf"
                row["index_note"] = "PDF file not found from file_path or fallback search"
                print(f"[Missing PDF] {policy_id} | {title}")
                missing_pdf += 1
                continue

            pages = read_pdf_pages(pdf_path)
            full_text = merge_pages(pages)

            if len(full_text.strip()) < 100:
                row["index_status"] = "needs_ocr"
                row["index_note"] = (
                    "PDF is openable but has no usable text layer or extracted text is shorter than "
                    "100 characters; OCR likely required"
                )
                print(f"[Needs OCR] {policy_id} | {pdf_path}")
                unreadable_pdf += 1
                continue

            language = detect_language(full_text)
            chunks = build_structured_chunks(pages)
            is_short_doc = False

            indexable_chunks = [
                chunk for chunk in chunks
                if len(chunk["chunk_text"].strip()) >= MIN_CHUNK_CHARS
            ]

            if not indexable_chunks and 100 <= len(full_text.strip()) < MIN_CHUNK_CHARS and fits_embedding_limit(full_text):
                indexable_chunks = build_short_doc_chunk(full_text)
                is_short_doc = True

            if not chunks and not indexable_chunks:
                row["index_status"] = "no_chunks"
                row["index_note"] = "No embedding-safe chunks were produced from PDF text"
                print(f"[No chunks] {policy_id} | {title}")
                unreadable_pdf += 1
                continue

            if not indexable_chunks:
                row["index_status"] = "short_doc_not_indexed"
                row["index_note"] = "Document text exists but is too short to produce an indexable chunk under current rules"
                print(f"[Short doc not indexed] {policy_id} | {title}")
                unreadable_pdf += 1
                continue

            ids = []
            docs = []
            embeddings = []
            metadatas = []

            for chunk in indexable_chunks:
                chunk_text = chunk["chunk_text"].strip()

                chunk_id = chunk["chunk_order"]
                full_chunk_id = f"{policy_id}_{chunk_id}"

                if full_chunk_id in existing_ids:
                    skipped_existing_chunks += 1
                    continue

                embedding = get_embedding(chunk_text)

                ids.append(full_chunk_id)
                docs.append(chunk_text)
                embeddings.append(embedding)

                metadata = {
                    "policy_id": clean_metadata_value(policy_id),
                    "faolex_no": clean_metadata_value(row.get("faolex_no")),
                    "jurisdiction": clean_metadata_value(row.get("jurisdiction")),
                    "title": clean_metadata_value(title),
                    "policy_year": clean_metadata_value(row.get("policy_year"), -1),
                    "primary_subject": clean_metadata_value(row.get("primary_subject")),
                    "keywords": clean_metadata_value(row.get("keywords")),
                    "non_food_bio_based_sector": clean_metadata_value(row.get("non_food_bio_based_sector")),
                    "pdf_url": clean_metadata_value(row.get("pdf_url")),
                    "category": clean_metadata_value(row.get("category")),
                    "confidence": clean_metadata_value(row.get("confidence")),
                    "reason": clean_metadata_value(row.get("reason")),
                    "continent": clean_metadata_value(row.get("continent")),
                    "continent_detail": clean_metadata_value(row.get("continent_detail")),
                    "policy_status": clean_metadata_value(row.get("policy_status")),
                    "jurisdiction_level": clean_metadata_value(row.get("jurisdiction_level")),
                    "jurisdiction_keep": True,
                    "include_non_food": clean_metadata_value(row.get("include_non_food")),
                    "language": clean_metadata_value(language),
                    "section_title": clean_metadata_value(chunk.get("section_title")),
                    "chunk_id": chunk_id,
                    "pdf_path": clean_metadata_value(str(pdf_path)),
                }
                metadatas.append(metadata)

            if ids:
                collection.add(
                    ids=ids,
                    documents=docs,
                    embeddings=embeddings,
                    metadatas=metadatas,
                )
                existing_ids.update(ids)
                existing_policy_ids.add(policy_id)
                if is_short_doc:
                    row["index_status"] = "indexed_short_doc"
                    row["index_note"] = f"Indexed short document successfully, language={language}"
                else:
                    row["index_status"] = "indexed"
                    row["index_note"] = f"Indexed successfully, language={language}"
                row["indexed_chunk_count"] = len(ids)
                total_docs += 1
                total_chunks_added += len(ids)
                print(f"[Indexed:{collection_name}] {policy_id} | added_chunks={len(ids)} | lang={language}")
            else:
                row["index_status"] = "skipped_existing_chunks_only"
                row["index_note"] = "All candidate chunks already existed or were too short"
                print(f"[Skipped existing chunks only] {policy_id} | {title}")
        except Exception as e:
            row["index_status"] = "failed_exception"
            row["index_note"] = str(e)[:500]
            failed_exception += 1
            print(f"[Failed exception] {policy_id} | {title} | {e}")
        finally:
            save_csv(catalog_output_path, final_catalog)

    save_csv(catalog_output_path, final_catalog)
    print(f"Saved catalog: {catalog_output_path}")

    print(f"\n=== Done: {collection_name} ===")
    print(f"Indexed documents: {total_docs}")
    print(f"Added chunks: {total_chunks_added}")
    print(f"Skipped existing policies: {skipped_existing_policies}")
    print(f"Skipped existing chunks: {skipped_existing_chunks}")
    print(f"Missing PDF: {missing_pdf}")
    print(f"Unreadable PDF: {unreadable_pdf}")
    print(f"Failed exception: {failed_exception}")


# =========================================================
# Main
# =========================================================
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    input_file = resolve_input_file()
    rows = load_rows(input_file)
    active_rows = sum(is_active_policy(r) for r in rows)
    repealed_rows = sum(normalize_policy_status(r.get("policy_status")) == "repealed" for r in rows)
    print(f"Loaded rows: {len(rows)}")
    print(f"Input file: {input_file}")

    j_true = sum(clean_bool(r.get("jurisdiction_keep")) for r in rows)
    nf_true = sum(normalize_non_food(r.get("include_non_food")) == "true" for r in rows)
    nf_review = sum(normalize_non_food(r.get("include_non_food")) == "review" for r in rows)
    counting_keep_true = sum(clean_str(r.get("counting_keep")).lower() == "true" for r in rows)
    counting_keep_false = sum(clean_str(r.get("counting_keep")).lower() == "false" for r in rows)
    both_true = sum(
        passes_base_filters(r) and normalize_non_food(r.get("include_non_food")) == "true"
        for r in rows
    )
    both_review = sum(
        passes_base_filters(r) and normalize_non_food(r.get("include_non_food")) == "review"
        for r in rows
    )

    print(f"jurisdiction_keep = TRUE: {j_true}")
    print(f"policy_status = active: {active_rows}")
    print(f"policy_status = repealed: {repealed_rows}")
    print(f"include_non_food = TRUE: {nf_true}")
    print(f"include_non_food = review: {nf_review}")
    print(f"counting_keep = true: {counting_keep_true}")
    print(f"counting_keep = false: {counting_keep_false}")
    print(f"main sample after base filters: {both_true}")
    print(f"review sample after base filters: {both_review}")

    main_rows = filter_main_rows(rows)
    unclear_rows = filter_unclear_rows(rows)

    if TEST_LIMIT_MAIN is not None:
        main_rows = main_rows[:TEST_LIMIT_MAIN]
    if TEST_LIMIT_UNCLEAR is not None:
        unclear_rows = unclear_rows[:TEST_LIMIT_UNCLEAR]

    index_rows_to_collection(
        rows=main_rows,
        collection_name=MAIN_COLLECTION_NAME,
        catalog_output_path=OUTPUT_MAIN_CATALOG,
    )

    index_rows_to_collection(
        rows=unclear_rows,
        collection_name=UNCLEAR_COLLECTION_NAME,
        catalog_output_path=OUTPUT_UNCLEAR_CATALOG,
    )

    print("\nAll indexing completed.")
    print(f"Vector DB path: {VECTOR_PATH}")
    print(f"Collections: {MAIN_COLLECTION_NAME}, {UNCLEAR_COLLECTION_NAME}")


if __name__ == "__main__":
    main()
