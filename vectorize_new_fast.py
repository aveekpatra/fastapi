"""
High-Quality RAG Data Pipeline for Czech Court Decisions
=========================================================
Fetches court decisions from justice.cz API, applies hybrid chunking,
embeds using Seznam Czech model, and uploads to Qdrant.

Features:
- Hybrid chunking (semantic + structural boundaries)
- Seznam/retromae-small-cs embedding model (256 dims)
- Enhanced metadata extraction
- Clean text extraction with ANON filtering
- Checkpoint-based resumability
- Comprehensive logging
"""

import hashlib
import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

# ============= CONFIGURATION =============
QDRANT_HOST = "hopper.proxy.rlwy.net"
QDRANT_PORT = 48447
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.cWl-rXCSkKbL9rIzNj00YIYFkMskD71UfoKfoECy7I0"
QDRANT_HTTPS = False

COLLECTION_NAME = "czech_court_decisions_v2"
EMBEDDING_MODEL = "Seznam/retromae-small-cs"
DENSE_VECTOR_SIZE = 256

# Chunking parameters (Czech-optimized)
CHUNK_SIZE = 1500  # ~375 tokens for Czech
CHUNK_OVERLAP = 200  # Context preservation
MIN_CHUNK_SIZE = 100  # Minimum viable chunk

# Processing parameters
EMBEDDING_BATCH_SIZE = 32
QDRANT_UPSERT_BATCH = 100
MAX_WORKERS_TEXT_FETCH = 50
MAX_WORKERS_HIERARCHY = 20
DECISIONS_BATCH_SIZE = 200

# API configuration
BASE_URL = "https://rozhodnuti.justice.cz/api/opendata"
REQUEST_TIMEOUT = 30
MAX_RETRIES = 3
RETRY_DELAY = 1

# Resume configuration
# Use Railway volume from env var, or check if path exists, otherwise current directory
VOLUME_PATH = os.getenv("RAILWAY_VOLUME_MOUNT_PATH", "/fastapi-volume" if os.path.exists("/fastapi-volume") else ".")

def get_volume_path():
    """Get volume path at runtime (after mount)."""
    env_path = os.getenv("RAILWAY_VOLUME_MOUNT_PATH")
    if env_path and os.path.exists(env_path):
        return env_path
    if os.path.exists("/fastapi-volume"):
        return "/fastapi-volume"
    return "."

# These will be set at runtime in RAGPipeline.__init__
CHECKPOINT_FILE = os.path.join(VOLUME_PATH, "rag_checkpoint.json")
LOG_FILE = os.path.join(VOLUME_PATH, "rag_pipeline.log")
ERROR_LOG_FILE = os.path.join(VOLUME_PATH, "rag_errors.log")
CHECKPOINT_FREQUENCY = 1  # Checkpoint every N days (1 = save after each day)


# ============= LOGGING =============
def setup_logging():
    """Configure comprehensive logging."""
    logger = logging.getLogger(__name__)
    logger.handlers.clear()
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    # Full log - use volume path
    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Error log - use volume path
    error_handler = logging.FileHandler(ERROR_LOG_FILE, encoding="utf-8")
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    logger.addHandler(error_handler)

    # Console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


logger = setup_logging()


# ============= DATA STRUCTURES =============
@dataclass
class DecisionChunk:
    """Represents a single chunk of a court decision."""
    chunk_id: str
    chunk_text: str
    chunk_index: int
    total_chunks: int
    section_type: str
    metadata: Dict


@dataclass
class CourtDecision:
    """Represents a complete court decision."""
    ecli: str
    case_number: str
    court: str
    court_code: str
    date_issued: str
    date_published: str
    decision_type: str
    case_subject: str
    case_result: List[str]
    judge_name: str
    judge_function: str
    keywords: List[str]
    legal_references: List[str]
    source_url: str
    header_text: str
    verdict_text: str
    justification_text: str
    information_text: str
    full_text: str


# ============= TEXT PROCESSING =============
def extract_text_from_section(section_data: List) -> str:
    """
    Extract clean text from nested JSON section, filtering out anonymized content.
    
    The API returns structured text like:
    [{"texts": [{"text": "...", "anonStyle": "NONE"}, {"text": "...", "anonStyle": "ANON"}]}]
    """
    if not section_data:
        return ""

    texts = []
    if isinstance(section_data, list):
        for item in section_data:
            if isinstance(item, dict) and "texts" in item:
                for text_item in item.get("texts", []):
                    if isinstance(text_item, dict):
                        text = text_item.get("text", "")
                        anon_style = text_item.get("anonStyle", "NONE")
                        # Only include non-anonymized text
                        if anon_style != "ANON" and text.strip():
                            texts.append(text)
            elif isinstance(item, str):
                texts.append(item)
    elif isinstance(section_data, str):
        texts.append(section_data)

    # Join and normalize whitespace only - no text manipulation
    result = " ".join(texts).strip()
    result = re.sub(r"\s+", " ", result)
    return result


def format_legal_references(regulations: List[Dict]) -> List[str]:
    """Format legal references into readable strings."""
    refs = []
    for reg in regulations:
        para = reg.get("paragraphNumber", "")
        lex_num = reg.get("lexNumber", "")
        lex_year = reg.get("lexYear", "")
        lex_type = reg.get("lexType", "")
        
        type_map = {
            "PREDPIS_ZAKON": "zák.",
            "PREDPIS_NARIZENI": "nař.",
            "PREDPIS_VYHLASKA": "vyhl.",
        }
        
        type_str = type_map.get(lex_type, "")
        if para and lex_num and lex_year:
            refs.append(f"§ {para} {type_str} č. {lex_num}/{lex_year} Sb.")
    return refs


def hybrid_chunk(text: str, section_type: str = "general",
                 chunk_size: int = CHUNK_SIZE, 
                 overlap: int = CHUNK_OVERLAP) -> List[Tuple[str, str]]:
    """
    Hybrid chunking strategy combining semantic and structural boundaries.
    
    Returns list of (chunk_text, section_type) tuples.
    """
    if not text or len(text) < MIN_CHUNK_SIZE:
        return [(text, section_type)] if text else []

    # Czech sentence boundary pattern
    # Handles numbered paragraphs (1., 2., etc.) and standard sentence endings
    sentence_pattern = r'(?<=[.!?])\s+(?=[A-ZÁČĎÉĚÍŇÓŘŠŤÚŮÝŽ0-9])'
    
    # Split by sentences
    sentences = re.split(sentence_pattern, text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return [(text, section_type)] if text else []

    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence_len = len(sentence)

        if current_length + sentence_len + 1 > chunk_size:
            if current_chunk:
                chunk_text = " ".join(current_chunk)
                chunks.append((chunk_text, section_type))

                # Calculate overlap from end of current chunk
                overlap_sentences = []
                overlap_len = 0
                for s in reversed(current_chunk):
                    if overlap_len + len(s) <= overlap:
                        overlap_sentences.insert(0, s)
                        overlap_len += len(s)
                    else:
                        break

                current_chunk = overlap_sentences + [sentence]
                current_length = sum(len(s) for s in current_chunk) + len(current_chunk)
            else:
                # Single sentence too long - split at punctuation or word boundary
                if sentence_len > chunk_size:
                    sub_chunks = split_long_sentence(sentence, chunk_size, overlap)
                    for sub in sub_chunks:
                        chunks.append((sub, section_type))
                else:
                    chunks.append((sentence, section_type))
                current_chunk = []
                current_length = 0
        else:
            current_chunk.append(sentence)
            current_length += sentence_len + 1

    # Add remaining
    if current_chunk:
        chunk_text = " ".join(current_chunk)
        if len(chunk_text) >= MIN_CHUNK_SIZE:
            chunks.append((chunk_text, section_type))
        elif chunks:
            # Merge small remainder with previous chunk
            prev_text, prev_type = chunks[-1]
            chunks[-1] = (prev_text + " " + chunk_text, prev_type)
        else:
            chunks.append((chunk_text, section_type))

    return chunks


def split_long_sentence(sentence: str, chunk_size: int, overlap: int) -> List[str]:
    """Split very long sentences at clause boundaries or word boundaries."""
    # Try clause boundaries first (semicolons, colons, conjunctions)
    clause_pattern = r'(?<=[;:])\s+|(?<=,)\s+(?=a\s|nebo\s|že\s|který\s|která\s|které\s)'
    clauses = re.split(clause_pattern, sentence)
    
    if len(clauses) > 1:
        chunks = []
        current = ""
        for clause in clauses:
            if len(current) + len(clause) + 1 <= chunk_size:
                current = (current + " " + clause).strip() if current else clause
            else:
                if current:
                    chunks.append(current)
                current = clause
        if current:
            chunks.append(current)
        return chunks
    
    # Fall back to word boundary splitting
    words = sentence.split()
    chunks = []
    current = ""
    for word in words:
        if len(current) + len(word) + 1 <= chunk_size:
            current = (current + " " + word).strip() if current else word
        else:
            if current:
                chunks.append(current)
            current = word
    if current:
        chunks.append(current)
    
    return chunks


def create_decision_chunks(decision: CourtDecision) -> List[DecisionChunk]:
    """
    Create all chunks for a court decision using structural boundaries.
    Only includes verdict and justification - the actual court reasoning.
    Excludes header (party details) and information (procedural notices).
    """
    all_chunks = []
    
    # Only include court's actual reasoning - verdict and justification
    # Skip header (party details/he-said-she-said) and information (procedural)
    sections = [
        ("verdict", decision.verdict_text),
        ("justification", decision.justification_text),
    ]
    
    for section_type, section_text in sections:
        if section_text and len(section_text) >= MIN_CHUNK_SIZE:
            section_chunks = hybrid_chunk(section_text, section_type)
            all_chunks.extend(section_chunks)
    
    # Create DecisionChunk objects
    result = []
    total = len(all_chunks)
    
    for idx, (chunk_text, section_type) in enumerate(all_chunks):
        chunk_id = f"{decision.ecli}_{idx}"
        
        metadata = {
            "ecli": decision.ecli,
            "case_number": decision.case_number,
            "court": decision.court,
            "court_code": decision.court_code,
            "date_issued": decision.date_issued,
            "date_published": decision.date_published,
            "decision_type": decision.decision_type,
            "case_subject": decision.case_subject,
            "case_result": decision.case_result,
            "judge_name": decision.judge_name,
            "judge_function": decision.judge_function,
            "keywords": decision.keywords,
            "legal_references": decision.legal_references,
            "source_url": decision.source_url,
            "section_type": section_type,
            "chunk_index": idx,
            "total_chunks": total,
            "has_full_text": idx == 0,
        }
        
        # Only first chunk gets full text (for payload efficiency)
        if idx == 0:
            metadata["full_text"] = decision.full_text
        
        result.append(DecisionChunk(
            chunk_id=chunk_id,
            chunk_text=chunk_text,
            chunk_index=idx,
            total_chunks=total,
            section_type=section_type,
            metadata=metadata,
        ))
    
    return result


# ============= CHECKPOINT =============
def load_checkpoint() -> Dict:
    """Load checkpoint from file."""
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load checkpoint: {e}")
    return {"processed_days": [], "total_chunks": 0}


def save_checkpoint(checkpoint: Dict):
    """Save checkpoint to file."""
    try:
        with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
            json.dump(checkpoint, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Failed to save checkpoint: {e}")


# ============= MAIN PIPELINE =============
class RAGPipeline:
    """High-quality RAG data pipeline for Czech court decisions."""

    def __init__(self):
        logger.info("🚀 Czech Court Decisions RAG Pipeline")
        logger.info("=" * 70)

        # Load embedding model
        logger.info(f"🧠 Loading embedding model: {EMBEDDING_MODEL}")
        self.model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")

        # Verify embedding dimension
        test_emb = self.model.encode(["test"])
        actual_dim = len(test_emb[0])
        if actual_dim != DENSE_VECTOR_SIZE:
            logger.warning(
                f"⚠️ Embedding dimension mismatch: expected {DENSE_VECTOR_SIZE}, got {actual_dim}"
            )

        # Connect to Qdrant
        logger.info("🔗 Connecting to Qdrant...")
        self.client = QdrantClient(
            host=QDRANT_HOST,
            port=QDRANT_PORT,
            api_key=QDRANT_API_KEY,
            https=QDRANT_HTTPS,
            timeout=120,
        )

        # HTTP session for API requests
        self.session = requests.Session()
        adapter = HTTPAdapter(pool_connections=100, pool_maxsize=100, max_retries=0)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        # State
        self.checkpoint = load_checkpoint()
        self.days_since_checkpoint = 0

        self.stats = {
            "decisions_fetched": 0,
            "decisions_processed": 0,
            "chunks_created": 0,
            "chunks_uploaded": 0,
            "text_fetch_success": 0,
            "text_fetch_errors": 0,
            "errors": 0,
            "embedding_time": 0.0,
            "upload_time": 0.0,
        }

    def setup_collection(self):
        """Create collection if it doesn't exist."""
        try:
            if not self.client.collection_exists(COLLECTION_NAME):
                logger.info(f"📦 Creating collection: {COLLECTION_NAME}")
                self.client.create_collection(
                    collection_name=COLLECTION_NAME,
                    vectors_config=VectorParams(
                        size=DENSE_VECTOR_SIZE, distance=Distance.COSINE
                    ),
                )
            else:
                info = self.client.get_collection(COLLECTION_NAME)
                logger.info(f"✅ Collection {COLLECTION_NAME} exists ({info.points_count:,} points)")
        except Exception as e:
            logger.error(f"Failed to setup collection: {e}")
            raise

    def api_request(self, url: str, description: str = "API request") -> Optional[any]:
        """Make API request with retry."""
        for attempt in range(MAX_RETRIES):
            try:
                response = self.session.get(url, timeout=REQUEST_TIMEOUT)
                response.raise_for_status()
                return response.json()
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY * (attempt + 1))
                else:
                    logger.debug(f"Failed {description} after {MAX_RETRIES} attempts: {e}")
                    return None

    def should_process_date(self, date_str: str) -> bool:
        """Check if date should be processed (not already in checkpoint)."""
        # Only skip if already processed
        if date_str in self.checkpoint.get("processed_days", []):
            return False
        return True

    def fetch_full_decision(self, odkaz_url: str, decision_data: Dict) -> Optional[CourtDecision]:
        """Fetch and parse complete court decision."""
        try:
            response = self.session.get(odkaz_url, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            data = response.json()

            # Extract structured text from sections
            header_text = extract_text_from_section(data.get("header", []))
            
            # Use pre-formatted text if available, otherwise extract from structured data
            # Keep text as-is from API - no manipulation
            verdict_text = data.get("verdictText", "") or extract_text_from_section(
                data.get("verdict", [])
            )
            justification_text = data.get(
                "justificationText", ""
            ) or extract_text_from_section(data.get("justification", []))
            
            information_text = extract_text_from_section(data.get("information", []))

            # Compose full text - only verdict and justification (court's reasoning)
            # Excludes header (party details) and information (procedural notices)
            full_text = f"""VÝROK:
{verdict_text}

ODŮVODNĚNÍ:
{justification_text}"""
            full_text = re.sub(r"\n{3,}", "\n\n", full_text).strip()

            # Extract metadata
            metadata = data.get("metadata", {})
            case_num = metadata.get("caseNumber", {})
            solver = metadata.get("solver", {})

            # Format case number
            case_number = decision_data.get("jednaciCislo", "")
            if not case_number and case_num:
                case_number = f"{case_num.get('senate', '')}/{case_num.get('registry', '')}/{case_num.get('index', '')}/{case_num.get('year', '')}"

            # Format judge name
            judge_name = ""
            if solver:
                parts = [
                    solver.get("titlesBefore", ""),
                    solver.get("firstName", ""),
                    solver.get("lastName", ""),
                    solver.get("titlesAfter", ""),
                ]
                judge_name = " ".join(p for p in parts if p).strip()

            # Legal references
            legal_refs = format_legal_references(metadata.get("regulations", []))

            return CourtDecision(
                ecli=metadata.get("ecli", decision_data.get("ecli", "")),
                case_number=case_number,
                court=decision_data.get("soud", ""),
                court_code=metadata.get("courtCode", ""),
                date_issued=metadata.get("decisionAt", decision_data.get("datumVydani", "")),
                date_published=metadata.get("publishedAt", decision_data.get("datumZverejneni", "")),
                decision_type=metadata.get("type", ""),
                case_subject=metadata.get("caseSubject", decision_data.get("predmetRizeni", "")),
                case_result=metadata.get("caseResultType", []),
                judge_name=judge_name,
                judge_function=solver.get("function", ""),
                keywords=metadata.get("flags", []),
                legal_references=legal_refs,
                source_url=odkaz_url,
                header_text=header_text,
                verdict_text=verdict_text,
                justification_text=justification_text,
                information_text=information_text,
                full_text=full_text,
            )

        except Exception as e:
            logger.debug(f"Error fetching full decision: {e}")
            return None

    def fetch_single_decision(
        self, decision_data: Dict
    ) -> Tuple[Dict, Optional[CourtDecision]]:
        """Fetch a single decision with its text."""
        odkaz = decision_data.get("odkaz")
        if not odkaz:
            return decision_data, None

        decision = self.fetch_full_decision(odkaz, decision_data)
        return decision_data, decision

    def process_decisions_batch(self, decisions: List[Dict]):
        """Process a batch of decisions: fetch text, chunk, embed, upload."""
        if not decisions:
            return

        logger.info(f"⚡ Processing batch of {len(decisions)} decisions...")
        batch_start = time.time()

        # Parallel text fetching
        all_decisions = []
        with ThreadPoolExecutor(max_workers=MAX_WORKERS_TEXT_FETCH) as executor:
            futures = {
                executor.submit(self.fetch_single_decision, d): d for d in decisions
            }
            for future in as_completed(futures):
                try:
                    _, decision = future.result()
                    if decision:
                        all_decisions.append(decision)
                        self.stats["text_fetch_success"] += 1
                    else:
                        self.stats["text_fetch_errors"] += 1
                except Exception as e:
                    logger.debug(f"Error in parallel fetch: {e}")
                    self.stats["text_fetch_errors"] += 1

        if not all_decisions:
            return

        # Create chunks from all decisions
        all_chunks = []
        for decision in all_decisions:
            chunks = create_decision_chunks(decision)
            all_chunks.extend(chunks)
            self.stats["decisions_processed"] += 1

        if not all_chunks:
            return

        self.stats["chunks_created"] += len(all_chunks)
        logger.info(f"   Created {len(all_chunks)} chunks from {len(all_decisions)} decisions")

        # Embed chunks
        embed_start = time.time()
        chunk_texts = [c.chunk_text for c in all_chunks]
        
        try:
            vectors = self.model.encode(
                chunk_texts,
                batch_size=EMBEDDING_BATCH_SIZE,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            embed_time = time.time() - embed_start
            self.stats["embedding_time"] += embed_time
            logger.info(f"   Embedded {len(vectors)} chunks in {embed_time:.2f}s")
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            self.stats["errors"] += 1
            return

        # Create Qdrant points
        points = []
        for chunk, vector in zip(all_chunks, vectors):
            # Generate deterministic ID from chunk_id
            point_id = int(hashlib.md5(chunk.chunk_id.encode()).hexdigest(), 16) % (
                2**63 - 1
            )

            payload = {
                "chunk_text": chunk.chunk_text,
                **chunk.metadata,
            }

            points.append(
                PointStruct(id=point_id, vector=vector.tolist(), payload=payload)
            )

        # Upload in batches
        upload_start = time.time()
        uploaded = 0
        for i in range(0, len(points), QDRANT_UPSERT_BATCH):
            batch = points[i : i + QDRANT_UPSERT_BATCH]
            for attempt in range(MAX_RETRIES):
                try:
                    self.client.upsert(
                        collection_name=COLLECTION_NAME, points=batch, wait=True
                    )
                    uploaded += len(batch)
                    break
                except Exception as e:
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(RETRY_DELAY * (attempt + 1))
                    else:
                        logger.error(f"Failed to upload batch: {e}")
                        self.stats["errors"] += 1

        upload_time = time.time() - upload_start
        self.stats["upload_time"] += upload_time
        self.stats["chunks_uploaded"] += uploaded

        batch_time = time.time() - batch_start
        logger.info(
            f"   ✅ Uploaded {uploaded} chunks in {upload_time:.2f}s "
            f"(total batch: {batch_time:.2f}s)"
        )

    def get_years(self) -> List:
        return self.api_request(BASE_URL, "years") or []

    def get_months(self, year: int) -> List:
        return self.api_request(f"{BASE_URL}/{year}", f"months for {year}") or []

    def get_days(self, year: int, month: int) -> List:
        return (
            self.api_request(f"{BASE_URL}/{year}/{month}", f"days for {year}/{month}")
            or []
        )

    def get_decisions(self, year: int, month: int, day: int, page: int = 0) -> Tuple[List, int]:
        data = self.api_request(
            f"{BASE_URL}/{year}/{month}/{day}?page={page}",
            f"decisions for {year}/{month}/{day} page {page}",
        )
        if data and isinstance(data, dict):
            return data.get("items", []), data.get("totalPages", 1)
        return [], 1

    def run(self):
        """Main execution pipeline."""
        start_time = time.time()

        processed_days = self.checkpoint.get('processed_days', [])
        logger.info(f"📊 Previously processed: {len(processed_days)} days")
        if processed_days:
            logger.info(f"📅 Last processed: {max(processed_days)}")
        else:
            logger.info("📅 Starting fresh - will process ALL available data")
        logger.info("=" * 70 + "\n")

        self.setup_collection()

        # Fetch years
        years_data = self.get_years()
        if not years_data:
            logger.error("❌ Failed to fetch years from API")
            return

        logger.info(f"📆 Found {len(years_data)} years in database")

        # Accumulator for batching
        accumulated_decisions = []
        last_checkpoint_date = None

        # Process each year
        for year_obj in sorted(years_data, key=lambda x: x.get("rok", 0)):
            year = year_obj.get("rok")
            if not year:
                continue

            year_total = year_obj.get("pocet", 0)
            logger.info(f"\n📅 Year {year} ({year_total:,} decisions)")

            months_data = self.get_months(year)
            if not months_data:
                continue

            for month_obj in sorted(months_data, key=lambda x: x.get("mesic", 0)):
                month = month_obj.get("mesic")
                if not month:
                    continue

                month_total = month_obj.get("pocet", 0)
                logger.info(f"  📆 Month {year}/{month:02d} ({month_total:,} decisions)")

                days_data = self.get_days(year, month)
                if not days_data:
                    continue

                for day_obj in days_data:
                    day_date = day_obj.get("datum")
                    if not day_date or not self.should_process_date(day_date):
                        continue

                    day_count = day_obj.get("pocet", 0)
                    day = int(day_date.split("-")[-1])
                    logger.info(f"    📅 {day_date} ({day_count} decisions)")

                    # Fetch all pages for this day
                    page = 0
                    total_pages = 1
                    while page < total_pages:
                        decisions, total_pages = self.get_decisions(year, month, day, page)
                        if decisions:
                            self.stats["decisions_fetched"] += len(decisions)
                            accumulated_decisions.extend(decisions)
                        page += 1

                    last_checkpoint_date = day_date
                    self.days_since_checkpoint += 1

                    # Process batch when accumulated enough
                    if len(accumulated_decisions) >= DECISIONS_BATCH_SIZE:
                        self.process_decisions_batch(accumulated_decisions)
                        accumulated_decisions = []

                    # Periodic checkpoint
                    if self.days_since_checkpoint >= CHECKPOINT_FREQUENCY:
                        if last_checkpoint_date not in self.checkpoint.get("processed_days", []):
                            self.checkpoint.setdefault("processed_days", []).append(last_checkpoint_date)
                        self.checkpoint["total_chunks"] = self.stats["chunks_uploaded"]
                        save_checkpoint(self.checkpoint)
                        self.days_since_checkpoint = 0
                        logger.info(f"    💾 Checkpoint saved at {last_checkpoint_date}")

        # Process remaining
        if accumulated_decisions:
            self.process_decisions_batch(accumulated_decisions)

        # Final checkpoint
        if last_checkpoint_date:
            if last_checkpoint_date not in self.checkpoint.get("processed_days", []):
                self.checkpoint.setdefault("processed_days", []).append(last_checkpoint_date)
            self.checkpoint["total_chunks"] = self.stats["chunks_uploaded"]
            save_checkpoint(self.checkpoint)

        # Final report
        elapsed = time.time() - start_time
        hours, remainder = divmod(int(elapsed), 3600)
        minutes, seconds = divmod(remainder, 60)

        try:
            info = self.client.get_collection(COLLECTION_NAME)
            total_vectors = info.points_count
        except:
            total_vectors = "Unknown"

        logger.info("\n" + "=" * 70)
        logger.info("✅ PIPELINE COMPLETE!")
        logger.info("=" * 70)
        logger.info(f"Time: {hours}h {minutes}m {seconds}s")
        logger.info("\n📊 Statistics:")
        logger.info(f"  Decisions fetched: {self.stats['decisions_fetched']:,}")
        logger.info(f"  Decisions processed: {self.stats['decisions_processed']:,}")
        logger.info(f"  Chunks created: {self.stats['chunks_created']:,}")
        logger.info(f"  Chunks uploaded: {self.stats['chunks_uploaded']:,}")
        logger.info(f"  Text fetch success: {self.stats['text_fetch_success']:,}")
        logger.info(f"  Text fetch errors: {self.stats['text_fetch_errors']:,}")
        if self.stats["text_fetch_success"] + self.stats["text_fetch_errors"] > 0:
            rate = self.stats["text_fetch_success"] / (
                self.stats["text_fetch_success"] + self.stats["text_fetch_errors"]
            ) * 100
            logger.info(f"  Success rate: {rate:.1f}%")
        logger.info(f"  Errors: {self.stats['errors']}")
        logger.info(f"  Total vectors in Qdrant: {total_vectors}")
        logger.info("\n⏱️ Performance:")
        logger.info(f"  Embedding time: {self.stats['embedding_time']:.2f}s")
        logger.info(f"  Upload time: {self.stats['upload_time']:.2f}s")
        if self.stats["decisions_fetched"] > 0:
            avg = elapsed / self.stats["decisions_fetched"]
            logger.info(f"  Avg per decision: {avg:.3f}s")
        logger.info("=" * 70)


if __name__ == "__main__":
    try:
        pipeline = RAGPipeline()
        pipeline.run()
    except KeyboardInterrupt:
        logger.info("\n\n⚠️ Pipeline interrupted by user")
    except Exception as e:
        logger.error(f"\n\n❌ Fatal error: {e}")
        import traceback
        logger.error(traceback.format_exc())