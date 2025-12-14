# Czech Court Decisions RAG Pipeline

High-quality RAG data pipeline for Czech court decisions from the **justice.cz OpenData API**.

## Quick Start

```bash
python vectorize_new_fast.py
```

## Configuration

Edit the top of `vectorize_new_fast.py`:

```python
# Qdrant connection
QDRANT_HOST = "hopper.proxy.rlwy.net"
QDRANT_PORT = 48447
QDRANT_API_KEY = "..."

# Collection name (change this for new runs)
COLLECTION_NAME = "czech_court_decisions_v2"

# Chunking parameters
CHUNK_SIZE = 1500    # Target chunk size in characters (~375 tokens)
CHUNK_OVERLAP = 200  # Overlap between chunks for context

# Processing
DECISIONS_BATCH_SIZE = 200  # Process in batches
CHECKPOINT_FREQUENCY = 5    # Save checkpoint every N days
```

The script will automatically:
- Process ALL available dates from the API chronologically
- Skip dates already in checkpoint (resumable)
- Save progress every 5 days

## What It Does

1. **Fetches** court decisions from justice.cz API (hierarchical: years → months → days → decisions)
2. **Extracts** only **verdict** and **justification** (excludes header/party details)
3. **Chunks** text at sentence boundaries with overlap
4. **Embeds** using Seznam Czech model (256 dimensions)
5. **Uploads** to Qdrant with rich metadata

## API Structure

```
BASE_URL = "https://rozhodnuti.justice.cz/api/opendata"

GET /                           → List of years
GET /{year}                     → List of months  
GET /{year}/{month}             → List of days
GET /{year}/{month}/{day}       → Paginated decisions
GET {decision.odkaz}            → Full decision text
```

### Full Decision Structure

```json
{
  "header": [...],           // Party details (EXCLUDED)
  "verdict": [...],          // Court ruling (INCLUDED)
  "verdictText": "...",      // Pre-formatted verdict
  "justification": [...],    // Legal reasoning (INCLUDED)
  "justificationText": "...", // Pre-formatted justification
  "information": [...],      // Procedural notices (EXCLUDED)
  "metadata": {
    "ecli": "ECLI:CZ:OSPR:2024:8.C.193.2024.1",
    "type": "JUDGEMENT",
    "courtCode": "OSPR",
    "caseSubject": "o určení vlastnictví",
    "caseResultType": ["VYHOVENI"],
    "regulations": [...],
    "flags": ["SMLOUVA_KUPNI"]
  }
}
```

## Payload Structure (What Gets Stored)

Each chunk stored in Qdrant has this structure:

```json
{
  "chunk_text": "1. Žalobce se podanou žalobou...",
  "ecli": "ECLI:CZ:OSPR:2024:8.C.193.2024.1",
  "case_number": "8 C 193/2024-23",
  "court": "Okresní soud v Přerově",
  "court_code": "OSPR",
  "date_issued": "2024-10-16",
  "date_published": "2024-12-09",
  "decision_type": "JUDGEMENT",
  "case_subject": "o určení vlastnictví osobního automobilu",
  "case_result": ["VYHOVENI"],
  "section_type": "justification",
  "chunk_index": 0,
  "total_chunks": 6,
  "has_full_text": true,
  "full_text": "VÝROK:\n...\n\nODŮVODNĚNÍ:\n...",  // Only in chunk 0
  "keywords": ["SMLOUVA_KUPNI"],
  "legal_references": ["§ 80 zák. č. 99/1963 Sb.", ...]
}
```

## Qdrant Collections

| Collection | Points | Description |
|------------|--------|-------------|
| `czech_constitutional_court` | 510,523 | Constitutional court decisions (existing) |
| `czech_court_decisions_v2` | - | New collection from this pipeline |

### Constitutional Court Structure (for reference)

```
case_number: "III.ÚS 323/15"
chunk_index: 0
chunk_text: (1390 chars)
date: "6/11/2015"
filename: "3-323-15_1.json"
full_text: (11095 chars)
has_full_text: True
total_chunks: 8
```

## Chunking Strategy

### Hybrid Chunking
- **Semantic**: Splits at Czech sentence boundaries (. ! ?)
- **Structural**: Keeps chunks within sections (verdict, justification)
- **Overlap**: 200 chars overlap for context continuity
- **Size**: Target 1500 chars, minimum 100 chars

### What Gets Stored
- ✅ **VÝROK** (Verdict) - The court's ruling
- ✅ **ODŮVODNĚNÍ** (Justification) - Legal reasoning
- ❌ **Header** - Party details, names, addresses (excluded)
- ❌ **Information** - Procedural notices (excluded)

## Embedding Model

```python
EMBEDDING_MODEL = "Seznam/retromae-small-cs"
DENSE_VECTOR_SIZE = 256
```

- Czech-optimized (better than multilingual models)
- 256 dimensions (efficient)
- Normalized embeddings for cosine similarity

## Resume & Checkpoint

The pipeline saves progress to `rag_checkpoint.json`:

```json
{
  "last_date": "2024-01-15",
  "processed_days": ["2024-01-01", "2024-01-02", ...],
  "total_chunks": 12345
}
```

To restart from scratch, delete this file.

## Logs

- `rag_pipeline.log` - Full processing log
- `rag_errors.log` - Errors only

## Files

```
vectorize_new_fast.py    # Main pipeline script
vectorize_and_upload.py  # Original reference script (for constitutional court JSON files)
rag_checkpoint.json      # Resume checkpoint (auto-created)
rag_pipeline.log         # Processing log (auto-created)
rag_errors.log           # Error log (auto-created)
```

## Usage for LawGPT

### Semantic Search
```python
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("Seznam/retromae-small-cs")
client = QdrantClient(host="...", port=48447, api_key="...")

query = "odpovědnost za škodu způsobenou provozem vozidla"
vector = model.encode(query, normalize_embeddings=True)

results = client.search(
    collection_name="czech_court_decisions_v2",
    query_vector=vector.tolist(),
    limit=5,
)

for r in results:
    print(f"Score: {r.score:.3f}")
    print(f"Case: {r.payload['case_number']}")
    print(f"Text: {r.payload['chunk_text'][:200]}...")
```

### Get Full Context
```python
# If you need full decision text, get chunk 0 by ECLI
ecli = result.payload['ecli']
full_result = client.scroll(
    collection_name="czech_court_decisions_v2",
    scroll_filter={"must": [
        {"key": "ecli", "match": {"value": ecli}},
        {"key": "chunk_index", "match": {"value": 0}}
    ]},
    limit=1,
    with_payload=True
)
full_text = full_result[0][0].payload['full_text']
```

### Filter by Metadata
```python
# Search only judgements from specific court
results = client.search(
    collection_name="czech_court_decisions_v2",
    query_vector=vector.tolist(),
    query_filter={
        "must": [
            {"key": "decision_type", "match": {"value": "JUDGEMENT"}},
            {"key": "court_code", "match": {"value": "OSPR"}}
        ]
    },
    limit=10,
)
```
