# Civic-AI

Civic-AI is a document-grounded question-answering system. Upload PDFs, ask questions in plain English, and get answers backed by citations to the exact source pages — with a clear "not enough evidence" response when the documents don't cover it.

It's domain-independent: works for education, health, law, finance, policy, or any other PDF-based content. No hard-coded schema or fixed set of attributes required.

## Features

- 📄 **PDF ingestion** — parses, chunks, and indexes documents automatically
- 🔍 **Hybrid retrieval** — combines semantic search (FAISS) with a knowledge graph (NetworkX) for better context
- 🤖 **Grounded answers** — powered by Gemini 2.5 Flash, always cited to source pages
- ✅ **Answer validation** — checks that claims in the answer are actually supported by evidence
- ⚡ **Caching** — Redis-backed caching for fast repeat queries
- 📥 **Async uploads** — optional RabbitMQ queue for background ingestion of large PDFs

## Tech Stack

| Component | Technology |
|---|---|
| API | Flask 3 |
| LLM | Google Gemini 2.5 Flash |
| Embeddings | Sentence Transformers (`all-MiniLM-L6-v2`) |
| Vector search | FAISS |
| Graph search | NetworkX |
| Document store | MongoDB |
| Cache | Redis |
| Job queue | RabbitMQ |
| PDF parsing | pdfplumber |

## How It Works

![Civic-AI Architecture](assets/architecture.png)

**Document Ingestion Pipeline**
`Parser` extracts text and page metadata from PDFs → `Cleaner` removes noise and normalizes text → `Chunker` splits text into meaningful chunks → `Embedder` generates vector embeddings (`all-MiniLM-L6-v2`) → `RabbitMQ` queues the job for async processing.

**AI Pipeline**
`Planner` understands the query and identifies information needs → `Step Definer` breaks it into retrieval steps → `Retriever` runs hybrid search (FAISS + Graph) → `Generator` produces an answer with Gemini 2.5 Flash → `Validator` checks the answer's grounding against retrieved evidence → a cited **Final Response** is returned.

**Storage & Indexes**
- **MongoDB** — document metadata, chunks, vectors, graph data, job status/cache
- **FAISS** — vector index for semantic search
- **Graph (NetworkX)** — entity relationships, concept connections, hybrid search support

**External Services**
- **Redis** — caches queries/results and tracks job status
- **RabbitMQ** — async ingestion queue for background processing
- **OCR** *(planned)* — text extraction from scanned/image-based PDFs

## Getting Started

```bash
# Install dependencies
pip install -r requirements.txt

# Ingest your PDFs (place them in data/pdfs/ first)
python -m ingestion.ingest --skip-mongo

# Start the API
python app.py
```

The API runs on `http://localhost:3001`.

**Optional — for async uploads:** start MongoDB, Redis, and RabbitMQ, then run the worker in a separate process:

```bash
python -m worker
```

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/api/v1/health` | GET | Check system status |
| `/api/v1/embed` | POST | Ingest all PDFs in `data/pdfs/` |
| `/api/v1/upload` | POST | Upload a single PDF |
| `/api/v1/status/<job_id>` | GET | Check async ingestion progress |
| `/api/v1/query` | POST | Ask a question |

### Example query

```json
POST /api/v1/query
{
  "query": "What are the main findings of the report?",
  "final_top_k": 5
}
```

## Performance: Caching & Async Processing

**Before:** every `/query` call ran the full AI pipeline from scratch, and every `/upload` blocked the HTTP request until ingestion finished — which could take minutes for large PDFs.

**After:**

```
/query  →  Redis?  →  hit:  return cached JSON immediately
                   →  miss: AI pipeline → store in Redis → return

/upload →  RabbitMQ available?  →  yes: enqueue job → 202 + job_id (returns in <1s)
                                →  no:  run sync ingestion (old behaviour, fallback)
```

### `/query` workflow

1. The query text is normalised (lowercased, whitespace collapsed), so `"What is X?"` and `"what is x?"` share the same cache entry.
2. Redis is checked with key `query::<sha256(normalised)>`.
   - **Hit** → the cached JSON is returned immediately.
   - **Miss** → the full pipeline runs: `plan_query()` (Gemini extracts intent/domain/keywords) → `define_steps()` (1–3 retrieval steps) → for each step, `retrieve()` (FAISS + Graph, fused with RRF) → `extract_evidence()` → `generate_response()` (Gemini answer with citations) → `validate_response()` (lexical grounding check, no LLM call) → result is cached in Redis (TTL 1 hour) → returned.

Pass `"no_cache": true` in the request body to bypass the cache for a specific query.

### `/embed` workflow

Runs full ingestion over every PDF in `data/pdfs/` (parsing and chunking are skipped if already cached in `data/processed/`). After ingestion:
- the in-process FAISS/graph/embedder context is reset, and
- all `query::*` and `retrieval::*` keys are flushed from Redis, so stale answers are never served.

### `/upload` async workflow

1. The file is validated and saved to `data/pdfs/`.
2. A job is published to the `docqa.ingest` RabbitMQ queue, and Redis stores `job::<job_id>` with status `queued`.
3. The API immediately returns `202 Accepted` with a `job_id`.
4. A separate worker process (`python -m worker`) consumes the queue, runs ingestion, flushes caches, and updates the job status to `completed` or `failed`.
5. The client polls `GET /api/v1/status/<job_id>` to check progress.

If RabbitMQ isn't running, `/upload` automatically falls back to synchronous ingestion and returns `200` instead of `202`.

### Running locally with caching + async uploads

```bash
# install dependencies
pip install -r requirements.txt

# start Redis (Docker)
docker run -d -p 6379:6379 redis:7

# start RabbitMQ (Docker)
docker run -d -p 5672:5672 -p 15672:15672 rabbitmq:3-management
```

```bash
# terminal 1 — API server
cd backend
python app.py

# terminal 2 — ingestion worker (only needed for async upload)
cd backend
python -m worker
```

```bash
# health check
curl http://localhost:3001/api/v1/health
```

## Project Structure

```
backend/
├── app.py                 # Flask API entry point
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables (API keys, DB URI, etc.)
├── data/
│   ├── pdfs/               # Source PDF documents
│   └── indexes/
│       ├── faiss/            # FAISS index files
│       └── graph/            # Graph database/index files
├── ingestion/              # PDF parsing, cleaning, chunking, embeddings
│   ├── parser.py
│   ├── chunker.py
│   └── embedder.py
├── retrieval/              # FAISS + graph hybrid retrieval
│   ├── faiss.py
│   ├── graph.py
│   └── hybrid.py
├── ai/                     # Core AI pipeline
│   ├── planner.py            # Understand user query
│   ├── step_definer.py       # Create retrieval steps
│   ├── generator.py          # Generate answer (Gemini)
│   ├── validator.py          # Check evidence grounding
│   └── pipeline.py           # Orchestrate the AI flow
├── database/
│   └── mongodb.py           # MongoDB connection & CRUD
├── agents/
│   └── crew.py               # Agentic orchestration (optional)
└── services/                # Additional infrastructure
    ├── redis.py               # Redis caching
    ├── rabbitmq.py            # Message queue
    └── ocr.py                 # OCR for scanned PDFs
```

### Development Phases

1. **Ingestion** — Parsing → Chunking → Embedding
2. **Retrieval** — FAISS + Graph (hybrid)
3. **AI Pipeline** — Planner → Step Definer → Retriever → Generator → Validator
4. **Integration** — `app.py` + API
5. **Agentic Orchestration** — optional agent-based workflow
6. **Extra Features** — Redis + RabbitMQ + OCR

## Known Limitations

- No OCR — only text-based PDFs are supported
- Grounding validation uses lexical overlap, so it may miss paraphrased claims
- No authentication, rate limiting, or upload size limits yet
- Ingestion is synchronous unless RabbitMQ is running
- Re-run ingestion with `rebuild=true` after changing source files