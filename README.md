# Civic-AI

Civic-AI is a document-grounded question-answering backend. Users upload PDF documents and ask questions in natural language. The system retrieves relevant passages, generates an answer with source-page citations, and validates whether the answer is supported by the indexed documents.

The application is domain-independent. It can work with policy, education, health, law, finance, or other text-based PDF collections.

## Features

- Hybrid retrieval — FAISS semantic search + NetworkX knowledge graph
- Gemini-based grounded answer generation with source-page citations
- Automatic single-hop and multi-hop query planning
- Answer validation with a heuristic fallback
- Redis query caching + optional async ingestion via RabbitMQ

## Architecture

![Civic-AI Architecture](backend/assests/architecture.png)

![How it Works](backend/assests/sequence.png)

## Setup

```bash
cd backend
pip install -r requirements.txt
```

Configure `backend/.env` — see [Environment Variables](#environment-variables).

Start the API:

```bash
python app.py
```

The API listens on `http://localhost:3001`.

## Ingestion

- Place PDFs in `backend/rag_pipeline/data/pdfs/`
- Each PDF is parsed, chunked, and embedded
- Results are stored in FAISS, the knowledge graph, and (optionally) MongoDB
- Re-run with `rebuild=true` after changing source documents — this also clears stale Redis caches

**Run ingestion:**

```bash
python -m rag_pipeline.ingestion.ingest --skip-mongo
```

**For async PDF uploads** (via the `/api/v1/upload` endpoint), start Redis, RabbitMQ, and a worker process:

```bash
# start Redis (Docker)
docker run -d -p 6379:6379 redis:7

# start RabbitMQ (Docker)
docker run -d -p 5672:5672 -p 15672:15672 rabbitmq:3-management

python -m rag_pipeline.services.worker
```

If RabbitMQ isn't running, uploads fall back to synchronous ingestion automatically.

## Query

Every question is automatically routed as single-hop or multi-hop:

**Single-hop** — one retrieval step is enough:
- Plans the query once
- Retrieves evidence in one hybrid search
- Generates and validates one cited answer
- Up to 3 LLM calls total

**Multi-hop** — the question needs several related findings (e.g. comparing two topics):
- Breaks the question into up to 3 sequential sub-questions
- Each sub-question retrieves its own evidence and generates its own answer
- Each step sees the previous steps' answers as context
- The final step synthesizes everything into one answer, which is then validated
- Up to 5 LLM calls total

**Ask a question:**

```bash
curl -X POST http://localhost:3001/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the main goals of the policy?", "final_top_k": 5}'
```

Repeat queries are served from the Redis cache (`cached: true` in the response). Pass `"no_cache": true` to bypass it for a single request.

## Docker

```bash
docker build -t civic_ai:latest .
docker tag civic_ai:latest <your_registry>/civic_ai:latest
docker push <your_registry>/civic_ai:latest
```

Change environmental variables in .env, then run

```bash
docker compose up -d
```

## Environment Variables

Configure these in `backend/.env`:

```text
MONGO_URI=...
DOCKERHUB_USERNAME=...
IMAGE_TAG=...
MONGO_URI_DOCKER=...
APP_DB_NAME=...
GEMINI_API_KEY_1=...
GEMINI_API_KEY_2=...
GEMINI_API_KEY_3=...
MONGO_URI=mongodb://localhost:27017
REDIS_URL=redis://localhost:6379
RABBITMQ_URL=amqp://guest:guest@localhost:5672/
FAISS_INDEX_DIR=...
GRAPH_INDEX_DIR=...
QUERY_CACHE_TTL=3600
RETRIEVAL_CACHE_TTL=1800
JOB_STATUS_TTL=86400
```

Gemini keys are rotated automatically across planning, generation, and validation calls. No key is hard-coded in the source.

## API

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/v1/health` | Check indexes and optional services |
| POST | `/api/v1/embed` | Synchronously ingest all PDFs |
| POST | `/api/v1/upload` | Upload a PDF for async or sync ingestion |
| GET | `/api/v1/status/<job_id>` | Read asynchronous ingestion status |
| POST | `/api/v1/query` | Ask a document-grounded question |

### Query response

```json
{
  "query": "What are the main goals of the policy?",
  "is_multi_hop": false,
  "answer": "...",
  "is_sufficient": true,
  "is_valid": true,
  "grounding_score": 0.92,
  "quality_score": 0.9,
  "offensive_score": 0.0,
  "citations": ["Policy p5"],
  "steps": [
    {
      "step_number": 1,
      "sub_query": "What are the main goals of the policy?",
      "answer": "...",
      "evidence": ["Policy p5"],
      "score": 0.88,
      "is_sufficient": true,
      "is_final": true
    }
  ],
  "flagged_sentences": [],
  "validation_note": "...",
  "elapsed_seconds": 1.23,
  "cached": false
}
```

## File Responsibilities

```text
backend/
├── app.py                         Flask entry point and API routes
├── requirements.txt               Python dependencies
├── .env                           Local configuration and secrets
├── rag_pipeline/
│   ├── ai/
│   │   ├── planner.py             Query planning and max-three-step limit
│   │   ├── step_definer.py        Domain detection for each step
│   │   ├── pipeline.py             End-to-end query orchestration
│   │   ├── extractor.py            Per-step retrieval adapter
│   │   ├── generator.py            Grounded Gemini answer generation
│   │   ├── validator.py            Answer validation and fallback heuristic
│   │   └── key.py                  Rotating Gemini key management
│   ├── ingestion/
│   │   ├── parser.py               PDF text, pages, and section detection
│   │   ├── chunker.py               Sentence-based overlapping chunks
│   │   ├── embedder.py              Sentence Transformer embeddings
│   │   ├── processed.py             Parsed/chunked JSON cache
│   │   ├── domain_index.py          Domain keyword map generation
│   │   └── ingest.py                Full ingestion orchestration
│   ├── retrieval/
│   │   ├── faiss.py                 Vector index build/load/search
│   │   ├── graph.py                 Knowledge graph and graph search
│   │   └── hybrid.py                FAISS/graph RRF fusion
│   ├── database/
│   │   └── mongodb.py               Chunk and embedding persistence
│   ├── memory/
│   │   └── history.py               Per-query multi-hop history
│   ├── services/
│   │   ├── redis.py                 Query cache and job-status store
│   │   ├── rabbitmq.py               Upload queue and consumer
│   │   └── worker.py                 Background ingestion process
│   └── data/
│       ├── pdfs/                     Source PDFs
│       ├── processed/                Parser/chunker cache
│       └── indexes/                  FAISS, graph, and domain artifacts
```

## Known Limitations

- OCR is not implemented; scanned image-only PDFs may produce no text.
- Domain filtering currently passes the first detected domain to graph retrieval.
- Multi-hop history is in memory for one query and is not persisted between requests.
- Retrieval cache helpers exist, but the hybrid retriever currently performs retrieval directly.
- Validation is a quality heuristic and does not prove every claim semantically.
- MongoDB, Redis, and RabbitMQ are optional depending on the selected workflow.
- Authentication, rate limiting, and upload-size limits are not implemented.
