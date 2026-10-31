# ResearchPaper AI

An evidence-grounded research assistant. Upload research papers (PDF), ask questions, and receive answers generated from retrieved passages, with validated source citations (document, page, passage).

> **Status:** Complete and containerized: ingestion, semantic retrieval, cross-encoder reranking, cited answer generation, feedback, automated tests, an evaluation suite and a Docker build. Not yet deployed to a public host.

## Problem

General-purpose chatbots answer questions about papers without showing where the answer came from. This project ties every claim to a source passage, a document and a page, and declines to answer when the retrieved evidence is insufficient.

## Architecture

```
PDF upload
   |  validate (extension, %PDF header, size limit)
   v
PyMuPDF extraction (per page) -> cleaning -> chunking (160 words, 25 overlap, within a page)
   |
   v
Embeddings (BAAI/bge-small-en-v1.5, 384-dim, normalized) -> PostgreSQL + pgvector

Question
   |
   v
Embed question -> top-20 candidate chunks (cosine similarity)
   |
   v
Evidence gate: best vector score below MIN_SCORE -> "not enough evidence", no LLM call
   |
   v
Cross-encoder reranker -> top-6 passages (assigned IDs S1..S6)
   |
   v
Gemini generates an answer citing [S#] markers
   |
   v
Server validates citations (unknown IDs removed), maps IDs to real chunks/pages
   |
   v
Answer + citations saved in PostgreSQL (question, answer, model, latency, retrieved chunks)
```

Chunks never cross page boundaries, so every chunk has an exact page number. Citations are mapped to database records on the server; the model does not supply page numbers.

## Tech stack

| Component | Technology |
|---|---|
| API | FastAPI, Pydantic |
| Database | PostgreSQL (Neon), SQLAlchemy 2.0 |
| Vector search | pgvector (exact cosine search) |
| PDF processing | PyMuPDF |
| Embeddings | sentence-transformers, `BAAI/bge-small-en-v1.5` |
| Reranking | Cross-encoder `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Generation | Gemini API (`google-genai`), model set via config |
| Testing | pytest, httpx |
| Packaging | Docker, Docker Compose |

## Project structure

```
app/
  main.py
  api/routes/        # documents.py, questions.py
  core/              # config.py, security.py
  db/session.py
  models/            # document, chunk, question, retrieved_chunk, feedback
  schemas/
  services/          # pdf_parser, chunker, embedder, retriever,
                     # reranker, generator, rag_pipeline
scripts/
  eval_utils.py
  evaluate_retrieval.py     # vector vs reranked retrieval, candidate sweep
  compare_encoders.py       # embedding model comparison
  compare_chunk_sizes.py    # chunk size comparison
tests/
data/
  uploads/
  eval_questions.json
  results/*.csv
Dockerfile
docker-compose.yml
```

Routes handle HTTP, models handle persistence, services hold the pipeline. Only `generator.py` depends on the LLM provider.

## Setup

Requirements: Python 3.11+, a PostgreSQL database with the `pgvector` extension (for example a free Neon project), and a Gemini API key from Google AI Studio.

```bash
git clone <your-repo-url>
cd researchpaper-ai
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # then edit .env
uvicorn app.main:app --reload
```

`.env`:

```
DATABASE_URL=postgresql+psycopg://<user>:<password>@<host>/<db>?sslmode=require
UPLOAD_DIR=data/uploads
MAX_UPLOAD_MB=20
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
GEMINI_API_KEY=<your-key>
GENERATION_MODEL=gemini-2.5-flash
CANDIDATES=20
TOP_K=6
MIN_SCORE=0.68
USE_RERANKER=true
APP_API_KEY=<optional shared secret>
TEST_DATABASE_URL=<separate database, tests only>
```

The app enables the `vector` extension and creates tables on startup. The first upload and first question download the embedding and reranker models (not needed in Docker, where they are baked into the image).

Interactive API docs: `http://127.0.0.1:8000/docs`

> **Privacy:** retrieved passages are sent to the Gemini API with every question. On Google's free tier, prompts may be used to improve its products, so do not upload confidential or unpublished papers.

## Run with Docker

```bash
docker compose build
docker compose up
```

The image uses CPU-only PyTorch and downloads both models at build time, so containers do not fetch anything at runtime (image size about 2.9 GB). Configuration comes from `.env` at runtime; no secrets are copied into the image. The database is the hosted Postgres from `DATABASE_URL`.

If `APP_API_KEY` is set, every endpoint except `GET /health` requires an `X-API-Key` header (use **Authorize** on the `/docs` page). All users share one document store, so a public demo should always be key-protected.

## API

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Health check (always open) |
| POST | `/documents` | Upload a PDF; extracts, chunks and embeds it |
| GET | `/documents` | List documents |
| GET | `/documents/{id}` | Get one document |
| DELETE | `/documents/{id}` | Delete a document, its chunks, sources and file |
| GET | `/documents/{id}/chunks` | Inspect chunks (temporary, for debugging) |
| POST | `/questions` | Ask a question over selected (or all) documents |
| GET | `/questions/{id}` | Retrieve a stored question, answer and citations |
| POST | `/questions/{id}/feedback` | Store a rating (1-5) and optional comment |

Example request:

```json
{
  "question": "How does the MRKL architecture route inputs to expert modules?",
  "document_ids": [1]
}
```

Response shape (values elided):

```json
{
  "question_id": 0,
  "answer": "... [S1] ... [S3]",
  "citations": [
    {"source_id": "S1", "document_id": 0, "filename": "...", "page_number": 0, "chunk_id": 0, "snippet": "...", "text": "..."}
  ],
  "latency_ms": 0
}
```

Errors: `400` invalid or unreadable PDF, `401` missing or wrong API key (when enabled), `404` unknown document or question, `413` file too large, `422` no extractable text (scanned PDFs unsupported) or invalid request, `502` generation failure.

## Testing

```bash
pip install -r requirements-dev.txt
pytest -q
```

The suite covers upload validation (wrong type, fake PDF, oversized, no text), ingestion (page numbers, embedding storage), document CRUD and cascade deletion, question scoping to selected documents, citation validation (unknown source IDs removed, citations map to real chunks), the insufficient-evidence path, generation failure handling, persistence, feedback validation and API-key protection. Embeddings, reranker and LLM are replaced by deterministic fakes, so tests need no model downloads or Gemini access. They run against a separate PostgreSQL database with pgvector (`TEST_DATABASE_URL`), and refuse to run if it equals `DATABASE_URL`.

## Evaluation

### Setup

- 30 hand-written questions over 3 papers: 24 answerable (factual, methodology, comparison, limitations) and 6 with no answer in the paper (a mix of unrelated and on-topic questions).
- Labels are page-level: a hit means a retrieved chunk comes from any page listed as containing the answer. Metrics: Hit@k (an expected page appears in the top k chunks) and MRR (reciprocal rank of the first hit).
- Every script reads `data/eval_questions.json` and writes per-question rows to `data/results/`.
- **Label audit:** after seeing results, I inspected cases where the top-ranked page differed from the label. In two questions the top page also contained the answer, so I added it to the expected pages. Because this happened after seeing results, the final configuration is reported with and without the second correction.

### Final configuration: vector search vs. reranking

160-word chunks with 25-word overlap, 20 candidates reranked, top 6 to the LLM.

| Metric | Vector only | + Reranker | Vector only (before Q13 label fix) | + Reranker (before Q13 label fix) |
|---|---|---|---|---|
| Hit@1 | 0.625 | 0.750 | 0.625 | 0.708 |
| Hit@3 | 0.875 | 0.958 | 0.833 | 0.958 |
| Hit@6 | 0.958 | 1.000 | 0.958 | 1.000 |
| MRR | 0.753 | 0.856 | 0.746 | 0.835 |

Reranking improved the expected-page rank for 6 of 24 questions, left 18 unchanged and made none worse.

### Candidate count (160/25, reranked)

| Candidates | Hit@1 | Hit@3 | Hit@6 | MRR | Median rerank time |
|---|---|---|---|---|---|
| 10 | 0.792 | 0.958 | 0.958 | 0.874 | 327 ms |
| 15 | 0.792 | 0.958 | 0.958 | 0.874 | 525 ms |
| 20 (used) | 0.750 | 0.958 | 1.000 | 0.856 | 749 ms |

Differences are one question each. 20 candidates were kept because the LLM sees all six passages, so getting an expected page into them (Hit@6) matters more than rank 1.

### Chunk size (vector search only, `bge-small`)

Measured with the labels at the time, before the final label corrections.

| Chunk / overlap (words) | Chunks | Hit@1 | Hit@3 | Hit@6 | MRR |
|---|---|---|---|---|---|
| 160 / 25 (used) | 253 | 0.625 | 0.833 | 0.958 | 0.746 |
| 320 / 50 | 133 | 0.542 | 0.792 | 0.875 | 0.674 |
| 480 / 75 | 100 | 0.458 | 0.708 | 0.833 | 0.620 |

Smaller chunks scored better at every metric, and the trend held when the full pipeline was re-indexed at 160 words. With 320-word chunks, 23.3% of chunks exceeded the embedding model's 512-token limit and were silently truncated (truncation was not measured at 160 words). Page-level hits are slightly easier to get with smaller chunks, since each page has more of them, and this is a limitation of the metric.

### Embedding models (vector search only, 320-word chunks)

| Model | Hit@1 | Hit@3 | Hit@6 | MRR |
|---|---|---|---|---|
| bert-base-uncased (raw, mean pooling) | 0.458 | 0.708 | 0.792 | 0.609 |
| all-MiniLM-L6-v2 | 0.458 | 0.833 | 0.917 | 0.653 |
| bge-small-en-v1.5 (used) | 0.542 | 0.792 | 0.875 | 0.674 |
| bge-base-en-v1.5 | 0.500 | 0.833 | 0.958 | 0.655 |

The three sentence-embedding models differ by one to three questions, so they are effectively tied; `bge-small` was kept as the smallest. Raw BERT was worst on every metric. all-MiniLM-L6-v2 has a 256-token limit, so 84.2% of 320-word chunks were truncated for it.

### Unanswerable questions

- A cosine-similarity gate rejects clearly unrelated questions before the LLM is called.
- Neither score separates answerable from on-topic unanswerable questions reliably. With 160-word chunks, one answerable question's best reranker score (-2.52) was lower than that of an unanswerable one (-0.76), and the lowest answerable cosine score (0.737) was only 0.002 below an unanswerable one (0.739). A numeric abstention threshold was therefore not built; the generation prompt, which instructs the model to say when the passages do not answer the question, is the main safeguard.
- In a manual check, the model declined to state a GPU-hours and cost figure that the paper does not give.

### Latency

Median reranking time for 20 candidates (160-word chunks, laptop CPU): 749 ms. With the earlier 320-word index, mean reranking time was 1.6 s on a warm run (a first run was 4.0 s because it included one-time model loading). Retrieval embeds the question and queries a hosted Neon database over a mobile hotspot. The Gemini call accounts for most of the end-to-end time; one full question took 9.3 s on the 320-word index.

## Known limitations

- Text-based PDFs only; no OCR.
- Chunks do not cross page boundaries, so passages continuing across pages are split.
- Dense retrieval can blur exact terms and numbers; no keyword or hybrid search.
- Citation validation checks that source IDs exist, not that each claim is supported; answer grounding has been checked only by hand on individual answers.
- The evaluation set is small (24 answerable questions, 3 papers), single-annotator and page-level; the configuration was chosen on the same questions it is reported on, and two labels were corrected after seeing results.
- No numeric abstention threshold beyond the cosine gate for clearly unrelated questions.
- Single shared document store; access control is one shared API key, not per-user accounts.
- Ingestion is synchronous; chunk size is measured in words as a proxy for tokens.
- The Docker image is about 2.9 GB, and the app has not been deployed to a public host.

## Roadmap

- [x] FastAPI service, file upload
- [x] PostgreSQL persistence, document CRUD
- [x] PDF extraction, cleaning, chunking
- [x] Embeddings, pgvector, semantic retrieval
- [x] Cited answer generation, question history
- [x] Cross-encoder reranking and evaluation (encoders, chunk size, candidate count)
- [x] Feedback endpoint, automated tests
- [x] Docker build and shared-key access protection
- [ ] Cloud deployment
- [ ] Systematic answer-grounding evaluation