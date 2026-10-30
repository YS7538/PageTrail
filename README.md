# ResearchPaper AI

An evidence-grounded research assistant: upload research papers (PDF), then retrieve the passages most relevant to a question, with document and page references. Answer generation with citations is planned.

> **Status:** Phases 1-4 complete (API, persistence, ingestion, semantic retrieval). Reranking, answer generation and deployment are in progress.

## Problem

General-purpose chatbots answer questions about papers without showing where the answer came from. This project keeps every result tied to a source passage, a document and a page number, so claims can be checked.

## Architecture

```
PDF upload
   |
   v
Validate (extension, %PDF header, size limit)
   |
   v
PyMuPDF text extraction (per page)
   |
   v
Cleaning (hyphenation, whitespace, repeated headers/footers)
   |
   v
Chunking (320-word sliding window, 50-word overlap, within a page)
   |
   v
Embeddings (BAAI/bge-small-en-v1.5, 384-dim, normalized)
   |
   v
PostgreSQL + pgvector  <----  question embedding (cosine distance)
                                      |
                                      v
                           Top-k chunks with doc, page, score
```

Chunks never cross page boundaries, so every chunk has an exact page number.

## Tech stack

| Component | Technology |
|---|---|
| API | FastAPI, Pydantic |
| Database | PostgreSQL (Neon), SQLAlchemy 2.0 |
| Vector search | pgvector (exact cosine search) |
| PDF processing | PyMuPDF |
| Embeddings | sentence-transformers, `BAAI/bge-small-en-v1.5` |

## Project structure

```
app/
  main.py
  api/routes/documents.py
  core/config.py
  db/session.py
  models/        # document, chunk, question, feedback
  schemas/
  services/      # pdf_parser, chunker, embedder, retriever
data/uploads/
```

Routes handle HTTP, models handle persistence, and services hold the pipeline logic.

## Setup

**Requirements:** Python 3.11+, a PostgreSQL database with the `pgvector` extension (for example a free Neon project).

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
```

The app enables the `vector` extension and creates tables on startup. The first upload downloads the embedding model (about 130 MB).

Interactive API docs: `http://127.0.0.1:8000/docs`

## API

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Health check |
| POST | `/documents` | Upload a PDF; extracts, chunks and embeds it |
| GET | `/documents` | List documents |
| GET | `/documents/{id}` | Get one document |
| DELETE | `/documents/{id}` | Delete a document, its chunks and its file |
| GET | `/documents/{id}/chunks` | Inspect chunks (temporary, for debugging) |
| POST | `/retrieve` | Semantic search over chunks (temporary, replaced by `/questions`) |

Example retrieval request:

```json
{
  "question": "What are the main limitations of the proposed method?",
  "document_ids": [1],
  "top_k": 10
}
```

Each result contains `chunk_id`, `document_id`, `filename`, `page_number`, `content` and `score`.

Upload errors: `400` for invalid or unreadable PDFs, `413` for oversized files, `422` for PDFs with no extractable text (scanned PDFs are not supported).

## Evaluation (preliminary)

Retrieval was tested on one paper (MRKL Systems) with 9 hand-written answerable questions and expected page numbers (factual, methodology, comparison, limitations). A hit means the expected page appears among the retrieved chunks. This is a small, single-paper test and should be read as an early signal, not a benchmark.

| Metric | Result |
|---|---|
| Hit@1 | 4/9 |
| Hit@3 | 8/9 |
| Hit@5 | 8/9 |
| MRR (page level) | about 0.63 |

Observations:

- Conceptual questions (methodology, results) reached rank 1 more often than exact-detail questions (a hyperparameter, a token count), where the right page typically ranked 2nd or 3rd.
- Scattered answers, such as a limitations question spread across several pages, were the hardest case.
- Similarity scores do not separate answerable from unanswerable questions on their own. An off-topic question scored 0.41, while an on-topic question the paper does not answer scored 0.77, higher than some correctly retrieved answerable questions. Handling "not enough evidence" will rely on the generation prompt and reranker scores, not a single threshold.

## Known limitations

- Text-based PDFs only; no OCR for scanned documents.
- Chunks do not cross page boundaries, so passages that continue across pages are split.
- Dense retrieval can blur exact terms and numbers; no keyword or hybrid search yet.
- Synchronous ingestion; large documents block the request.
- Chunk size is measured in words as a proxy for tokens.
- Evaluation set is small and covers a single paper.

## Roadmap

- [x] Phase 1: FastAPI service, file upload
- [x] Phase 2: PostgreSQL persistence, document CRUD
- [x] Phase 3: PDF extraction, cleaning, chunking
- [x] Phase 4: Embeddings, pgvector, semantic retrieval
- [ ] Phase 5: RAG answer generation, citation validation, question history
- [ ] Phase 6: Cross-encoder reranking, evaluation on about 50 questions, BERT-based encoder comparison
- [ ] Phase 7: Feedback endpoint, tests, Docker, deployment