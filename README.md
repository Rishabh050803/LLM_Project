# Academic RAG

An academics-only PDF RAG project, separate from the earlier institute-wide
[`pro`](../pro) project.

## Phase 1 status

This phase establishes the application foundation only:

- typed environment configuration
- safe `.env.example`
- explicit local/Ollama versus OpenAI model policy
- data directory layout
- FastAPI health endpoint
- CLI health and configuration commands
- Python package metadata and development dependencies

No PDFs are downloaded, no OpenAI requests are made, and no Ollama requests
are made by the Phase 1 checks.

## Model policy

- Ollama handles metadata extraction, academic-scope classification, and query normalization.
- OpenAI handles embeddings and final answer generation.
- One `OPENAI_API_KEY` is expected; key rotation is intentionally not implemented.

## Run locally

```bash
cp .env.example .env
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
academic-rag health
academic-rag show-config
uvicorn app.main:app --reload
```

The initial endpoint is `GET /health`.

## Phase 2 ingestion

Phase 2 adds local PDF ingestion. Put curated academic PDFs under
`data/raw_pdfs/`, then run:

```bash
academic-rag ingest data/raw_pdfs
```

Each PDF is hashed and extracted page by page. Text PDFs use local Ollama for
classification and metadata. If local extraction produces no usable text, the
PDF is sent directly to the configured OpenAI vision model (`gpt-4o-mini`) as a
fallback. Accepted documents are saved as JSON under `data/parsed/`; ingestion
state is tracked in `data/index/manifest.sqlite3`.

Direct-PDF fallback is metadata/classification support only: because scanned
documents have no local page text, they still need OCR or a later vision-text
extraction step before they can be embedded for retrieval. PDFs larger than the
OpenAI direct-input limit are recorded as `pending_openai`.

Scanned/image-only PDFs are rejected for now because OCR is planned for a later
phase.

The ingestion pipeline is split into three independent stages. Discovery has
no HTML page limit: it stops only when the PDF URL limit is reached or the
deduplicated HTML-link queue is exhausted.

```bash
academic-rag crawl https://www.mnit.ac.in/academics/ --pdf-limit 100
academic-rag download --limit 10
academic-rag process --limit 10

# Build embeddings and the local vector database
env -u OPENAI_API_KEY academic-rag index

# Retrieve academic evidence and generate a cited answer (no LLM reranking)
env -u OPENAI_API_KEY academic-rag ask "What is the formula used for CGPA?"
```

The crawler excludes external subdomains and prioritizes academic routes by
default. `crawl` extracts links only from visited HTML pages and never from PDF
bytes. `download` downloads only discovered PDFs. `process` is the only stage
that extracts text and calls Ollama. Ollama analysis uses one combined local-
model call with Qwen3 thinking disabled, bounded output, serialized processing,
retries, and a 120-second timeout; unavailable analysis remains pending for
retry.

