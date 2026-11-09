# Phase 1 Report

## Scope completed

The new project was created at `academic-rag/`, beside the previous `pro/`
project. The previous project was not edited.

Implemented:

1. A standalone Python package with Python 3.12 support.
2. Typed settings loaded from `.env` using Pydantic Settings.
3. A safe `.env.example` with one OpenAI key and configurable Ollama models.
4. A model policy that assigns low-cost preprocessing to Ollama and paid
   embeddings/answer generation to OpenAI.
5. Separate raw PDF, parsed-data, and index directories.
6. FastAPI application factory and a configuration-safe `/health` endpoint.
7. CLI commands for local health checks and safe configuration inspection.
8. Git exclusions for secrets, PDFs, parsed files, indexes, and databases.

## Validation

The scaffold was created without making external model requests. Ollama was
not reachable from the current execution environment, so provider connectivity
is intentionally left to the local run step.

## Not included yet

- PDF downloading or crawling
- academic document classification
- PDF parsing/OCR
- metadata extraction
- embeddings or vector storage
- retrieval and answer generation
- evaluation dataset
- UI

Those belong to Phase 2 onward.

## Recommended next phase

Phase 2 should implement curated academic PDF ingestion first, using a source
allowlist, checksum-based deduplication, page-preserving extraction, and local
Ollama classification before any document is indexed.

