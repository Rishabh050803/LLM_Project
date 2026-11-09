# Phase 2 Report

## Implemented

Phase 2 adds the first real ingestion pipeline for academic PDFs:

- page-preserving PDF extraction using `pypdf`
- SHA-256 content hashing for duplicate detection
- SQLite ingestion manifest at `data/index/manifest.sqlite3`
- configurable PDF size limits
- configurable academic source-domain allowlist
- combined local academic classification and metadata extraction with `qwen3:4b`
- direct OpenAI PDF-input fallback for scanned/image-only PDFs when an API key
  is configured
- parsed page text and metadata saved under `data/parsed/`
- rejection reasons for non-PDF, oversized, disallowed, duplicate, scanned,
  or non-academic documents
- CLI ingestion command for one file or a directory

## Usage

```bash
cd /home/rkp/LLM_Project/academic-rag
academic-rag ingest data/raw_pdfs
```

Normal text PDFs use Ollama for classification and metadata extraction. PDFs
with no locally extractable text use the configured OpenAI vision model as a
fallback. Direct-PDF fallback records metadata but does not create searchable
page text; OCR or vision-based text extraction is still required before
embedding those documents.

## Ollama connectivity

The agent environment could not connect to `127.0.0.1:11434` because local
socket access is restricted in this session. The implementation targets the
configured `OLLAMA_BASE_URL` and will surface a provider error if Ollama is not
available when run locally.

## Deliberate limitations

- scanned/image-only PDFs are rejected because OCR is not included yet
- same-domain HTML discovery, remote downloading, and local processing are
  separate commands and persisted through JSON manifests
- accepted PDFs are parsed and stored, but not embedded or searchable yet
- discovery is separate from processing; the initial batch is processed with
  one local Ollama worker

## Scraping controls

The crawler discovers up to 100 PDF URLs, excludes external subdomains by
default, and has no page-count limit. It visits HTML pages only, extracting
links from page anchors; PDF bytes are not used for discovery. The downloader
and processor are independent stages, with processing limited to 10 PDFs in
the initial batch. Local Ollama processing is serialized by default because the available
model is CPU-bound. Qwen3 thinking is disabled for this structured extraction
task, the prompt excerpt/output are bounded, and requests use a 120-second
timeout with two attempts. If analysis remains unavailable, the document is
recorded as `pending_ollama` and can be retried.

