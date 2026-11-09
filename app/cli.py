"""Phase 1 command-line entry point."""

import json
import logging
from pathlib import Path
from typing import Annotated

import typer

from app.config import get_settings
from app.evaluation.baseline import run_baseline
from app.evaluation.generate_dataset import generate_dataset
from app.indexing.service import AcademicIndexingService
from app.ingestion.crawler import AcademicCrawler
from app.ingestion.downloader import PdfDownloader
from app.ingestion.service import AcademicIngestionService
from app.logging_config import configure_logging
from app.models.policy import ModelPolicy
from app.rag.service import AcademicRAGService
from app.retrieval.schemas import QueryRequest

cli = typer.Typer(help="Academic-only PDF RAG commands.")
logger = logging.getLogger(__name__)


@cli.command()
def show_config() -> None:
    """Show safe configuration and the model-routing policy."""
    settings = get_settings()
    configure_logging(settings.log_level)
    policy = ModelPolicy.from_settings(settings)
    typer.echo(json.dumps({
        "environment": settings.app_env,
        "academic_only": settings.academic_only,
        "openai_key_configured": settings.openai_api_key is not None,
        "openai_pdf_fallback": {
            "enabled": settings.openai_api_key is not None,
            "model": settings.openai_pdf_model,
            "detail": settings.openai_pdf_detail,
            "max_mb": settings.openai_pdf_max_mb,
        },
        "ollama_base_url": settings.ollama_base_url,
        "models": policy.__dict__,
        "directories": {
            "raw_pdfs": str(settings.raw_pdf_dir),
            "parsed": str(settings.parsed_dir),
            "index": str(settings.index_dir),
        },
    }, indent=2))


@cli.command()
def health() -> None:
    """Run local foundation checks without making provider calls."""
    settings = get_settings()
    configure_logging(settings.log_level)
    typer.echo("Academic RAG foundation: healthy")
    typer.echo(f"Academic-only mode: {settings.academic_only}")
    typer.echo(f"OpenAI key configured: {settings.openai_api_key is not None}")
    typer.echo(f"Ollama endpoint: {settings.ollama_base_url}")


@cli.command()
def ingest(path: Annotated[Path, typer.Argument(exists=True, readable=True)]) -> None:
    """Ingest one academic PDF or a directory of PDFs using local Ollama models."""
    settings = get_settings()
    configure_logging(settings.log_level)
    service = AcademicIngestionService(settings)
    results = (
        service.ingest_directory(path) if path.is_dir()
        else [service.ingest_file(path)]
    )
    typer.echo(json.dumps({
        "processed": len(results),
        "statuses": service.manifest.count_by_status(),
        "results": [result.model_dump() for result in results],
    }, indent=2))


@cli.command()
def crawl(
    start_url: str = typer.Argument(..., help="Absolute HTTP(S) URL to crawl."),
    pdf_limit: int | None = typer.Option(None, min=1, help="Maximum PDF URLs to discover."),
) -> None:
    """Discover PDF URLs from HTML pages only; do not download or analyze PDFs."""
    settings = get_settings()
    configure_logging(settings.log_level)
    crawler = AcademicCrawler(settings)
    report = crawler.discover(start_url, pdf_limit)
    crawler.save_discovery(report)
    typer.echo(json.dumps(report, indent=2, ensure_ascii=False))


@cli.command()
def download(
    limit: int | None = typer.Option(None, min=1, help="Maximum discovered PDFs to download."),
) -> None:
    """Download discovered PDF URLs; do not extract metadata or call Ollama."""
    settings = get_settings()
    configure_logging(settings.log_level)
    discovery = AcademicCrawler(settings).load_discovery()
    report = PdfDownloader(settings).download(list(discovery.get("pdf_urls", [])),
                                              limit or settings.scrape_process_limit)
    typer.echo(json.dumps(report, indent=2, ensure_ascii=False))


@cli.command(name="process")
def process_downloads(
    limit: int | None = typer.Option(None, min=1, help="Maximum downloaded PDFs to analyze."),
) -> None:
    """Extract, classify, and metadata-process downloaded PDFs using local Ollama."""
    settings = get_settings()
    configure_logging(settings.log_level)
    manifest_path = settings.index_dir / "downloaded_pdfs.json"
    if not manifest_path.exists():
        raise typer.BadParameter(f"Download manifest not found: {manifest_path}; run download first")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    documents = [item for item in manifest.get("documents", []) if item.get("status") == "downloaded"]
    documents = documents[:limit or settings.scrape_process_limit]
    service = AcademicIngestionService(settings)
    results = []
    for item in documents:
        logger.info("Metadata processing started: %s", item["url"])
        result = service.ingest_file(Path(str(item["path"])), source_url=str(item["url"]))
        results.append(result)
        logger.info("Metadata processing completed: status=%s file=%s", result.status, result.filename)
    typer.echo(json.dumps({
        "processed": len(results),
        "statuses": service.manifest.count_by_status(),
        "results": [result.model_dump() for result in results],
    }, indent=2, ensure_ascii=False))


@cli.command(name="index")
def build_index(
    limit: int | None = typer.Option(None, min=1, help="Maximum accepted parsed documents to index."),
    reset: bool = typer.Option(False, help="Delete and rebuild the local vector collection."),
) -> None:
    """Chunk accepted documents, create embeddings, and build the Chroma index."""
    settings = get_settings()
    configure_logging(settings.log_level)
    if settings.openai_api_key is None:
        raise typer.BadParameter("OPENAI_API_KEY is required for indexing")
    report = AcademicIndexingService(settings).build(limit=limit, reset=reset)
    typer.echo(json.dumps(report, indent=2))


@cli.command(name="ask")
def ask(
    question: str = typer.Argument(..., help="Academic question to answer."),
    top_k: int = typer.Option(8, min=1, max=20, help="Number of evidence chunks."),
) -> None:
    """Retrieve academic evidence and generate a cited answer without reranking."""
    settings = get_settings()
    configure_logging(settings.log_level)
    if settings.openai_api_key is None:
        raise typer.BadParameter("OPENAI_API_KEY is required for querying")
    result = AcademicRAGService(settings).query(QueryRequest(question=question, top_k=top_k))
    typer.echo(json.dumps(result.model_dump(), indent=2, ensure_ascii=False))


@cli.command(name="evaluate-baseline")
def evaluate_baseline(
    dataset: Annotated[Path, typer.Option(exists=True)] = Path("data/evaluation/baseline_queries.jsonl"),
    output: Annotated[Path, typer.Option()] = Path("data/evaluation/baseline_results.json"),
    top_k: Annotated[int, typer.Option(min=1, max=20)] = 3,
) -> None:
    """Run the local baseline evaluation without query expansion or reranking."""
    settings = get_settings()
    configure_logging(settings.log_level)
    if settings.openai_api_key is None:
        raise typer.BadParameter("OPENAI_API_KEY is required for baseline evaluation")
    report = run_baseline(settings, dataset, output, top_k)
    typer.echo(json.dumps({key: value for key, value in report.items() if key != "results"}, indent=2))


@cli.command(name="generate-eval-dataset")
def generate_eval_dataset(
    output: Annotated[Path, typer.Option()] = Path("data/evaluation/generated_100_queries.jsonl"),
    model: Annotated[str, typer.Option()] = "gemma3:4b",
) -> None:
    """Generate 100 evaluation questions from the accepted corpus."""
    settings = get_settings()
    configure_logging(settings.log_level)
    report = generate_dataset(settings, output, model)
    typer.echo(json.dumps(report, indent=2))


def main() -> None:
    cli()


if __name__ == "__main__":
    main()

