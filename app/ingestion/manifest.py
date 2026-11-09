"""Small SQLite manifest for idempotent ingestion."""

import sqlite3
from pathlib import Path

from app.ingestion.schemas import IngestRecord


class Manifest:
    """Stores content hashes and outcomes without storing PDF binary data."""

    def __init__(self, database_path: Path):
        self.database_path = database_path
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    content_hash TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    path TEXT NOT NULL,
                    source_url TEXT,
                    page_count INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL,
                    rejection_reason TEXT,
                    classification_json TEXT,
                    metadata_json TEXT,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    def has_hash(self, content_hash: str) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT status, rejection_reason FROM documents WHERE content_hash = ?",
                (content_hash,),
            ).fetchone()
        if row is None or row["status"] == "accepted":
            return row is not None
        if row["status"] != "rejected":
            return False
        return row["rejection_reason"] not in {
            "no_extractable_text",
        } and not str(row["rejection_reason"] or "").startswith("pdf_extraction_error:")

    def upsert(self, record: IngestRecord) -> None:
        classification = (
            record.classification.model_dump_json() if record.classification else None
        )
        metadata = record.metadata.model_dump_json() if record.metadata else None
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO documents (
                    content_hash, filename, path, source_url, page_count, status,
                    rejection_reason, classification_json, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(content_hash) DO UPDATE SET
                    filename = excluded.filename,
                    path = excluded.path,
                    source_url = excluded.source_url,
                    page_count = excluded.page_count,
                    status = excluded.status,
                    rejection_reason = excluded.rejection_reason,
                    classification_json = excluded.classification_json,
                    metadata_json = excluded.metadata_json,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (
                    record.content_hash,
                    record.filename,
                    record.path,
                    record.source_url,
                    record.page_count,
                    record.status,
                    record.rejection_reason,
                    classification,
                    metadata,
                ),
            )

    def count_by_status(self) -> dict[str, int]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT status, COUNT(*) AS count FROM documents GROUP BY status"
            ).fetchall()
        return {row["status"]: row["count"] for row in rows}

