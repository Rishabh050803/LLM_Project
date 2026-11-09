"""Schemas persisted for academic document ingestion."""

from pydantic import BaseModel, Field


class AcademicClassification(BaseModel):
    """Local-model decision about whether a PDF belongs in the academic corpus."""

    is_academic: bool
    confidence: float = Field(ge=0, le=1)
    category: str = "unknown"
    reason: str = ""


class AcademicMetadata(BaseModel):
    """Normalized metadata extracted from an academic PDF."""

    title: str = "Unknown"
    institution: str = "Unknown"
    department: str = "Unknown"
    program: str = "Unknown"
    document_type: str = "unknown"
    academic_year: str = "Unknown"
    semester: str = "Unknown"
    audience: str = "Unknown"
    course_code: str = "Unknown"
    keywords: list[str] = Field(default_factory=list)
    summary: str = ""


class AcademicAnalysis(BaseModel):
    """Combined classification and metadata response from one local model call."""

    classification: AcademicClassification
    metadata: AcademicMetadata


class PageText(BaseModel):
    """Text extracted from one PDF page, retaining page identity."""

    page_number: int = Field(ge=1)
    text: str


class IngestRecord(BaseModel):
    """Manifest record for one ingestion attempt."""

    content_hash: str
    filename: str
    path: str
    source_url: str | None = None
    page_count: int = 0
    status: str
    rejection_reason: str | None = None
    classification: AcademicClassification | None = None
    metadata: AcademicMetadata | None = None

