"""Pydantic models shared by the agents, the API and the frontend."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, field_validator

from app.security import sanitize_question

# --------------------------------------------------------------------------- #
# API input
# --------------------------------------------------------------------------- #


class Depth(StrEnum):
    """How thoroughly to research: quick uses fewer sub-questions and claims (fewer calls)."""

    QUICK = "quick"
    STANDARD = "standard"


class ResearchRequest(BaseModel):
    """A research question submitted by a user."""

    question: str = Field(..., min_length=10, max_length=500)
    depth: Depth = Depth.STANDARD

    @field_validator("question")
    @classmethod
    def _clean(cls, value: str) -> str:
        cleaned = sanitize_question(value)
        if len(cleaned) < 10:
            raise ValueError("Question must contain at least 10 meaningful characters.")
        return cleaned


# --------------------------------------------------------------------------- #
# Evidence
# --------------------------------------------------------------------------- #


class CredibilityTier(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Source(BaseModel):
    id: str
    url: str
    title: str
    domain: str
    credibility: float = Field(ge=0.0, le=1.0)
    tier: CredibilityTier
    snippet: str = ""


class Evidence(BaseModel):
    """Raw output of a search backend: a text digest plus the sources behind it."""

    text: str
    sources: list[Source]
    backend: str


# --------------------------------------------------------------------------- #
# LLM structured outputs (kept small and flat so Gemini can fill them reliably)
# --------------------------------------------------------------------------- #


class PlannedQuestion(BaseModel):
    question: str = Field(description="A focused sub-question that can be researched on its own.")
    search_query: str = Field(description="A short keyword search query (2-4 words).")


class PlanOutput(BaseModel):
    sub_questions: list[PlannedQuestion]


class ExtractedClaim(BaseModel):
    statement: str = Field(description="One specific, checkable factual claim.")
    source_ids: list[str] = Field(description="IDs of the sources that state this claim, e.g. S3.")


class ExtractionOutput(BaseModel):
    claims: list[ExtractedClaim]


class Verdict(StrEnum):
    SUPPORTED = "supported"
    DISPUTED = "disputed"
    REFUTED = "refuted"
    UNVERIFIED = "unverified"


class ClaimJudgement(BaseModel):
    claim_id: str
    verdict: Verdict
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str
    supporting_source_ids: list[str]
    contradicting_source_ids: list[str]


class FactCheckOutput(BaseModel):
    judgements: list[ClaimJudgement]


class ConflictReason(StrEnum):
    METHODOLOGY = "methodology"
    TIMEFRAME = "timeframe"
    DEFINITION = "definition"
    SOURCE_BIAS = "source_bias"
    OUTDATED = "outdated"
    OTHER = "other"


class DetectedConflict(BaseModel):
    topic: str
    claim_ids: list[str]
    explanation: str
    reason: ConflictReason


class ConflictOutput(BaseModel):
    conflicts: list[DetectedConflict]


class ReportParagraph(BaseModel):
    text: str
    citations: list[str] = Field(description="Source IDs that back this paragraph, e.g. S1.")


class ReportSection(BaseModel):
    heading: str
    paragraphs: list[ReportParagraph]


class ReportDraft(BaseModel):
    title: str
    summary: str
    key_findings: list[str]
    sections: list[ReportSection]
    limitations: list[str]


# --------------------------------------------------------------------------- #
# Final report
# --------------------------------------------------------------------------- #


class Claim(BaseModel):
    id: str
    statement: str
    sub_question: int
    source_ids: list[str]
    verdict: Verdict = Verdict.UNVERIFIED
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    rationale: str = ""
    supporting_source_ids: list[str] = Field(default_factory=list)
    contradicting_source_ids: list[str] = Field(default_factory=list)


class Conflict(BaseModel):
    id: str
    topic: str
    claim_ids: list[str]
    explanation: str
    reason: ConflictReason


class Report(BaseModel):
    id: str
    question: str
    depth: Depth = Depth.STANDARD
    created_at: str
    search_backend: str
    title: str
    summary: str
    key_findings: list[str]
    sections: list[ReportSection]
    limitations: list[str]
    sub_questions: list[PlannedQuestion]
    claims: list[Claim]
    conflicts: list[Conflict]
    sources: list[Source]
    duration_s: float
