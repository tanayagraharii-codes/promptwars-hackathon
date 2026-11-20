"""Report-writer agent: turns verified claims into an evidence-backed report."""

from __future__ import annotations

from app.llm import LLM
from app.schemas import Claim, Conflict, ReportDraft, ReportParagraph, ReportSection, Source

from .base import clean_lines, known_ids

SYSTEM = (
    "You are the Report Writer in a multi-agent fact-checking team. Write a balanced, "
    "evidence-backed research report for a general audience. Rules:\n"
    "- Every paragraph must cite the source IDs (e.g. S2) that back it.\n"
    "- Lead with well-supported findings; clearly label disputed or unverified points.\n"
    "- Present conflicting evidence fairly and explain why sources disagree.\n"
    "- Never state refuted claims as fact. Never invent sources, numbers or quotes.\n"
    "- Plain language, short paragraphs, 2-4 sections."
)
MAX_KEY_FINDINGS = 8
MAX_LIMITATIONS = 6


class ReportWriterAgent:
    """Writes the final report; every citation is validated against real sources."""

    name = "report_writer"

    def __init__(self, llm: LLM) -> None:
        self._llm = llm

    async def run(
        self, question: str, claims: list[Claim], conflicts: list[Conflict], sources: list[Source]
    ) -> ReportDraft:
        """Draft the report from verified claims, then strip any invalid citations."""
        prompt = (
            f"Research question: {question}\n\n"
            f"Verified claims:\n{_claim_lines(claims)}\n\n"
            f"Conflicts:\n{_conflict_lines(conflicts)}\n\n"
            f"Sources:\n{_source_lines(sources)}"
        )
        draft = await self._llm.generate_json(prompt, ReportDraft, system=SYSTEM, tier="smart")
        return validate_citations(draft, {s.id for s in sources})


def validate_citations(draft: ReportDraft, valid_ids: set[str]) -> ReportDraft:
    """Drop citations to sources that do not exist, and empty paragraphs or sections."""
    sections: list[ReportSection] = []
    for section in draft.sections:
        paragraphs = [
            ReportParagraph(text=p.text.strip(), citations=known_ids(p.citations, valid_ids))
            for p in section.paragraphs
            if p.text.strip()
        ]
        if paragraphs:
            sections.append(ReportSection(heading=section.heading.strip(), paragraphs=paragraphs))
    return draft.model_copy(
        update={
            "sections": sections,
            "key_findings": clean_lines(draft.key_findings, MAX_KEY_FINDINGS),
            "limitations": clean_lines(draft.limitations, MAX_LIMITATIONS),
        }
    )


def _claim_lines(claims: list[Claim]) -> str:
    return "\n".join(
        f"{c.id} [{c.verdict.value}, confidence {c.confidence:.2f}] {c.statement} "
        f"(supporting: {', '.join(c.supporting_source_ids) or 'none'}; "
        f"contradicting: {', '.join(c.contradicting_source_ids) or 'none'})"
        for c in claims
    )


def _conflict_lines(conflicts: list[Conflict]) -> str:
    lines = (
        f"{k.id} ({k.reason.value}) {k.topic}: {k.explanation} [claims {', '.join(k.claim_ids)}]"
        for k in conflicts
    )
    return "\n".join(lines) or "None detected."


def _source_lines(sources: list[Source]) -> str:
    return "\n".join(f"{s.id}: {s.title} ({s.domain}, credibility {s.tier.value})" for s in sources)
