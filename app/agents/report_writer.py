"""Report-writer agent: turns verified claims into an evidence-backed report."""

from __future__ import annotations

from app.llm import LLM
from app.schemas import Claim, Conflict, ReportDraft, ReportParagraph, ReportSection, Source

SYSTEM = (
    "You are the Report Writer in a multi-agent fact-checking team. Write a balanced, "
    "evidence-backed research report for a general audience. Rules:\n"
    "- Every paragraph must cite the source IDs (e.g. S2) that back it.\n"
    "- Lead with well-supported findings; clearly label disputed or unverified points.\n"
    "- Present conflicting evidence fairly and explain why sources disagree.\n"
    "- Never state refuted claims as fact. Never invent sources, numbers or quotes.\n"
    "- Plain language, short paragraphs, 2-4 sections."
)


class ReportWriterAgent:
    name = "report_writer"

    def __init__(self, llm: LLM) -> None:
        self._llm = llm

    async def run(
        self, question: str, claims: list[Claim], conflicts: list[Conflict], sources: list[Source]
    ) -> ReportDraft:
        claim_lines = "\n".join(
            f"{c.id} [{c.verdict.value}, confidence {c.confidence:.2f}] {c.statement} "
            f"(supporting: {', '.join(c.supporting_source_ids) or 'none'}; "
            f"contradicting: {', '.join(c.contradicting_source_ids) or 'none'})"
            for c in claims
        )
        conflict_lines = (
            "\n".join(
                f"{k.id} ({k.reason.value}) {k.topic}: {k.explanation} [claims "
                f"{', '.join(k.claim_ids)}]"
                for k in conflicts
            )
            or "None detected."
        )
        source_lines = "\n".join(
            f"{s.id}: {s.title} ({s.domain}, credibility {s.tier.value})" for s in sources
        )
        prompt = (
            f"Research question: {question}\n\nVerified claims:\n{claim_lines}\n\n"
            f"Conflicts:\n{conflict_lines}\n\nSources:\n{source_lines}"
        )
        draft = await self._llm.generate_json(prompt, ReportDraft, system=SYSTEM, tier="smart")
        return _validate_citations(draft, {s.id for s in sources})


def _validate_citations(draft: ReportDraft, known_ids: set[str]) -> ReportDraft:
    """Drop citations to sources that do not exist, and empty paragraphs/sections."""
    sections: list[ReportSection] = []
    for section in draft.sections:
        paragraphs = [
            ReportParagraph(
                text=p.text.strip(),
                citations=[c for c in dict.fromkeys(p.citations) if c in known_ids],
            )
            for p in section.paragraphs
            if p.text.strip()
        ]
        if paragraphs:
            sections.append(ReportSection(heading=section.heading.strip(), paragraphs=paragraphs))
    return draft.model_copy(
        update={
            "sections": sections,
            "key_findings": [f.strip() for f in draft.key_findings if f.strip()][:8],
            "limitations": [lim.strip() for lim in draft.limitations if lim.strip()][:6],
        }
    )
