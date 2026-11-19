"""The five specialised agents that make up the research pipeline."""

from app.agents.conflict_detector import ConflictDetectorAgent
from app.agents.fact_checker import FactCheckerAgent, score_confidence
from app.agents.planner import PlannerAgent
from app.agents.report_writer import ReportWriterAgent
from app.agents.researcher import ResearcherAgent

__all__ = [
    "ConflictDetectorAgent",
    "FactCheckerAgent",
    "PlannerAgent",
    "ReportWriterAgent",
    "ResearcherAgent",
    "score_confidence",
]
