"""Typed progress events streamed to the browser as Server-Sent Events.

Each event is a JSON object with a ``type`` discriminator; the frontend
(``static/js/app.js``) switches on it to update the live agent timeline.
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict

AgentName = Literal["planner", "researcher", "fact_checker", "conflict_detector", "report_writer"]
AgentStatus = Literal["running", "progress", "done"]


class AgentEvent(TypedDict):
    type: Literal["agent"]
    agent: AgentName
    status: AgentStatus
    detail: str


class PlanEvent(TypedDict):
    type: Literal["plan"]
    sub_questions: list[dict[str, Any]]


class ClaimsEvent(TypedDict):
    type: Literal["claims"]
    claims: list[dict[str, Any]]


class CachedEvent(TypedDict):
    type: Literal["cached"]
    detail: str


class ReportEvent(TypedDict):
    type: Literal["report"]
    report: dict[str, Any]


class ErrorEvent(TypedDict):
    type: Literal["error"]
    message: str


Event = AgentEvent | PlanEvent | ClaimsEvent | CachedEvent | ReportEvent | ErrorEvent


def agent_event(agent: AgentName, status: AgentStatus, detail: str) -> AgentEvent:
    return {"type": "agent", "agent": agent, "status": status, "detail": detail}


def error_event(message: str) -> ErrorEvent:
    return {"type": "error", "message": message}
