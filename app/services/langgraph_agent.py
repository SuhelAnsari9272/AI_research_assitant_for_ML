
import json
import os
from typing import Any, Dict, List, Optional, TypedDict

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

# llama-3.3-70b-versatile is a strong, cheap, fast default on Groq for this
# kind of structured, evidence-grounded reasoning task. Override via env var.
DEFAULT_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
DEFAULT_TEMPERATURE = float(os.environ.get("GROQ_TEMPERATURE", "0.2"))


class ReasoningOutput(BaseModel):
    """Structured schema the LLM is constrained to produce.

    Field-for-field identical to `schemas.feature_profile.AIReasoning` so the
    client layer can convert between them with a plain `model_dump()`.
    """

    summary: str = Field(..., description="One or two sentence plain-English rationale for the decision.")
    evidence: List[str] = Field(default_factory=list, description="Concrete statistics/signals that justify it.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Calibrated confidence, 0-1.")
    alternatives_considered: List[str] = Field(default_factory=list)
    risks: List[str] = Field(default_factory=list)
    expected_impact: str = Field(..., description="What happens downstream in the pipeline if this is approved.")
    intervention_recommended: bool = Field(
        default=False, description="True if the AI itself thinks a human should double-check this."
    )


class AgentState(TypedDict, total=False):
    task_type: str
    subject: str
    context: Dict[str, Any]
    previous_reasoning: Optional[Dict[str, Any]]
    user_feedback: Optional[str]
    final_reasoning: ReasoningOutput


_SYSTEM_PROMPT = """You are the reasoning engine of a Human-in-the-Loop AutoML platform.
A data scientist reviews and approves every recommendation you make before anything
proceeds in the pipeline, so you must be transparent, concrete, and honest about
uncertainty. Never invent statistics - ground every claim in the machine-computed
context you are given. If the context is thin, say so and lower your confidence.

Task type: {task_type}
Subject: {subject}

Return your recommendation as the required structured schema."""

_DRAFT_HUMAN_TEMPLATE = """Machine-computed context:
{context}

Produce your reasoning now."""

_REVISE_SYSTEM_PROMPT = _SYSTEM_PROMPT + """

You previously produced this recommendation:
{previous}

The data scientist has requested changes. Incorporate their feedback faithfully:
if they've asked you to change a decision, actually change it and reflect that in
`summary` and `evidence` - do not just acknowledge the feedback without changing
the substance of the recommendation."""

_REVISE_HUMAN_TEMPLATE = """Original machine-computed context:
{context}

Data scientist's feedback:
{feedback}

Produce your revised reasoning now."""


def _get_llm() -> ChatGroq:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add it to your .env (see .env.example) to enable "
            "live AI reasoning; the app falls back to local heuristics without it."
        )
    return ChatGroq(model=DEFAULT_MODEL, api_key=api_key, temperature=DEFAULT_TEMPERATURE)


def _format_context(context: Dict[str, Any]) -> str:
    return json.dumps(context, indent=2, default=str)

def _draft_node(state: AgentState) -> AgentState:
    llm = _get_llm().with_structured_output(ReasoningOutput)
    prompt = ChatPromptTemplate.from_messages([("system", _SYSTEM_PROMPT), ("human", _DRAFT_HUMAN_TEMPLATE)])
    chain = prompt | llm
    result: ReasoningOutput = chain.invoke({
        "task_type": state["task_type"],
        "subject": state["subject"],
        "context": _format_context(state["context"]),
    })
    return {"final_reasoning": result}


def _revise_node(state: AgentState) -> AgentState:
    llm = _get_llm().with_structured_output(ReasoningOutput)
    prompt = ChatPromptTemplate.from_messages([("system", _REVISE_SYSTEM_PROMPT), ("human", _REVISE_HUMAN_TEMPLATE)])
    chain = prompt | llm
    result: ReasoningOutput = chain.invoke({
        "task_type": state["task_type"],
        "subject": state["subject"],
        "previous": json.dumps(state.get("previous_reasoning") or {}, indent=2, default=str),
        "context": _format_context(state["context"]),
        "feedback": state.get("user_feedback") or "",
    })
    return {"final_reasoning": result}


def _route_entry(state: AgentState) -> str:
    """Skip straight to `revise` when feedback + a prior reasoning exist."""
    if state.get("user_feedback") and state.get("previous_reasoning"):
        return "revise"
    return "draft"


def _build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("draft", _draft_node)
    graph.add_node("revise", _revise_node)
    graph.set_conditional_entry_point(_route_entry, {"draft": "draft", "revise": "revise"})
    graph.add_edge("draft", END)
    graph.add_edge("revise", END)
    return graph.compile()


_GRAPH = None

def get_graph():
    """Lazily build + cache the compiled graph (building it imports/creates
    the Groq client, which requires GROQ_API_KEY - keep this lazy so the app
    can boot and use local heuristics with no key configured at all)."""
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = _build_graph()
    return _GRAPH