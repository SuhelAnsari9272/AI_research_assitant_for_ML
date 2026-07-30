from __future__ import annotations

import os
from typing import Any, Dict, Optional

from schemas.feature_profile import AIReasoning

from dotenv import load_dotenv

load_dotenv()

class LangGraphUnavailable(RuntimeError):
    "Raised whenever the Groq/LangGraph agent can't be used right now;"

class LangGraphClient:
    """Thin, resilient wrapper around the compiled LangGraph agent graph."""

    def __init__(self) -> None:
        self._graph = None

    @property
    def is_configured(self) -> bool:
        # print(os.environ.get("GROQ_API_KEY"))
        return bool(os.environ.get("GROQ_API_KEY"))

    def _get_graph(self):
        if not self.is_configured:
            raise LangGraphUnavailable("GROQ_API_KEY is not set - see .env.example.")
        if self._graph is None:
            from services.langgraph_agent import get_graph  # deferred: avoids requiring Groq creds just to import
            self._graph = get_graph()
        return self._graph

    def generate_reasoning(self, task_type: str, subject: str, context: Dict[str, Any]) -> AIReasoning:
        """First-time generation of a recommendation for `subject`."""
        graph = self._get_graph()
        try:
            result = graph.invoke({"task_type": task_type, "subject": subject, "context": context})
            return AIReasoning(**result["final_reasoning"].model_dump())
        except LangGraphUnavailable:
            raise
        except Exception as exc:  # noqa: BLE001 - any LLM/graph failure -> typed fallback signal
            raise LangGraphUnavailable(f"Agent call failed: {exc}") from exc

    def refine_reasoning(
        self,
        task_type: str,
        subject: str,
        context: Dict[str, Any],
        feedback: str,
    ) -> AIReasoning:
        """Re-invoke the agent with the data scientist's feedback on `previous`."""
        graph = self._get_graph()
        try:
            result = graph.invoke({
                "task_type": task_type,
                "subject": subject,
                "context": context,
                # "previous_reasoning": previous.model_dump(),
                "user_feedback": feedback,
            })
            return AIReasoning(**result["final_reasoning"].model_dump())
        except LangGraphUnavailable:
            raise
        except Exception as exc:  # noqa: BLE001
            raise LangGraphUnavailable(f"Agent revision call failed: {exc}") from exc

_client: Optional[LangGraphClient] = None

def get_client() -> LangGraphClient:
    """Process-wide singleton accessor."""
    global _client
    if _client is None:
        _client = LangGraphClient()
    return _client

def generate_with_fallback(task_type: str, subject: str, context: Dict[str, Any], fallback: AIReasoning) -> AIReasoning:
    """Convenience for call sites that already have a deterministic local
    `AIReasoning` on hand (pages 5-8): try the live agent first, and use the
    local one unchanged if the agent isn't configured or the call fails."""
    try:
        return get_client().generate_reasoning(task_type, subject, context)
    except LangGraphUnavailable:
        return fallback
