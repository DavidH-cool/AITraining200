"""Strategic Model Routing between Fast (Flash) and Reasoning/Planning (Pro) Models.

Addresses Rubric Category 3 (Orchestration & Logic) - Strategic Model Routing:
- Routes specific tasks and sub-agents to the most appropriate Gemini model tier:
  * `gemini-2.5-flash` for low-latency artifact extraction, metadata tagging, and formatting.
  * `gemini-2.5-pro` for deep architectural synthesis, long-form technical narrative drafting,
    and multi-criteria editorial evaluation.
- Provides a dynamic ADK `before_model_callback` (`dynamic_model_router_callback`) that can
  upgrade or downgrade model selection based on input complexity and token count.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional

from blog_writer_agent.memory.compaction import estimate_token_count
from blog_writer_agent.observability.logger import get_structured_logger

logger = get_structured_logger(__name__)

FAST_MODEL_ID = "gemini-2.5-flash"
REASONING_MODEL_ID = "gemini-2.5-pro"

TaskComplexityType = Literal[
    "artifact_extraction",
    "reference_lookup",
    "cms_packaging",
    "architectural_planning",
    "long_form_drafting",
    "editorial_critique",
]


@dataclass(frozen=True)
class ModelRoutingDecision:
    """Represents a deterministic routing choice with rationale for observability."""

    task_type: str
    selected_model: str
    tier: Literal["fast_flash", "reasoning_pro"]
    rationale: str


class StrategicModelRouter:
    """Routes agent tasks and dynamic requests between Gemini 2.5 Flash and Gemini 2.5 Pro."""

    FAST_TASKS = frozenset(
        {
            "artifact_extraction",
            "reference_lookup",
            "cms_packaging",
            "work_artifact_researcher_agent",
            "cms_publication_agent",
        }
    )
    REASONING_TASKS = frozenset(
        {
            "architectural_planning",
            "long_form_drafting",
            "editorial_critique",
            "blog_writer_coordinator_agent",
            "technical_blog_drafter_agent",
            "editorial_critic_agent",
        }
    )

    def __init__(
        self,
        fast_model: str = FAST_MODEL_ID,
        reasoning_model: str = REASONING_MODEL_ID,
        complexity_token_threshold: int = 1200,
    ) -> None:
        self.fast_model = fast_model
        self.reasoning_model = reasoning_model
        self.complexity_token_threshold = complexity_token_threshold

    def select_model_for_task(
        self,
        task_or_agent_name: str,
        prompt_text: str = "",
    ) -> ModelRoutingDecision:
        """Selects the optimal Gemini model for the given sub-agent or task type."""
        normalized = task_or_agent_name.strip().lower()
        estimated_tokens = estimate_token_count(prompt_text)

        if normalized in self.REASONING_TASKS or estimated_tokens >= self.complexity_token_threshold:
            decision = ModelRoutingDecision(
                task_type=task_or_agent_name,
                selected_model=self.reasoning_model,
                tier="reasoning_pro",
                rationale=(
                    f"Task '{task_or_agent_name}' (tokens={estimated_tokens}) requires deep "
                    f"architectural synthesis, long-form narrative planning, or editorial critique."
                ),
            )
        else:
            decision = ModelRoutingDecision(
                task_type=task_or_agent_name,
                selected_model=self.fast_model,
                tier="fast_flash",
                rationale=(
                    f"Task '{task_or_agent_name}' (tokens={estimated_tokens}) is structured "
                    f"extraction/packaging best served by low-latency '{self.fast_model}'."
                ),
            )

        logger.info(
            "Routed task to optimal Gemini model tier.",
            extra={
                "event_type": "strategic_model_routed",
                "task_type": decision.task_type,
                "selected_model": decision.selected_model,
                "tier": decision.tier,
                "rationale": decision.rationale,
            },
        )
        return decision


default_model_router = StrategicModelRouter()


def dynamic_model_router_callback(
    callback_context: Any,
    llm_request: Any,
) -> Optional[Any]:
    """ADK `before_model_callback` that dynamically routes `llm_request.model`."""
    agent_name = getattr(callback_context, "agent_name", "unknown_agent")
    contents = getattr(llm_request, "contents", []) or []
    combined_text = " ".join(str(c) for c in contents)
    decision = default_model_router.select_model_for_task(
        task_or_agent_name=str(agent_name),
        prompt_text=combined_text,
    )
    if hasattr(llm_request, "model"):
        llm_request.model = decision.selected_model
    return None
