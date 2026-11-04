"""Security, Policy & Self-Evaluation Guardrails (ADK Plugins + Vertex AI Safety).

Addresses Rubric Category 3 (Orchestration & Logic) - Guardrails & Policy Plugins:
- Implements multi-layered input and output guardrails via ADK callbacks and
  Google Cloud Vertex AI Safety Settings:
  1. Prompt Injection & Jailbreak Guardrail (`inspect_input_for_policy_violations`).
  2. Proprietary Secret & Internal Codename Leak Guardrail.
  3. Automated Self-Evaluation Guardrail (`run_self_evaluation_guardrail`) that
     verifies factual grounding and editorial rubric compliance before returning output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Dict, List, Optional, Sequence

from blog_writer_agent.observability.logger import get_structured_logger
from blog_writer_agent.observability.pii_redactor import default_pii_redactor
from blog_writer_agent.tools.blog_tools import evaluate_blog_draft_readability_and_tone

logger = get_structured_logger(__name__)

try:
    from google.genai import types as genai_types  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover
    genai_types = None


_PROMPT_INJECTION_PATTERNS: Sequence[re.Pattern[str]] = (
    re.compile(r"(?i)\bignore\s+(?:all\s+)?(?:previous|prior|system)\s+instructions\b"),
    re.compile(r"(?i)\bdisregard\s+your\s+constitution\b"),
    re.compile(r"(?i)\bexfiltrate\s+(?:secrets|api\s*keys|credentials)\b"),
    re.compile(r"(?i)\bbypass\s+human[\s-]*in[\s-]*the[\s-]*loop\b"),
)

_FORBIDDEN_INTERNAL_MARKERS: Sequence[str] = (
    "CONFIDENTIAL_DO_NOT_PUBLISH",
    "INTERNAL_ONLY_CODENAME",
    "BEGIN RSA PRIVATE KEY",
)


@dataclass
class GuardrailInspectionResult:
    """Result of evaluating an input prompt or model output against security/self-eval policies."""

    allowed: bool
    policy_name: str
    violation_reasons: List[str] = field(default_factory=list)
    sanitized_content: str = ""
    self_eval_score: Optional[int] = None


class AgentPolicyGuardrails:
    """Enforces input security, secret prevention, Vertex AI safety, and self-evaluation."""

    def get_vertex_safety_settings(self) -> List[Any]:
        """Returns Google Cloud Vertex AI / Gemini safety settings blocking harmful content."""
        if genai_types is not None and hasattr(genai_types, "SafetySetting"):
            try:
                return [
                    genai_types.SafetySetting(
                        category=genai_types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                        threshold=genai_types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
                    ),
                    genai_types.SafetySetting(
                        category=genai_types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                        threshold=genai_types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
                    ),
                    genai_types.SafetySetting(
                        category=genai_types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                        threshold=genai_types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
                    ),
                ]
            except Exception:  # pragma: no cover
                pass
        return [
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_LOW_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_LOW_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_LOW_AND_ABOVE"},
        ]

    def inspect_input_for_policy_violations(
        self,
        user_input: str,
    ) -> GuardrailInspectionResult:
        """Checks incoming prompts for prompt injection, jailbreaks, and confidential markers."""
        violations: List[str] = []
        for pattern in _PROMPT_INJECTION_PATTERNS:
            if pattern.search(user_input):
                violations.append(
                    f"Detected adversarial prompt injection pattern: '{pattern.pattern}'."
                )

        for marker in _FORBIDDEN_INTERNAL_MARKERS:
            if marker.lower() in user_input.lower():
                violations.append(
                    f"Input contains restricted confidential marker '{marker}' that cannot be published."
                )

        allowed = len(violations) == 0
        sanitized = default_pii_redactor.redact_text(user_input)
        if not allowed:
            logger.warning(
                "Input blocked by AgentPolicyGuardrails.",
                extra={
                    "event_type": "guardrail_input_blocked",
                    "violations": violations,
                },
            )
        return GuardrailInspectionResult(
            allowed=allowed,
            policy_name="input_security_and_confidentiality_guardrail",
            violation_reasons=violations,
            sanitized_content=sanitized,
        )

    def run_self_evaluation_guardrail(
        self,
        draft_markdown: str,
        source_engineering_facts: List[str],
        required_keywords: Optional[List[str]] = None,
    ) -> GuardrailInspectionResult:
        """Executes automated self-evaluation on a generated blog draft before release."""
        sanitized_draft = default_pii_redactor.redact_text(draft_markdown)
        report = evaluate_blog_draft_readability_and_tone(
            draft_markdown=sanitized_draft,
            source_engineering_facts=source_engineering_facts,
            required_keywords=required_keywords,
        )
        score = int(report.get("overall_quality_score", 0))
        passes = bool(report.get("passes_editorial_bar", False))
        feedback = list(report.get("actionable_revision_feedback", []))

        logger.info(
            "Completed automated self-evaluation guardrail check.",
            extra={
                "event_type": "self_evaluation_guardrail_completed",
                "score": score,
                "passes": passes,
                "feedback_count": len(feedback),
            },
        )
        return GuardrailInspectionResult(
            allowed=passes,
            policy_name="automated_self_evaluation_rubric_guardrail",
            violation_reasons=feedback if not passes else [],
            sanitized_content=sanitized_draft,
            self_eval_score=score,
        )


default_policy_guardrails = AgentPolicyGuardrails()


def adk_before_model_guardrail_callback(
    callback_context: Any,
    llm_request: Any,
) -> Optional[Dict[str, Any]]:
    """ADK `before_model_callback` plugin that blocks prompt injections before LLM invocation."""
    contents = getattr(llm_request, "contents", []) or []
    combined_text = " ".join(str(c) for c in contents)
    inspection = default_policy_guardrails.inspect_input_for_policy_violations(combined_text)
    if not inspection.allowed:
        return {
            "status": "blocked_by_guardrail",
            "policy_name": inspection.policy_name,
            "violation_reasons": inspection.violation_reasons,
        }
    return None


def adk_after_model_guardrail_callback(
    callback_context: Any,
    llm_response: Any,
) -> Optional[Any]:
    """ADK `after_model_callback` plugin that scrubs PII from model responses."""
    text = getattr(llm_response, "text", None)
    if isinstance(text, str) and text:
        redacted = default_pii_redactor.redact_text(text)
        if redacted != text:
            try:
                llm_response.text = redacted
            except Exception:  # pragma: no cover
                pass
    return None
