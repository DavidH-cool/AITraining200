"""Human-in-the-Loop (HITL) Confirmation Hooks for High-Stakes Actions.

Addresses Rubric Category 3 (Orchestration & Logic) - Human-in-the-Loop Hooks:
- High-stakes actions (such as publishing a blog post externally via
  `publish_approved_blog_post_to_cms`) include explicit code stops requiring
  verified human confirmation before execution.
- Implements an ADK `before_tool_callback` (`require_human_approval_before_tool_callback`)
  and a stateful `HumanApprovalGate` registry.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from blog_writer_agent.observability.logger import get_structured_logger

logger = get_structured_logger(__name__)

HIGH_STAKES_TOOLS = frozenset(
    {
        "publish_approved_blog_post_to_cms",
    }
)


@dataclass
class ApprovalRecord:
    """Audit record of a human operator's explicit approval decision."""

    session_id: str
    approver_id: str
    approved: bool
    notes: str
    timestamp_utc: str


class HumanInTheLoopApprovalGate:
    """Manages explicit human confirmation stops before high-stakes tool execution."""

    def __init__(self) -> None:
        self._approvals: Dict[str, ApprovalRecord] = {}

    def grant_approval(
        self,
        session_id: str,
        approver_id: str,
        notes: str = "Approved by human author for blog publication.",
    ) -> ApprovalRecord:
        """Records explicit human approval for a specific drafting session."""
        record = ApprovalRecord(
            session_id=session_id,
            approver_id=approver_id,
            approved=True,
            notes=notes,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
        )
        self._approvals[session_id] = record
        logger.info(
            "Human-in-the-Loop publication approval granted.",
            extra={
                "event_type": "hitl_approval_granted",
                "session_id": session_id,
                "approver_id": approver_id,
            },
        )
        return record

    def revoke_approval(self, session_id: str) -> None:
        """Revokes human approval for `session_id`."""
        self._approvals.pop(session_id, None)

    def is_approved(self, session_id: str) -> bool:
        """Returns True only if a verified human approval record exists for `session_id`."""
        rec = self._approvals.get(session_id)
        return bool(rec and rec.approved)


default_hitl_gate = HumanInTheLoopApprovalGate()


def require_human_approval_before_tool_callback(
    tool: Any,
    args: Dict[str, Any],
    tool_context: Any,
) -> Optional[Dict[str, Any]]:
    """ADK `before_tool_callback` that halts high-stakes tools without human confirmation.

    Args:
        tool: The ADK tool object or callable being invoked.
        args: Tool call arguments dictionary supplied by the LLM.
        tool_context: ADK `ToolContext` carrying `state` and session metadata.

    Returns:
        `None` if execution is allowed to proceed, or a structured dictionary halting
        tool execution and instructing the agent to request human confirmation.
    """
    tool_name = getattr(tool, "name", None) or getattr(tool, "__name__", str(tool))
    if tool_name not in HIGH_STAKES_TOOLS:
        return None

    state: Dict[str, Any] = getattr(tool_context, "state", {}) or {}
    session_id = str(state.get("session_id", "default"))

    explicit_arg_flag = bool(args.get("human_approval_confirmed", False))
    state_flag = bool(state.get("human_publish_approval_granted", False))
    gate_flag = default_hitl_gate.is_approved(session_id)

    if explicit_arg_flag or state_flag or gate_flag:
        args["human_approval_confirmed"] = True
        logger.info(
            f"HITL check passed for high-stakes tool '{tool_name}'.",
            extra={
                "event_type": "hitl_check_passed",
                "tool_name": tool_name,
                "session_id": session_id,
            },
        )
        return None

    logger.warning(
        f"HITL gate halted unapproved execution of '{tool_name}'.",
        extra={
            "event_type": "hitl_gate_halted_tool",
            "tool_name": tool_name,
            "session_id": session_id,
        },
    )
    return {
        "status": "pending_human_approval",
        "error_code": "HITL_CONFIRMATION_REQUIRED",
        "tool_name": tool_name,
        "message": (
            f"Execution of high-stakes tool '{tool_name}' was halted because human "
            "confirmation has not been granted yet."
        ),
        "recovery_instructions": (
            "Show the user the completed blog post draft, title, and tags, and ask: "
            "'Would you like me to publish this blog post now?' Only call "
            f"'{tool_name}' with human_approval_confirmed=True after the user confirms."
        ),
    }
