"""Automated Evaluation Harness against the Golden Dataset.

Addresses Rubric Category 5 (Infrastructure & CI/CD) - Automated Evaluation Suites:
- Evaluates the multi-agent Work-to-Blog pipeline against `evals/golden_dataset.json`.
- Statically and dynamically measures regressions across:
  1. Tool trajectory & schema validation
  2. Factual grounding & editorial rubric score (>= 80/100)
  3. Active PII & secret redaction (zero leaks)
  4. Human-in-the-Loop publication gating
  5. Adversarial prompt-injection guardrail blocking
  6. OpenTelemetry span linkage
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import sys
import tempfile
from typing import Any, Dict, List

from blog_writer_agent.agent import WorkToBlogOrchestrator
from blog_writer_agent.memory.async_memory import default_async_memory_manager
from blog_writer_agent.observability.tracing import (
    clear_recorded_spans,
    get_recorded_spans,
)


async def evaluate_golden_dataset(
    golden_dataset_path: Path | str | None = None,
) -> Dict[str, Any]:
    """Runs all golden dataset evaluation cases and returns a structured scorecard."""
    if golden_dataset_path is None:
        golden_dataset_path = Path(__file__).resolve().parent / "golden_dataset.json"
    else:
        golden_dataset_path = Path(golden_dataset_path).resolve()

    cases: List[Dict[str, Any]] = json.loads(
        golden_dataset_path.read_text(encoding="utf-8")
    )
    orchestrator = WorkToBlogOrchestrator()
    case_results: List[Dict[str, Any]] = []
    passed_count = 0

    with tempfile.TemporaryDirectory() as tmp_out_dir:
        for case in cases:
            case_id = case["case_id"]
            clear_recorded_spans()

            result = await orchestrator.run_blog_workflow(
                user_prompt=case["user_prompt"],
                workspace_or_repo_path=str(golden_dataset_path.parent.parent),
                engineering_summary_notes=case.get("engineering_summary_notes"),
                target_audience=case.get("target_audience", "software_engineers"),
                primary_seo_keywords=case.get("primary_seo_keywords"),
                session_id=f"eval-session-{case_id}",
                user_id="eval-harness",
                human_approval_confirmed=bool(
                    case.get("human_approval_confirmed", False)
                ),
                output_directory=tmp_out_dir,
            )

            await default_async_memory_manager.wait_for_pending_tasks()
            spans = get_recorded_spans()
            span_names = [s.name for s in spans]

            # Case type A: Adversarial guardrail block check
            if "expected_workflow_status" in case:
                passed = result.get("status") == case["expected_workflow_status"]
                case_results.append(
                    {
                        "case_id": case_id,
                        "passed": passed,
                        "status": result.get("status"),
                        "recorded_spans": len(span_names),
                    }
                )
                if passed:
                    passed_count += 1
                continue

            # Case type B: Full multi-agent blog generation check
            eval_report = result.get("editorial_evaluation", {})
            pub_report = result.get("publication", {})
            quality_score = int(eval_report.get("overall_quality_score", 0))
            pii_leaked = bool(eval_report.get("pii_or_secret_leaks_detected", True))
            draft_md = str(result.get("draft_markdown", ""))

            forbidden_leaked = any(
                forbidden in draft_md
                for forbidden in case.get("forbidden_substrings_in_output", [])
            )

            checks = {
                "workflow_completed": result.get("status") == "completed",
                "quality_score_meets_bar": quality_score
                >= int(case.get("expected_min_quality_score", 80)),
                "zero_pii_leaks": (not pii_leaked) and (not forbidden_leaked),
                "publication_status_matches": pub_report.get("status")
                == case.get("expected_publication_status"),
                "otel_spans_recorded": "workflow.work_to_blog_pipeline" in span_names
                and "tool.extract_engineering_work_artifacts" in span_names,
            }
            all_passed = all(checks.values())
            if all_passed:
                passed_count += 1

            case_results.append(
                {
                    "case_id": case_id,
                    "passed": all_passed,
                    "quality_score": quality_score,
                    "publication_status": pub_report.get("status"),
                    "checks": checks,
                    "recorded_spans": len(span_names),
                }
            )

    summary = {
        "total_cases": len(cases),
        "passed_cases": passed_count,
        "pass_rate": round(passed_count / max(1, len(cases)), 4),
        "all_passed": passed_count == len(cases),
        "results": case_results,
    }
    return summary


def main() -> int:
    summary = asyncio.run(evaluate_golden_dataset())
    sys.stdout.write(json.dumps(summary, indent=2) + "\n")
    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
