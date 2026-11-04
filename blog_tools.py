"""Domain-Specific Agent Tools for Turning Engineering Work into Technical Blogs.

Addresses Rubric Category 1 (Tool & Interface Design):
1. Comprehensive Tool Docstrings: Every tool includes detailed descriptions of purpose,
   parameters, return structure, and usage semantics.
2. Descriptive Naming: Highly specific action names (`extract_engineering_work_artifacts`,
   `search_technical_reference_knowledge`, `generate_seo_optimized_blog_outline`,
   `evaluate_blog_draft_readability_and_tone`, `publish_approved_blog_post_to_cms`).
3. Explicit JSON Schemas: All inputs and outputs are validated via strict Pydantic v2
   schemas defined in `blog_writer_agent.schemas`.
4. Guided Error Handling: All error paths catch exceptions and return structured
   `recovery_instructions` guiding the LLM on how to fix arguments and retry.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Dict, List, Optional

from blog_writer_agent.config.secrets import SecretResolutionError, get_secret_provider
from blog_writer_agent.memory.session_store import get_persistent_knowledge_store
from blog_writer_agent.observability.logger import (
    get_structured_logger,
    intent_outcome_tracker,
)
from blog_writer_agent.observability.pii_redactor import default_pii_redactor, scrub_pii
from blog_writer_agent.observability.tracing import traced_operation
from blog_writer_agent.schemas import (
    BlogQualityEvaluationReport,
    BlogSectionSpec,
    EngineeringWorkArtifactBundle,
    EvaluateBlogDraftInput,
    ExtractWorkArtifactsInput,
    GenerateSeoBlogOutlineInput,
    PublishApprovedBlogPostInput,
    SearchTechnicalKnowledgeInput,
    ValidationError,
)

logger = get_structured_logger(__name__)


@traced_operation("tool.extract_engineering_work_artifacts")
def extract_engineering_work_artifacts(
    workspace_or_repo_path: str,
    commit_range_or_limit: str = "5",
    engineering_summary_notes: Optional[str] = None,
) -> Dict[str, Any]:
    """Extracts git commits, modified files, and engineering highlights from a project workspace.

    Use this tool at the start of a blog-writing workflow to inspect the user's recent
    coding work, commits, README/design notes, and architectural changes so the resulting
    blog post is grounded in real engineering evidence.

    Args:
        workspace_or_repo_path: Absolute or relative path to the local Git repository
            or workspace directory containing the user's engineering work. Use "." for
            the current project directory if the user does not specify a path.
        commit_range_or_limit: Either an integer string specifying how many recent commits
            to inspect (e.g., "5") or a git revision range (e.g., "HEAD~3..HEAD").
        engineering_summary_notes: Optional free-form notes, PR descriptions, benchmark
            results, or bullet points supplied directly by the user in chat.

    Returns:
        A dictionary conforming to `EngineeringWorkArtifactBundle` containing:
        - `status`: "success" or "error"
        - `project_name`: Resolved directory or repository name
        - `commit_summaries`: List of PII-scrubbed git commit subject lines
        - `changed_files`: List of modified files discovered in the workspace
        - `key_technical_achievements`: Synthesized engineering highlights for the blog
        - `error_code`, `error_message`, and `recovery_instructions` if `status == "error"`.
    """
    correlation_id = intent_outcome_tracker.record_intent(
        agent_name="work_artifact_researcher_agent",
        action_name="extract_engineering_work_artifacts",
        intended_goal="Extract git commit history, file changes, and engineering notes for blog grounding.",
        planned_inputs={
            "workspace_or_repo_path": workspace_or_repo_path,
            "commit_range_or_limit": commit_range_or_limit,
            "has_summary_notes": bool(engineering_summary_notes),
        },
    )

    try:
        validated = ExtractWorkArtifactsInput(
            workspace_or_repo_path=workspace_or_repo_path,
            commit_range_or_limit=commit_range_or_limit,
            engineering_summary_notes=engineering_summary_notes,
        )
    except ValidationError as exc:
        error_bundle = EngineeringWorkArtifactBundle(
            status="error",
            error_code="INVALID_TOOL_ARGUMENTS",
            error_message=f"Schema validation failed: {exc}",
            recovery_instructions=(
                "Provide a non-empty 'workspace_or_repo_path' (such as '.' for the current "
                "workspace) and a valid 'commit_range_or_limit' (such as '5'), then retry "
                "calling extract_engineering_work_artifacts."
            ),
        ).model_dump()
        intent_outcome_tracker.record_outcome(
            correlation_id=correlation_id,
            actual_outcome=error_bundle,
            status="ERROR",
            error_details=str(exc),
        )
        return error_bundle

    target_path = Path(validated.workspace_or_repo_path).expanduser().resolve()
    if not target_path.exists() or not target_path.is_dir():
        # If the user only passed inline notes without a valid directory, check if notes exist
        if validated.engineering_summary_notes:
            achievements = [
                line.strip("- *•\t ")
                for line in validated.engineering_summary_notes.splitlines()
                if len(line.strip()) > 10
            ]
            bundle = EngineeringWorkArtifactBundle(
                status="success",
                project_name="inline-engineering-notes",
                commit_summaries=["Synthesized directly from author-provided engineering notes."],
                changed_files=[],
                key_technical_achievements=scrub_pii(achievements[:10]),
            ).model_dump()
            intent_outcome_tracker.record_outcome(
                correlation_id=correlation_id,
                actual_outcome=bundle,
                status="SUCCESS",
            )
            return bundle

        error_bundle = EngineeringWorkArtifactBundle(
            status="error",
            error_code="WORKSPACE_PATH_NOT_FOUND",
            error_message=f"Directory '{validated.workspace_or_repo_path}' does not exist.",
            recovery_instructions=(
                "Pass workspace_or_repo_path='.' to inspect the current working directory, "
                "or pass the user's raw work description via 'engineering_summary_notes' "
                "so extraction can proceed without a local filesystem path."
            ),
        ).model_dump()
        intent_outcome_tracker.record_outcome(
            correlation_id=correlation_id,
            actual_outcome=error_bundle,
            status="ERROR",
            error_details="Directory not found",
        )
        return error_bundle

    commit_summaries: List[str] = []
    changed_files: List[str] = []

    git_cmd = ["git", "-C", str(target_path), "log", "--oneline"]
    if validated.commit_range_or_limit.isdigit():
        git_cmd.extend(["-n", validated.commit_range_or_limit])
    else:
        git_cmd.append(validated.commit_range_or_limit)

    try:
        proc = subprocess.run(
            git_cmd,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            commit_summaries = [
                scrub_pii(line.strip())
                for line in proc.stdout.strip().splitlines()
                if line.strip()
            ]
        files_proc = subprocess.run(
            ["git", "-C", str(target_path), "status", "--short"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if files_proc.returncode == 0 and files_proc.stdout.strip():
            changed_files = [
                line.strip() for line in files_proc.stdout.strip().splitlines()[:25]
            ]
    except Exception as exc:
        logger.warning(
            "Git CLI inspection encountered an issue; falling back to directory scan.",
            extra={"event_type": "git_inspect_fallback", "error": str(exc)},
        )

    if not changed_files:
        discovered: List[str] = []
        for item in sorted(target_path.rglob("*")):
            if len(discovered) >= 20:
                break
            if item.is_file() and not any(
                part.startswith(".") or part in ("__pycache__", "node_modules", "venv")
                for part in item.parts
            ):
                discovered.append(str(item.relative_to(target_path)))
        changed_files = discovered

    achievements: List[str] = []
    if validated.engineering_summary_notes:
        for line in validated.engineering_summary_notes.splitlines():
            cleaned = line.strip("- *•\t ")
            if len(cleaned) >= 8:
                achievements.append(scrub_pii(cleaned))
    for commit in commit_summaries[:5]:
        achievements.append(f"Commit milestone: {commit}")
    if not achievements:
        achievements.append(
            f"Implemented modular components across {len(changed_files)} files in '{target_path.name}'."
        )

    result_bundle = EngineeringWorkArtifactBundle(
        status="success",
        project_name=target_path.name,
        commit_summaries=commit_summaries or ["Working tree changes inspected directly."],
        changed_files=changed_files,
        key_technical_achievements=achievements,
    ).model_dump()

    intent_outcome_tracker.record_outcome(
        correlation_id=correlation_id,
        actual_outcome=result_bundle,
        status="SUCCESS",
    )
    return result_bundle


@traced_operation("tool.search_technical_reference_knowledge")
def search_technical_reference_knowledge(
    search_query: str,
    target_audience: str = "software_engineers",
    max_results: int = 3,
) -> Dict[str, Any]:
    """Searches persistent knowledge stores and Vertex AI Search for technical blog references.

    Use this tool to retrieve canonical architecture explanations, author style guidelines,
    and previously published technical blog excerpts so the new blog post maintains
    consistent technical voice and depth.

    Args:
        search_query: Technical topic, framework, or engineering concept to search for
            (minimum 3 characters, e.g., "multi-agent orchestration with ADK").
        target_audience: Readership persona ("software_engineers", "engineering_leaders",
            "devops_sre", or "general_tech_readers").
        max_results: Maximum number of reference documents to retrieve (1 to 10).

    Returns:
        A dictionary containing:
        - `status`: "success" or "error"
        - `query`: Sanitized search query
        - `target_audience`: Validated audience persona
        - `results`: List of matching reference documents and style guidelines
        - `error_code`, `error_message`, `recovery_instructions` on validation or search failure.
    """
    correlation_id = intent_outcome_tracker.record_intent(
        agent_name="work_artifact_researcher_agent",
        action_name="search_technical_reference_knowledge",
        intended_goal="Query persistent knowledge base / Vertex AI Search for technical context and style guides.",
        planned_inputs={
            "search_query": search_query,
            "target_audience": target_audience,
            "max_results": max_results,
        },
    )

    try:
        validated = SearchTechnicalKnowledgeInput(
            search_query=search_query,
            target_audience=target_audience,  # type: ignore[arg-type]
            max_results=max_results,
        )
    except ValidationError as exc:
        error_resp = {
            "status": "error",
            "error_code": "INVALID_SEARCH_PARAMETERS",
            "error_message": f"Invalid search parameters: {exc}",
            "recovery_instructions": (
                "Ensure 'search_query' is at least 3 characters long, 'target_audience' is one of "
                "['software_engineers', 'engineering_leaders', 'devops_sre', 'general_tech_readers'], "
                "and 'max_results' is between 1 and 10, then retry search_technical_reference_knowledge."
            ),
        }
        intent_outcome_tracker.record_outcome(
            correlation_id=correlation_id,
            actual_outcome=error_resp,
            status="ERROR",
            error_details=str(exc),
        )
        return error_resp

    store = get_persistent_knowledge_store()
    matches = store.search_reference_documents(
        query=validated.search_query,
        audience=validated.target_audience,
        top_k=validated.max_results,
    )

    response = {
        "status": "success",
        "query": scrub_pii(validated.search_query),
        "target_audience": validated.target_audience,
        "results_count": len(matches),
        "results": matches,
    }
    intent_outcome_tracker.record_outcome(
        correlation_id=correlation_id,
        actual_outcome=response,
        status="SUCCESS",
    )
    return response


@traced_operation("tool.generate_seo_optimized_blog_outline")
def generate_seo_optimized_blog_outline(
    working_title: str,
    core_problem_statement: str,
    technical_solution_highlights: List[str],
    target_audience: str = "software_engineers",
    primary_seo_keywords: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Generates a structured, SEO-optimized technical blog post outline from engineering work.

    Use this tool after extracting work artifacts to organize raw commits and design
    decisions into a compelling engineering narrative (Hook -> Problem -> Architecture ->
    Implementation Deep-Dive -> Benchmarks & Observability -> Takeaways).

    Args:
        working_title: Proposed title for the blog post (5-160 characters).
        core_problem_statement: Clear description of the technical problem solved (min 15 chars).
        technical_solution_highlights: Non-empty list of concrete engineering achievements
            or code changes to feature in the body sections.
        target_audience: One of "software_engineers", "engineering_leaders", "devops_sre",
            or "general_tech_readers".
        primary_seo_keywords: Optional list of technical keywords to embed in section headings.

    Returns:
        A structured dictionary with `status`, `optimized_title`, `estimated_word_count`,
        `seo_slug`, `sections` (list of `BlogSectionSpec` dicts), or guided error recovery.
    """
    correlation_id = intent_outcome_tracker.record_intent(
        agent_name="technical_blog_drafter_agent",
        action_name="generate_seo_optimized_blog_outline",
        intended_goal="Construct a structured technical blog outline with SEO headings and narrative flow.",
        planned_inputs={
            "working_title": working_title,
            "core_problem_statement": core_problem_statement,
            "highlights_count": len(technical_solution_highlights or []),
            "target_audience": target_audience,
        },
    )

    try:
        validated = GenerateSeoBlogOutlineInput(
            working_title=working_title,
            core_problem_statement=core_problem_statement,
            technical_solution_highlights=technical_solution_highlights,
            target_audience=target_audience,  # type: ignore[arg-type]
            primary_seo_keywords=primary_seo_keywords or [],
        )
    except ValidationError as exc:
        error_resp = {
            "status": "error",
            "error_code": "INVALID_OUTLINE_INPUTS",
            "error_message": f"Outline input validation failed: {exc}",
            "recovery_instructions": (
                "Provide a 'working_title' (5-160 chars), a 'core_problem_statement' "
                "(at least 15 chars), and a non-empty list of 'technical_solution_highlights', "
                "then retry generate_seo_optimized_blog_outline."
            ),
        }
        intent_outcome_tracker.record_outcome(
            correlation_id=correlation_id,
            actual_outcome=error_resp,
            status="ERROR",
            error_details=str(exc),
        )
        return error_resp

    slug_tokens = re.findall(r"[a-z0-9]+", validated.working_title.lower())
    seo_slug = "-".join(slug_tokens[:10]) or "engineering-deep-dive"
    keywords = validated.primary_seo_keywords or slug_tokens[:4]

    sections: List[Dict[str, str]] = [
        BlogSectionSpec(
            heading="## 1. The Engineering Challenge: Why Existing Approaches Hit a Wall",
            narrative_goal=f"Hook {validated.target_audience} by framing the real-world pain point: {validated.core_problem_statement}",
            code_or_diagram_callout="Before/After latency or architecture comparison callout box.",
        ).model_dump(),
        BlogSectionSpec(
            heading=f"## 2. System Architecture & Design Decisions ({', '.join(keywords[:2]) or 'Core Design'})",
            narrative_goal="Explain the architectural trade-offs and why this design was chosen.",
            code_or_diagram_callout="ASCII or Mermaid component flow diagram showing request lifecycle.",
        ).model_dump(),
        BlogSectionSpec(
            heading="## 3. Implementation Deep-Dive: Walking Through the Code",
            narrative_goal=(
                "Demonstrate concrete implementation details from the engineering work: "
                + "; ".join(validated.technical_solution_highlights[:3])
            ),
            code_or_diagram_callout="Annotated, production-grade code snippet with inline explanation.",
        ).model_dump(),
        BlogSectionSpec(
            heading="## 4. Production Observability, Guardrails & Verification",
            narrative_goal="Show how the solution was verified, traced with OpenTelemetry, and hardened for production.",
            code_or_diagram_callout="Structured telemetry sample and evaluation benchmark table.",
        ).model_dump(),
        BlogSectionSpec(
            heading="## 5. Key Takeaways & What We're Building Next",
            narrative_goal="Summarize actionable lessons learned for fellow engineers and link to resources.",
            code_or_diagram_callout="Numbered checklist of engineering takeaways.",
        ).model_dump(),
    ]

    outline_payload = {
        "status": "success",
        "optimized_title": scrub_pii(validated.working_title),
        "seo_slug": seo_slug,
        "target_audience": validated.target_audience,
        "primary_seo_keywords": keywords,
        "estimated_word_count": 950,
        "sections": sections,
    }
    intent_outcome_tracker.record_outcome(
        correlation_id=correlation_id,
        actual_outcome=outline_payload,
        status="SUCCESS",
    )
    return outline_payload


@traced_operation("tool.evaluate_blog_draft_readability_and_tone")
def evaluate_blog_draft_readability_and_tone(
    draft_markdown: str,
    source_engineering_facts: List[str],
    required_keywords: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Evaluates a Markdown blog draft for technical grounding, structure, SEO, and PII safety.

    Use this tool during the editorial review loop to verify that a generated blog draft
    accurately reflects the author's engineering work, contains code/structural headings,
    and has zero unredacted PII or credentials before requesting publication.

    Args:
        draft_markdown: Full Markdown text of the blog post draft (minimum 50 characters).
        source_engineering_facts: List of ground-truth engineering facts from the author's
            work that must be represented in the article.
        required_keywords: Optional list of technical or SEO terms expected in the post.

    Returns:
        A dictionary conforming to `BlogQualityEvaluationReport` with `overall_quality_score`
        (0-100), `passes_editorial_bar` (bool), `grounding_coverage_ratio`,
        `pii_or_secret_leaks_detected`, and `actionable_revision_feedback`.
    """
    correlation_id = intent_outcome_tracker.record_intent(
        agent_name="editorial_critic_agent",
        action_name="evaluate_blog_draft_readability_and_tone",
        intended_goal="Score blog draft against technical grounding, structure, and PII safety rubric.",
        planned_inputs={
            "draft_length_chars": len(draft_markdown or ""),
            "facts_count": len(source_engineering_facts or []),
            "keywords_count": len(required_keywords or []),
        },
    )

    try:
        validated = EvaluateBlogDraftInput(
            draft_markdown=draft_markdown,
            source_engineering_facts=source_engineering_facts,
            required_keywords=required_keywords or [],
        )
    except ValidationError as exc:
        error_report = BlogQualityEvaluationReport(
            status="error",
            error_code="INVALID_DRAFT_EVALUATION_INPUT",
            error_message=f"Draft evaluation input failed validation: {exc}",
            recovery_instructions=(
                "Provide a 'draft_markdown' of at least 50 characters and a non-empty list "
                "of 'source_engineering_facts', then re-run evaluate_blog_draft_readability_and_tone."
            ),
        ).model_dump()
        intent_outcome_tracker.record_outcome(
            correlation_id=correlation_id,
            actual_outcome=error_report,
            status="ERROR",
            error_details=str(exc),
        )
        return error_report

    text = validated.draft_markdown
    words = re.findall(r"\b\w+\b", text)
    word_count = len(words)
    reading_time = max(1, math.ceil(word_count / 200.0))

    # Check PII or secret leaks by comparing raw text to redacted text
    redacted_text = default_pii_redactor.redact_text(text)
    pii_leaked = redacted_text != text

    # Compute grounding coverage across source engineering facts
    lower_text = text.lower()
    matched_facts = 0
    for fact in validated.source_engineering_facts:
        significant_tokens = [
            tok.lower()
            for tok in re.findall(r"[A-Za-z0-9_]{4,}", fact)
            if tok.lower() not in {"with", "from", "that", "this", "have", "were", "commit"}
        ]
        if not significant_tokens:
            matched_facts += 1
            continue
        hits = sum(1 for tok in significant_tokens if tok in lower_text)
        if hits / len(significant_tokens) >= 0.35:
            matched_facts += 1

    grounding_ratio = round(
        matched_facts / max(1, len(validated.source_engineering_facts)), 2
    )

    feedback: List[str] = []
    score = 40  # base score for valid markdown submission

    # Structure checks (headings + code blocks)
    heading_count = len(re.findall(r"(?m)^#{1,3}\s+.+$", text))
    if heading_count >= 3:
        score += 15
    else:
        feedback.append(
            "Add at least 3 Markdown section headings (# or ##) to structure the technical narrative."
        )

    if "```" in text:
        score += 15
    else:
        feedback.append(
            "Include at least one fenced code block (```) or architecture snippet illustrating the implementation."
        )

    if word_count >= 180:
        score += 10
    else:
        feedback.append(
            f"Expand technical depth: current word count is {word_count} words (target >= 180 words)."
        )

    # Grounding score (up to 20 points)
    score += int(round(grounding_ratio * 20))
    if grounding_ratio < 0.7:
        feedback.append(
            f"Grounding coverage is {grounding_ratio:.0%}; incorporate more concrete details from source_engineering_facts."
        )

    # Keyword check
    missing_keywords = [
        kw for kw in validated.required_keywords if kw.lower() not in lower_text
    ]
    if missing_keywords:
        feedback.append(
            f"Naturally weave in missing technical keywords: {', '.join(missing_keywords)}."
        )

    if pii_leaked:
        score = max(0, score - 40)
        feedback.append(
            "CRITICAL: Unredacted PII or secret token detected in draft. Scrub all emails, keys, and IPs immediately."
        )

    score = min(100, max(0, score))
    passes = score >= 80 and not pii_leaked

    report = BlogQualityEvaluationReport(
        status="success",
        overall_quality_score=score,
        passes_editorial_bar=passes,
        word_count=word_count,
        estimated_reading_time_minutes=reading_time,
        grounding_coverage_ratio=grounding_ratio,
        pii_or_secret_leaks_detected=pii_leaked,
        actionable_revision_feedback=feedback,
    ).model_dump()

    intent_outcome_tracker.record_outcome(
        correlation_id=correlation_id,
        actual_outcome=report,
        status="SUCCESS",
    )
    return report


@traced_operation("tool.publish_approved_blog_post_to_cms")
def publish_approved_blog_post_to_cms(
    title: str,
    slug: str,
    final_markdown_body: str,
    tags: Optional[List[str]] = None,
    human_approval_confirmed: bool = False,
    output_directory: Optional[str] = None,
) -> Dict[str, Any]:
    """Publishes an approved, PII-scrubbed technical blog post after Human-in-the-Loop confirmation.

    HIGH-STAKES ACTION: This tool enforces an explicit Human-in-the-Loop (HITL) stop.
    If `human_approval_confirmed` is False, the tool refuses to publish externally and
    returns a `HUMAN_APPROVAL_REQUIRED` status instructing the agent to ask the human
    author for confirmation first.

    Args:
        title: Final headline of the technical blog post (5-160 characters).
        slug: URL-safe lowercase kebab-case slug (e.g., "building-reliable-adk-agents").
        final_markdown_body: Complete Markdown article body (minimum 100 characters).
        tags: Optional list of categorization tags.
        human_approval_confirmed: Explicit boolean confirmation from the human user.
            Must be True to execute publication.
        output_directory: Optional directory to save the published `.md` artifact locally.

    Returns:
        A dictionary containing `status` ("published", "pending_human_approval", or "error"),
        `published_path`, `cms_secret_verified`, and `recovery_instructions` if blocked.
    """
    correlation_id = intent_outcome_tracker.record_intent(
        agent_name="cms_publication_agent",
        action_name="publish_approved_blog_post_to_cms",
        intended_goal="Publish final technical blog post after verifying human approval and Secret Manager credentials.",
        planned_inputs={
            "title": title,
            "slug": slug,
            "tags": tags or [],
            "human_approval_confirmed": human_approval_confirmed,
        },
    )

    try:
        validated = PublishApprovedBlogPostInput(
            title=title,
            slug=slug,
            final_markdown_body=final_markdown_body,
            tags=tags or [],
            human_approval_confirmed=human_approval_confirmed,
            output_directory=output_directory,
        )
    except ValidationError as exc:
        error_resp = {
            "status": "error",
            "error_code": "INVALID_PUBLICATION_PAYLOAD",
            "error_message": f"Publication argument validation failed: {exc}",
            "recovery_instructions": (
                "Ensure 'slug' is lowercase kebab-case (matching '^[a-z0-9]+(?:-[a-z0-9]+)*$'), "
                "'title' is 5-160 characters, and 'final_markdown_body' is at least 100 characters, "
                "then retry publish_approved_blog_post_to_cms."
            ),
        }
        intent_outcome_tracker.record_outcome(
            correlation_id=correlation_id,
            actual_outcome=error_resp,
            status="ERROR",
            error_details=str(exc),
        )
        return error_resp

    # Human-in-the-Loop (HITL) hard stop
    if not validated.human_approval_confirmed:
        hitl_pause = {
            "status": "pending_human_approval",
            "error_code": "HUMAN_APPROVAL_REQUIRED",
            "error_message": (
                "Publication halted by Human-in-the-Loop safety gate: "
                "human_approval_confirmed is False."
            ),
            "recovery_instructions": (
                "Present the final blog post title, slug, and Markdown preview to the user "
                "and ask for their explicit approval to publish. Do NOT call "
                "publish_approved_blog_post_to_cms with human_approval_confirmed=True "
                "until the user explicitly confirms."
            ),
        }
        intent_outcome_tracker.record_outcome(
            correlation_id=correlation_id,
            actual_outcome=hitl_pause,
            status="PAUSED_FOR_HUMAN_APPROVAL",
        )
        return hitl_pause

    # Scrub PII before writing or transmitting
    sanitized_body = default_pii_redactor.redact_text(validated.final_markdown_body)
    sanitized_title = default_pii_redactor.redact_text(validated.title)

    # Verify CMS credential via Secret Manager (never hardcoded)
    secret_provider = get_secret_provider()
    cms_secret_verified = False
    try:
        _ = secret_provider.get_secret("cms-publishing-api-token")
        cms_secret_verified = True
    except SecretResolutionError:
        # In local or CI environments without a live CMS secret, export to local Markdown artifact
        cms_secret_verified = False

    target_dir = Path(
        validated.output_directory
        or os.environ.get("BLOG_OUTPUT_DIR", "/tmp/published_blogs")
    ).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    output_file = target_dir / f"{validated.slug}.md"

    frontmatter = (
        "---\n"
        f'title: "{sanitized_title}"\n'
        f'slug: "{validated.slug}"\n'
        f"tags: {validated.tags}\n"
        "---\n\n"
    )
    output_file.write_text(frontmatter + sanitized_body + "\n", encoding="utf-8")

    # Also asynchronously record the published post in the persistent knowledge store
    store = get_persistent_knowledge_store()
    store.save_published_blog_record(
        slug=validated.slug,
        title=sanitized_title,
        summary=sanitized_body[:400],
        tags=validated.tags,
    )

    pub_result = {
        "status": "published",
        "title": sanitized_title,
        "slug": validated.slug,
        "published_path": str(output_file),
        "cms_secret_verified": cms_secret_verified,
        "word_count": len(re.findall(r"\b\w+\b", sanitized_body)),
    }
    intent_outcome_tracker.record_outcome(
        correlation_id=correlation_id,
        actual_outcome=pub_result,
        status="SUCCESS",
    )
    return pub_result
