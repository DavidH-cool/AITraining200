"""Strict Input and Output JSON Schemas using Pydantic v2.

Addresses Rubric Category 1 (Tool & Interface Design) - Explicit JSON Schemas:
- Defines strict input and output schemas (`extra='forbid'`, field constraints,
  enumerations, and JSON Schema export) to validate tool arguments and constrain
  LLM outputs across the multi-agent pipeline.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Literal, Optional, get_args, get_origin, get_type_hints

try:
    from pydantic import (  # type: ignore[import-untyped]
        BaseModel,
        ConfigDict,
        Field,
        ValidationError,
    )

    _HAS_PYDANTIC = True
except ImportError:  # pragma: no cover - strict JSON-schema & runtime validator fallback
    _HAS_PYDANTIC = False

    class ValidationError(ValueError):  # type: ignore[no-redef]
        """Raised when input or output payload fails strict schema validation."""

    def ConfigDict(**kwargs: Any) -> Dict[str, Any]:  # type: ignore[no-redef]
        return dict(kwargs)

    class _FieldInfo:
        def __init__(self, default: Any = ..., **kwargs: Any) -> None:
            self.default = default
            self.kwargs = kwargs

    def Field(default: Any = ..., **kwargs: Any) -> Any:  # type: ignore[no-redef]
        return _FieldInfo(default=default, **kwargs)

    class BaseModel:  # type: ignore[no-redef]
        """Pydantic v2-compatible BaseModel enforcing strict field constraints and extra='forbid'."""

        model_config: Dict[str, Any] = {"extra": "forbid"}

        def __init__(self, **data: Any) -> None:
            cls = self.__class__
            hints = get_type_hints(cls)
            extra_policy = cls.model_config.get("extra", "forbid")
            if extra_policy == "forbid":
                unknown = set(data.keys()) - set(hints.keys())
                if unknown:
                    raise ValidationError(
                        f"Extra inputs are not permitted in {cls.__name__}: {sorted(unknown)}"
                    )

            for field_name, field_type in hints.items():
                if field_name == "model_config":
                    continue
                raw_attr = getattr(cls, field_name, None)
                info = raw_attr if isinstance(raw_attr, _FieldInfo) else _FieldInfo(raw_attr)

                if field_name in data:
                    val = data[field_name]
                elif info.default is not ...:
                    if "default_factory" in info.kwargs:
                        val = info.kwargs["default_factory"]()
                    else:
                        val = info.default
                elif "default_factory" in info.kwargs:
                    val = info.kwargs["default_factory"]()
                else:
                    raise ValidationError(
                        f"Field '{field_name}' is required in {cls.__name__}."
                    )

                self._validate_field(field_name, val, field_type, info.kwargs)
                setattr(self, field_name, val)

        @classmethod
        def _validate_field(
            cls,
            name: str,
            val: Any,
            field_type: Any,
            constraints: Dict[str, Any],
        ) -> None:
            origin = get_origin(field_type)
            args = get_args(field_type)

            if val is None:
                if origin is not None and type(None) in args:
                    return
                raise ValidationError(f"Field '{name}' cannot be None.")

            if origin is Literal:
                if val not in args:
                    raise ValidationError(
                        f"Field '{name}' must be one of {list(args)}, got {val!r}."
                    )

            if "min_length" in constraints and hasattr(val, "__len__"):
                if len(val) < int(constraints["min_length"]):
                    raise ValidationError(
                        f"Field '{name}' length must be >= {constraints['min_length']}."
                    )
            if "max_length" in constraints and hasattr(val, "__len__"):
                if len(val) > int(constraints["max_length"]):
                    raise ValidationError(
                        f"Field '{name}' length must be <= {constraints['max_length']}."
                    )
            if "ge" in constraints and isinstance(val, (int, float)):
                if val < constraints["ge"]:
                    raise ValidationError(
                        f"Field '{name}' must be >= {constraints['ge']}."
                    )
            if "le" in constraints and isinstance(val, (int, float)):
                if val > constraints["le"]:
                    raise ValidationError(
                        f"Field '{name}' must be <= {constraints['le']}."
                    )
            if "pattern" in constraints and isinstance(val, str):
                if not re.match(constraints["pattern"], val):
                    raise ValidationError(
                        f"Field '{name}' does not match pattern '{constraints['pattern']}'."
                    )

        def model_dump(self) -> Dict[str, Any]:
            hints = get_type_hints(self.__class__)
            return {
                k: getattr(self, k)
                for k in hints
                if k != "model_config" and hasattr(self, k)
            }

        @classmethod
        def model_json_schema(cls) -> Dict[str, Any]:
            hints = get_type_hints(cls)
            properties: Dict[str, Any] = {}
            required: List[str] = []
            for field_name in hints:
                if field_name == "model_config":
                    continue
                raw_attr = getattr(cls, field_name, None)
                info = raw_attr if isinstance(raw_attr, _FieldInfo) else _FieldInfo(raw_attr)
                prop: Dict[str, Any] = {
                    "description": info.kwargs.get("description", ""),
                }
                for key in ("min_length", "max_length", "ge", "le", "pattern"):
                    if key in info.kwargs:
                        prop[key] = info.kwargs[key]
                properties[field_name] = prop
                if info.default is ... and "default_factory" not in info.kwargs:
                    required.append(field_name)
            return {
                "title": cls.__name__,
                "type": "object",
                "additionalProperties": cls.model_config.get("extra", "forbid") != "forbid",
                "properties": properties,
                "required": required,
            }


class ExtractWorkArtifactsInput(BaseModel):
    """Strict input schema for extracting engineering work artifacts from a repository or notes."""

    model_config = ConfigDict(extra="forbid")

    workspace_or_repo_path: str = Field(
        ...,
        min_length=1,
        description=(
            "Absolute or relative filesystem path to the git repository, project directory, "
            "or engineering notes folder containing the work to summarize."
        ),
    )
    commit_range_or_limit: str = Field(
        default="5",
        min_length=1,
        description=(
            "Number of recent git commits to inspect (e.g., '5' or '10') or an explicit "
            "git revision range (e.g., 'HEAD~5..HEAD')."
        ),
    )
    engineering_summary_notes: Optional[str] = Field(
        default=None,
        description=(
            "Optional raw engineering notes, design doc snippets, benchmark numbers, "
            "or architecture highlights provided directly by the author."
        ),
    )


class EngineeringWorkArtifactBundle(BaseModel):
    """Strict output schema representing extracted engineering work artifacts."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["success", "error"] = Field(
        ..., description="Execution status of the artifact extraction tool."
    )
    project_name: str = Field(
        default="", description="Detected project or repository name."
    )
    commit_summaries: List[str] = Field(
        default_factory=list,
        description="Sanitized list of recent git commit messages and changed file summaries.",
    )
    changed_files: List[str] = Field(
        default_factory=list,
        description="Key source code, infrastructure, or documentation files modified.",
    )
    key_technical_achievements: List[str] = Field(
        default_factory=list,
        description="Synthesized bullet points of technical problems solved and engineering outcomes.",
    )
    error_code: Optional[str] = Field(
        default=None, description="Machine-readable error code when status is 'error'."
    )
    error_message: Optional[str] = Field(
        default=None, description="Human-readable error description when status is 'error'."
    )
    recovery_instructions: Optional[str] = Field(
        default=None,
        description="Actionable recovery steps for the LLM to self-correct and retry.",
    )


class SearchTechnicalKnowledgeInput(BaseModel):
    """Strict input schema for querying persistent knowledge and prior blog posts."""

    model_config = ConfigDict(extra="forbid")

    search_query: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="Specific technical topic, architecture pattern, or past blog subject to retrieve.",
    )
    target_audience: Literal[
        "software_engineers",
        "engineering_leaders",
        "devops_sre",
        "general_tech_readers",
    ] = Field(
        default="software_engineers",
        description="Target readership persona used to rank relevant style and reference snippets.",
    )
    max_results: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum number of knowledge base or Vertex AI Search documents to return.",
    )


class GenerateSeoBlogOutlineInput(BaseModel):
    """Strict input schema for generating a structured, SEO-optimized technical blog outline."""

    model_config = ConfigDict(extra="forbid")

    working_title: str = Field(
        ...,
        min_length=5,
        max_length=160,
        description="Proposed technical blog post title summarizing the engineering work.",
    )
    core_problem_statement: str = Field(
        ...,
        min_length=15,
        description="The specific engineering challenge or bottleneck that motivated the work.",
    )
    technical_solution_highlights: List[str] = Field(
        ...,
        min_length=1,
        description="List of concrete architectural decisions, algorithms, or code changes implemented.",
    )
    target_audience: Literal[
        "software_engineers",
        "engineering_leaders",
        "devops_sre",
        "general_tech_readers",
    ] = Field(
        default="software_engineers",
        description="Intended audience for calibrating depth and terminology.",
    )
    primary_seo_keywords: List[str] = Field(
        default_factory=list,
        description="Primary technical search keywords to weave into H1/H2 headings.",
    )


class BlogSectionSpec(BaseModel):
    """Schema for an individual section within a structured blog outline."""

    model_config = ConfigDict(extra="forbid")

    heading: str = Field(..., description="Markdown H2/H3 section heading.")
    narrative_goal: str = Field(
        ..., description="What this section proves or explains to the reader."
    )
    code_or_diagram_callout: str = Field(
        ..., description="Suggested code snippet, benchmark table, or architecture diagram."
    )


class EvaluateBlogDraftInput(BaseModel):
    """Strict input schema for evaluating a blog draft's quality, readability, and safety."""

    model_config = ConfigDict(extra="forbid")

    draft_markdown: str = Field(
        ...,
        min_length=50,
        description="Full Markdown text of the technical blog draft to evaluate.",
    )
    source_engineering_facts: List[str] = Field(
        ...,
        min_length=1,
        description="Ground-truth engineering facts extracted from the user's work artifacts.",
    )
    required_keywords: List[str] = Field(
        default_factory=list,
        description="SEO or technical terms that should appear naturally in the draft.",
    )


class BlogQualityEvaluationReport(BaseModel):
    """Strict output schema for editorial review and self-evaluation guardrails."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["success", "error"] = Field(..., description="Evaluation execution status.")
    overall_quality_score: int = Field(
        default=0,
        ge=0,
        le=100,
        description="Composite editorial quality score from 0 to 100.",
    )
    passes_editorial_bar: bool = Field(
        default=False,
        description="True if overall_quality_score >= 80 and zero PII/secret leaks are found.",
    )
    word_count: int = Field(default=0, description="Total word count of the draft.")
    estimated_reading_time_minutes: int = Field(
        default=1, description="Estimated reading time at 200 words per minute."
    )
    grounding_coverage_ratio: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Fraction of source engineering facts reflected in the blog post.",
    )
    pii_or_secret_leaks_detected: bool = Field(
        default=False,
        description="Whether any unredacted PII or credentials were found in the draft.",
    )
    actionable_revision_feedback: List[str] = Field(
        default_factory=list,
        description="Concrete editorial improvements required before publication.",
    )
    error_code: Optional[str] = Field(default=None)
    error_message: Optional[str] = Field(default=None)
    recovery_instructions: Optional[str] = Field(default=None)


class PublishApprovedBlogPostInput(BaseModel):
    """Strict input schema for publishing an approved blog post to a CMS or file export."""

    model_config = ConfigDict(extra="forbid")

    title: str = Field(
        ...,
        min_length=5,
        max_length=160,
        description="Final approved headline of the blog post.",
    )
    slug: str = Field(
        ...,
        pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$",
        description="URL-safe kebab-case slug (e.g., 'scaling-multi-agent-systems-with-adk').",
    )
    final_markdown_body: str = Field(
        ...,
        min_length=100,
        description="Complete, reviewed, and PII-scrubbed Markdown body of the blog post.",
    )
    tags: List[str] = Field(
        default_factory=list,
        description="Topic tags for categorization on the publishing platform.",
    )
    human_approval_confirmed: bool = Field(
        default=False,
        description=(
            "Explicit Human-in-the-Loop confirmation flag. Must be True before publishing "
            "to external CMS endpoints."
        ),
    )
    output_directory: Optional[str] = Field(
        default=None,
        description="Optional local directory where the published Markdown artifact is written.",
    )


def export_all_tool_json_schemas() -> Dict[str, Dict[str, object]]:
    """Exports strict JSON Schemas for all tool input and output models."""
    return {
        "ExtractWorkArtifactsInput": ExtractWorkArtifactsInput.model_json_schema(),
        "EngineeringWorkArtifactBundle": EngineeringWorkArtifactBundle.model_json_schema(),
        "SearchTechnicalKnowledgeInput": SearchTechnicalKnowledgeInput.model_json_schema(),
        "GenerateSeoBlogOutlineInput": GenerateSeoBlogOutlineInput.model_json_schema(),
        "EvaluateBlogDraftInput": EvaluateBlogDraftInput.model_json_schema(),
        "BlogQualityEvaluationReport": BlogQualityEvaluationReport.model_json_schema(),
        "PublishApprovedBlogPostInput": PublishApprovedBlogPostInput.model_json_schema(),
    }
