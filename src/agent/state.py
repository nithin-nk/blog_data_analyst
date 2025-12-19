"""
State module - State schema, Pydantic models, and job management.

This module defines:
- Phase enum for pipeline states
- BlogAgentState TypedDict for LangGraph
- Pydantic models for structured LLM outputs
- JobManager for checkpoint/resume functionality
"""

import json
import re
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, TypedDict

from pydantic import BaseModel, Field


# =============================================================================
# Phase Enum
# =============================================================================


class Phase(str, Enum):
    """
    Current phase of the blog agent pipeline.

    State flow:
    topic_discovery → planning → researching → validating_sources →
    writing → reviewing → assembling → final_review → done | failed
    """

    TOPIC_DISCOVERY = "topic_discovery"
    PLANNING = "planning"
    RESEARCHING = "researching"
    VALIDATING_SOURCES = "validating_sources"
    WRITING = "writing"
    REVIEWING = "reviewing"
    ASSEMBLING = "assembling"
    FINAL_REVIEW = "final_review"
    DONE = "done"
    FAILED = "failed"


# =============================================================================
# BlogAgentState (TypedDict for LangGraph)
# =============================================================================


class BlogAgentState(TypedDict, total=False):
    """
    LangGraph state for the blog agent.

    All data flows through this state between nodes.
    Designed to be JSON-serializable for checkpointing.
    """

    # === Input (set at start) ===
    job_id: str
    title: str
    context: str
    target_length: str  # "short" | "medium" | "long"
    flags: dict[str, bool]  # review_sections, review_final, no_citations, etc.

    # === Phase Tracking ===
    current_phase: str  # Phase enum value
    current_section_index: int

    # === Phase 0.5: Topic Discovery ===
    discovery_queries: list[str]
    topic_context: list[dict[str, str]]  # [{title, url, snippet}]

    # === Phase 1: Planning ===
    plan: dict[str, Any]  # BlogPlan as dict

    # === Phase 2: Research ===
    research_cache: dict[str, dict[str, Any]]  # url_hash -> content

    # === Phase 2.5: Validation ===
    validated_sources: dict[str, list[dict[str, Any]]]  # section_id -> sources

    # === Phase 3: Writing ===
    section_drafts: dict[str, str]  # section_id -> markdown
    section_reviews: dict[str, dict[str, Any]]  # section_id -> critic result
    fact_check_items: list[str]

    # === Phase 4: Assembly ===
    combined_draft: str
    final_review: dict[str, Any]
    rendered_diagrams: dict[str, str]  # diagram_id -> path

    # === Phase 5: Human Review ===
    human_review_decision: str  # "approve" | "edit" | "reject"
    human_feedback: str

    # === Output ===
    final_markdown: str
    metadata: dict[str, Any]

    # === Error Handling ===
    error_message: str | None


# =============================================================================
# Pydantic Models for LLM Structured Output
# =============================================================================


class DiscoveryQueries(BaseModel):
    """Output from topic discovery LLM call."""

    queries: list[str] = Field(
        description="3-5 search queries for topic discovery",
        min_length=3,
        max_length=5,
    )


class PlanSection(BaseModel):
    """A single section in the blog plan."""

    id: str = Field(description="Unique identifier for the section")
    title: str | None = Field(default=None, description="Section title")
    role: str = Field(
        description="Section role: hook, problem, why, implementation, deep_dive, conclusion"
    )
    search_queries: list[str] = Field(
        default_factory=list,
        description="2-3 specific queries for research",
    )
    needs_code: bool = Field(default=False, description="Whether section needs code examples")
    needs_diagram: bool = Field(
        default=False,
        description="Whether section needs a diagram",
    )
    target_words: int = Field(default=200, description="Target word count")
    optional: bool = Field(
        default=False,
        description="If true, this is an extra topic the user can choose to include or skip",
    )


class BlogPlan(BaseModel):
    """Complete blog plan from planning phase."""

    blog_title: str = Field(description="The blog post title")
    target_words: int = Field(description="Total target word count")
    sections: list[PlanSection] = Field(description="List of sections in order")


class CriticScore(BaseModel):
    """8 scoring dimensions (1-10 scale)."""

    technical_accuracy: int = Field(ge=1, le=10, description="Technical correctness")
    completeness: int = Field(ge=1, le=10, description="Coverage of topic")
    code_quality: int = Field(ge=1, le=10, description="Quality of code examples")
    clarity: int = Field(ge=1, le=10, description="Ease of understanding")
    voice: int = Field(ge=1, le=10, description="Consistent author voice")
    originality: int = Field(ge=1, le=10, description="Not copied from sources")
    length: int = Field(ge=1, le=10, description="Appropriate word count")
    diagram_quality: int = Field(ge=1, le=10, default=10, description="Diagram quality (10 if no diagrams)")


class CriticIssue(BaseModel):
    """A single issue identified by the critic."""

    dimension: str = Field(description="Which dimension failed: technical_accuracy, completeness, etc.")
    location: str = Field(description="Where in the section: 'paragraph 2', 'code block 1', 'diagram'")
    problem: str = Field(description="What's wrong")
    suggestion: str = Field(description="How to fix it")


class SectionCriticResult(BaseModel):
    """Full critic evaluation of a section."""

    scores: CriticScore
    overall_pass: bool = Field(description="Whether section passes quality gate (avg >= 8)")
    issues: list[CriticIssue] = Field(default_factory=list, description="Specific issues found")
    fact_check_needed: list[str] = Field(
        default_factory=list,
        description="Claims that need verification",
    )


class SourceValidation(BaseModel):
    """Validation result for a single source."""

    url: str = Field(description="Source URL")
    relevant: bool = Field(description="Is source relevant to section")
    quality: str = Field(description="Quality rating: high, medium, low")
    use: bool = Field(description="Whether to use this source")
    reason: str = Field(description="Reason for decision")


class SourceValidationList(BaseModel):
    """List of source validations for a section."""

    sources: list[SourceValidation] = Field(description="Validation results")


class AlternativeQueries(BaseModel):
    """Alternative search queries for retry when sources are insufficient."""

    queries: list[str] = Field(
        min_length=2,
        max_length=3,
        description="2-3 alternative search queries different from the originals",
    )


# =============================================================================
# JobManager - Checkpoint/Resume
# =============================================================================


class JobManager:
    """
    Manages blog agent jobs with checkpoint/resume functionality.

    Directory structure:
    ~/.blog_agent/
    └── jobs/
        └── {job_id}/              # e.g., "semantic-caching-for-llm"
            ├── state.json         # Current phase, progress
            ├── input.json         # Original title + context
            ├── plan.json          # Blog outline
            ├── research/
            │   └── cache/         # Fetched articles
            ├── drafts/
            │   └── sections/      # Individual section drafts
            └── images/            # Rendered diagrams

    Example:
        manager = JobManager()

        # Create new job
        job_id = manager.create_job(
            title="Semantic Caching for LLM Applications",
            context="Saw GPTCache on Twitter..."
        )
        # Returns: "semantic-caching-for-llm-applications"

        # Save state during execution
        manager.save_state(job_id, state)

        # Resume later
        state = manager.load_state(job_id)
    """

    BASE_DIR = Path.home() / ".blog_agent"

    def __init__(self, base_dir: Path | None = None):
        """
        Initialize JobManager.

        Args:
            base_dir: Override base directory (for testing)
        """
        if base_dir is not None:
            self.BASE_DIR = base_dir
        self._ensure_base_dirs()

    def _ensure_base_dirs(self) -> None:
        """Ensure base directories exist."""
        (self.BASE_DIR / "jobs").mkdir(parents=True, exist_ok=True)

    def create_job(
        self,
        title: str,
        context: str,
        target_length: str = "medium",
        flags: dict[str, bool] | None = None,
    ) -> str:
        """
        Create a new job and return its ID.

        Args:
            title: Blog title
            context: Context/notes for the blog
            target_length: "short", "medium", or "long"
            flags: Optional flags dict

        Returns:
            Job ID (slugified title)
        """
        job_id = self.slugify(title)
        job_dir = self.get_job_dir(job_id)

        # Create directory structure
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / "research" / "cache").mkdir(parents=True, exist_ok=True)
        (job_dir / "drafts" / "sections").mkdir(parents=True, exist_ok=True)
        (job_dir / "images").mkdir(exist_ok=True)

        # Save input
        input_data = {
            "title": title,
            "context": context,
            "target_length": target_length,
            "flags": flags or {},
            "created_at": datetime.now().isoformat(),
        }
        self._save_json(job_dir / "input.json", input_data)

        # Initialize state
        state_data = {
            "current_phase": Phase.TOPIC_DISCOVERY.value,
            "current_section_index": 0,
            "can_resume": True,
            "last_updated": datetime.now().isoformat(),
        }
        self._save_json(job_dir / "state.json", state_data)

        return job_id

    def save_state(self, job_id: str, state: BlogAgentState) -> None:
        """
        Save current state to disk.

        Saves:
        - state.json: phase, section_index, can_resume
        - plan.json: if plan exists
        - drafts/sections/{id}.md: each section draft
        """
        job_dir = self.get_job_dir(job_id)
        if not job_dir.exists():
            raise ValueError(f"Job not found: {job_id}")

        # Save main state
        state_data = {
            "current_phase": state.get("current_phase", Phase.TOPIC_DISCOVERY.value),
            "current_section_index": state.get("current_section_index", 0),
            "can_resume": True,
            "last_updated": datetime.now().isoformat(),
        }
        self._save_json(job_dir / "state.json", state_data)

        # Save topic context
        if state.get("topic_context"):
            self._save_json(
                job_dir / "topic_context.json",
                {
                    "queries_used": state.get("discovery_queries", []),
                    "results": state.get("topic_context", []),
                },
            )

        # Save plan
        if state.get("plan"):
            self._save_json(job_dir / "plan.json", state["plan"])

        # Save section drafts
        for section_id, content in state.get("section_drafts", {}).items():
            draft_path = job_dir / "drafts" / "sections" / f"{section_id}.md"
            draft_path.write_text(content)

        # Save section reviews
        for section_id, review in state.get("section_reviews", {}).items():
            review_path = job_dir / "feedback" / f"section_{section_id}_critic.json"
            review_path.parent.mkdir(exist_ok=True)
            self._save_json(review_path, review)

        # Save combined draft
        if state.get("combined_draft"):
            (job_dir / "drafts" / "v1.md").write_text(state["combined_draft"])

        # Save final
        if state.get("final_markdown"):
            (job_dir / "final.md").write_text(state["final_markdown"])

        # Save metadata
        if state.get("metadata"):
            self._save_json(job_dir / "metadata.json", state["metadata"])

    def load_state(self, job_id: str) -> BlogAgentState | None:
        """
        Load state from disk for resume.

        Returns:
            BlogAgentState if job exists, None otherwise
        """
        job_dir = self.get_job_dir(job_id)

        if not (job_dir / "state.json").exists():
            return None

        state: BlogAgentState = {}

        # Load main state
        main_state = self._load_json(job_dir / "state.json")
        state["current_phase"] = main_state.get("current_phase")
        state["current_section_index"] = main_state.get("current_section_index", 0)
        state["job_id"] = job_id

        # Load input
        if (job_dir / "input.json").exists():
            input_data = self._load_json(job_dir / "input.json")
            state["title"] = input_data.get("title", "")
            state["context"] = input_data.get("context", "")
            state["target_length"] = input_data.get("target_length", "medium")
            state["flags"] = input_data.get("flags", {})

        # Load topic context
        if (job_dir / "topic_context.json").exists():
            tc = self._load_json(job_dir / "topic_context.json")
            state["discovery_queries"] = tc.get("queries_used", [])
            state["topic_context"] = tc.get("results", [])

        # Load plan
        if (job_dir / "plan.json").exists():
            state["plan"] = self._load_json(job_dir / "plan.json")

        # Load section drafts
        drafts_dir = job_dir / "drafts" / "sections"
        if drafts_dir.exists():
            state["section_drafts"] = {}
            for draft_file in drafts_dir.glob("*.md"):
                section_id = draft_file.stem
                state["section_drafts"][section_id] = draft_file.read_text()

        # Load section reviews
        feedback_dir = job_dir / "feedback"
        if feedback_dir.exists():
            state["section_reviews"] = {}
            for review_file in feedback_dir.glob("section_*_critic.json"):
                # Extract section_id from filename
                match = re.search(r"section_(.+)_critic\.json", review_file.name)
                if match:
                    section_id = match.group(1)
                    state["section_reviews"][section_id] = self._load_json(review_file)

        # Load combined draft
        if (job_dir / "drafts" / "v1.md").exists():
            state["combined_draft"] = (job_dir / "drafts" / "v1.md").read_text()

        # Load final
        if (job_dir / "final.md").exists():
            state["final_markdown"] = (job_dir / "final.md").read_text()

        return state

    def list_jobs(self, status: str | None = None) -> list[dict[str, Any]]:
        """
        List all jobs with optional status filter.

        Args:
            status: Filter by "complete" or "incomplete"

        Returns:
            List of job info dicts
        """
        jobs = []
        jobs_dir = self.BASE_DIR / "jobs"

        if not jobs_dir.exists():
            return []

        for job_dir in jobs_dir.iterdir():
            if not job_dir.is_dir():
                continue

            state_file = job_dir / "state.json"
            if not state_file.exists():
                continue

            state = self._load_json(state_file)
            input_file = job_dir / "input.json"
            input_data = self._load_json(input_file) if input_file.exists() else {}

            is_complete = state.get("current_phase") == Phase.DONE.value

            job_info = {
                "job_id": job_dir.name,
                "title": input_data.get("title", ""),
                "phase": state.get("current_phase"),
                "last_updated": state.get("last_updated"),
                "complete": is_complete,
            }

            if status is None:
                jobs.append(job_info)
            elif status == "complete" and is_complete:
                jobs.append(job_info)
            elif status == "incomplete" and not is_complete:
                jobs.append(job_info)

        return sorted(jobs, key=lambda x: x.get("last_updated") or "", reverse=True)

    def get_job_dir(self, job_id: str) -> Path:
        """Get path to job directory."""
        return self.BASE_DIR / "jobs" / job_id

    @staticmethod
    def slugify(text: str, max_length: int = 50) -> str:
        """
        Convert text to URL-friendly slug.

        Args:
            text: Text to slugify
            max_length: Maximum slug length

        Returns:
            Slugified string

        Example:
            >>> JobManager.slugify("Semantic Caching for LLM Apps!")
            "semantic-caching-for-llm-apps"
        """
        # Convert to lowercase
        text = text.lower()
        # Replace special characters with hyphens
        text = re.sub(r"[^a-z0-9]+", "-", text)
        # Remove leading/trailing hyphens
        text = text.strip("-")
        # Truncate
        if len(text) > max_length:
            text = text[:max_length].rstrip("-")
        return text

    def _save_json(self, path: Path, data: dict[str, Any]) -> None:
        """Save data as JSON."""
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def _load_json(self, path: Path) -> dict[str, Any]:
        """Load JSON data."""
        with open(path) as f:
            return json.load(f)
