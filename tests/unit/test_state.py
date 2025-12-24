"""Unit tests for state module."""

import json
from datetime import datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.agent.state import (
    Phase,
    BlogAgentState,
    DiscoveryQueries,
    PlanSection,
    BlogPlan,
    CriticScore,
    SectionCriticResult,
    SourceValidation,
    SourceValidationList,
    JobManager,
)


# =============================================================================
# Phase Enum Tests
# =============================================================================


class TestPhaseEnum:
    """Tests for Phase enum."""

    def test_all_phases_defined(self):
        """All expected phases exist."""
        expected = [
            "topic_discovery",
            "content_landscape",
            "planning",
            "researching",
            "validating_sources",
            "writing",
            "reviewing",
            "assembling",
            "final_review",
            "done",
            "failed",
        ]
        actual = [p.value for p in Phase]
        assert actual == expected

    def test_phase_is_string(self):
        """Phase values are strings (JSON-serializable)."""
        for phase in Phase:
            assert isinstance(phase.value, str)
            # Can be used in JSON
            json.dumps({"phase": phase.value})

    def test_phase_comparison(self):
        """Phase can be compared to strings."""
        assert Phase.PLANNING.value == "planning"
        assert Phase.DONE.value == "done"

    def test_phase_from_value(self):
        """Phase can be created from string value."""
        phase = Phase("planning")
        assert phase == Phase.PLANNING


# =============================================================================
# Pydantic Models Tests
# =============================================================================


class TestDiscoveryQueries:
    """Tests for DiscoveryQueries model."""

    def test_valid_queries(self):
        """Valid query list is accepted."""
        dq = DiscoveryQueries(queries=["query1", "query2", "query3"])
        assert len(dq.queries) == 3

    def test_min_queries_validation(self):
        """Less than 3 queries raises error."""
        with pytest.raises(ValidationError):
            DiscoveryQueries(queries=["query1", "query2"])

    def test_max_queries_validation(self):
        """More than 5 queries raises error."""
        with pytest.raises(ValidationError):
            DiscoveryQueries(queries=["q1", "q2", "q3", "q4", "q5", "q6"])

    def test_five_queries_valid(self):
        """Exactly 5 queries is valid."""
        dq = DiscoveryQueries(queries=["q1", "q2", "q3", "q4", "q5"])
        assert len(dq.queries) == 5


class TestPlanSection:
    """Tests for PlanSection model."""

    def test_minimal_section(self):
        """Section with only required fields."""
        section = PlanSection(id="intro", role="hook")
        assert section.id == "intro"
        assert section.role == "hook"
        assert section.title is None
        assert section.search_queries == []
        assert section.needs_code is False
        assert section.needs_diagram is False
        assert section.target_words == 200

    def test_full_section(self):
        """Section with all fields."""
        section = PlanSection(
            id="implementation",
            title="Building the Solution",
            role="implementation",
            search_queries=["how to implement caching", "redis python tutorial"],
            needs_code=True,
            needs_diagram=True,
            target_words=500,
        )
        assert section.title == "Building the Solution"
        assert len(section.search_queries) == 2
        assert section.needs_code is True
        assert section.target_words == 500


class TestBlogPlan:
    """Tests for BlogPlan model."""

    def test_valid_plan(self):
        """Valid blog plan structure."""
        plan = BlogPlan(
            blog_title="Semantic Caching for LLMs",
            target_words=2000,
            sections=[
                PlanSection(id="intro", role="hook"),
                PlanSection(id="problem", role="problem"),
                PlanSection(id="conclusion", role="conclusion"),
            ],
        )
        assert plan.blog_title == "Semantic Caching for LLMs"
        assert plan.target_words == 2000
        assert len(plan.sections) == 3

    def test_plan_to_dict(self):
        """Plan can be converted to dict for state storage."""
        plan = BlogPlan(
            blog_title="Test Blog",
            target_words=1000,
            sections=[PlanSection(id="s1", role="hook")],
        )
        plan_dict = plan.model_dump()
        assert isinstance(plan_dict, dict)
        assert plan_dict["blog_title"] == "Test Blog"


class TestCriticScore:
    """Tests for CriticScore model."""

    def test_valid_scores(self):
        """Valid scores (1-10) are accepted."""
        scores = CriticScore(
            technical_accuracy=8,
            completeness=9,
            code_quality=8,
            clarity=7,
            voice=8,
            originality=9,
            length=8,
            diagram_quality=10,
        )
        assert scores.technical_accuracy == 8

    def test_score_below_min_fails(self):
        """Score below 1 raises error."""
        with pytest.raises(ValidationError):
            CriticScore(
                technical_accuracy=0,
                completeness=8,
                code_quality=8,
                clarity=8,
                voice=8,
                originality=8,
                length=8,
                diagram_quality=10,
            )

    def test_score_above_max_fails(self):
        """Score above 10 raises error."""
        with pytest.raises(ValidationError):
            CriticScore(
                technical_accuracy=11,
                completeness=8,
                code_quality=8,
                clarity=8,
                voice=8,
                originality=8,
                length=8,
                diagram_quality=10,
            )

    def test_boundary_scores_valid(self):
        """Boundary scores (1 and 10) are valid."""
        scores = CriticScore(
            technical_accuracy=1,
            completeness=10,
            code_quality=5,
            clarity=1,
            voice=10,
            originality=5,
            length=10,
            diagram_quality=10,
        )
        assert scores.technical_accuracy == 1
        assert scores.completeness == 10


class TestSectionCriticResult:
    """Tests for SectionCriticResult model."""

    def test_passing_result(self):
        """Passing critic result."""
        result = SectionCriticResult(
            scores=CriticScore(
                technical_accuracy=9,
                completeness=8,
                code_quality=9,
                clarity=9,
                voice=8,
                originality=9,
                length=8,
                diagram_quality=10,
            ),
            overall_pass=True,
            issues=[],
            fact_check_needed=[],
        )
        assert result.overall_pass is True
        assert result.issues == []

    def test_failing_result_with_issues(self):
        """Failing result with issues."""
        from src.agent.state import CriticIssue

        result = SectionCriticResult(
            scores=CriticScore(
                technical_accuracy=5,
                completeness=6,
                code_quality=6,
                clarity=5,
                voice=6,
                originality=4,
                length=7,
                diagram_quality=10,
            ),
            overall_pass=False,
            issues=[
                CriticIssue(
                    dimension="technical_accuracy",
                    location="paragraph 2",
                    problem="Lacks technical depth",
                    suggestion="Add more technical details",
                ),
                CriticIssue(
                    dimension="originality",
                    location="overall",
                    problem="Missing code examples",
                    suggestion="Add code examples",
                ),
            ],
            fact_check_needed=["Claim about 50% performance improvement"],
        )
        assert result.overall_pass is False
        assert len(result.issues) == 2
        assert len(result.fact_check_needed) == 1


class TestSourceValidation:
    """Tests for SourceValidation model."""

    def test_valid_source(self):
        """Valid source validation."""
        sv = SourceValidation(
            url="https://example.com/article",
            relevant=True,
            quality="high",
            use=True,
            reason="Authoritative source with detailed explanation",
        )
        assert sv.relevant is True
        assert sv.quality == "high"

    def test_rejected_source(self):
        """Rejected source validation."""
        sv = SourceValidation(
            url="https://spam.com/clickbait",
            relevant=False,
            quality="low",
            use=False,
            reason="Low quality content, no technical depth",
        )
        assert sv.use is False


class TestSourceValidationList:
    """Tests for SourceValidationList model."""

    def test_empty_list(self):
        """Empty source list is valid."""
        svl = SourceValidationList(sources=[])
        assert len(svl.sources) == 0

    def test_multiple_sources(self):
        """Multiple sources in list."""
        svl = SourceValidationList(
            sources=[
                SourceValidation(
                    url="https://a.com",
                    relevant=True,
                    quality="high",
                    use=True,
                    reason="Good",
                ),
                SourceValidation(
                    url="https://b.com",
                    relevant=False,
                    quality="low",
                    use=False,
                    reason="Bad",
                ),
            ]
        )
        assert len(svl.sources) == 2


# =============================================================================
# JobManager Tests
# =============================================================================


@pytest.fixture
def temp_base_dir(tmp_path: Path) -> Path:
    """Create a temporary base directory for JobManager."""
    return tmp_path / ".blog_agent"


@pytest.fixture
def manager(temp_base_dir: Path) -> JobManager:
    """Create a JobManager with temp storage."""
    return JobManager(base_dir=temp_base_dir)


class TestJobManagerSlugify:
    """Tests for JobManager.slugify static method."""

    def test_basic_slugify(self):
        """Basic text is slugified."""
        assert JobManager.slugify("Hello World") == "hello-world"

    def test_removes_special_chars(self):
        """Special characters are removed."""
        assert JobManager.slugify("Hello! World?") == "hello-world"

    def test_handles_multiple_spaces(self):
        """Multiple spaces become single hyphen."""
        assert JobManager.slugify("Hello   World") == "hello-world"

    def test_truncates_long_text(self):
        """Long text is truncated."""
        long_text = "a" * 100
        result = JobManager.slugify(long_text, max_length=50)
        assert len(result) <= 50

    def test_removes_trailing_hyphen_after_truncate(self):
        """Trailing hyphen is removed after truncation."""
        result = JobManager.slugify("this-is-a-very-long-title-here", max_length=20)
        assert not result.endswith("-")

    def test_real_title_example(self):
        """Real blog title example."""
        title = "Semantic Caching for LLM Applications!"
        expected = "semantic-caching-for-llm-applications"
        assert JobManager.slugify(title) == expected


class TestJobManagerCreateJob:
    """Tests for JobManager.create_job method."""

    def test_create_job_returns_slugified_id(self, manager: JobManager):
        """Job ID is slugified title."""
        job_id = manager.create_job(
            title="My Test Blog Post",
            context="Some context here",
        )
        assert job_id == "my-test-blog-post"

    def test_create_job_creates_directories(self, manager: JobManager):
        """Correct directory structure is created."""
        job_id = manager.create_job(
            title="Test Blog",
            context="Context",
        )
        job_dir = manager.get_job_dir(job_id)

        assert job_dir.exists()
        assert (job_dir / "research" / "cache").exists()
        assert (job_dir / "drafts" / "sections").exists()
        assert (job_dir / "images").exists()

    def test_create_job_saves_input(self, manager: JobManager):
        """Input data is saved to input.json."""
        job_id = manager.create_job(
            title="Test Blog",
            context="My context",
            target_length="long",
            flags={"review_sections": True},
        )
        job_dir = manager.get_job_dir(job_id)

        input_file = job_dir / "input.json"
        assert input_file.exists()

        with open(input_file) as f:
            data = json.load(f)

        assert data["title"] == "Test Blog"
        assert data["context"] == "My context"
        assert data["target_length"] == "long"
        assert data["flags"]["review_sections"] is True

    def test_create_job_initializes_state(self, manager: JobManager):
        """Initial state is created."""
        job_id = manager.create_job(
            title="Test Blog",
            context="Context",
        )
        job_dir = manager.get_job_dir(job_id)

        state_file = job_dir / "state.json"
        assert state_file.exists()

        with open(state_file) as f:
            data = json.load(f)

        assert data["current_phase"] == "topic_discovery"
        assert data["current_section_index"] == 0
        assert data["can_resume"] is True


class TestJobManagerSaveLoadState:
    """Tests for JobManager.save_state and load_state methods."""

    def test_save_load_roundtrip(self, manager: JobManager):
        """State saves and loads correctly."""
        job_id = manager.create_job(
            title="Test Blog",
            context="Context",
        )

        # Create state with data
        state: BlogAgentState = {
            "job_id": job_id,
            "title": "Test Blog",
            "context": "Context",
            "current_phase": Phase.PLANNING.value,
            "current_section_index": 2,
            "plan": {
                "blog_title": "Test Blog",
                "target_words": 2000,
                "sections": [],
            },
        }

        manager.save_state(job_id, state)
        loaded = manager.load_state(job_id)

        assert loaded is not None
        assert loaded["current_phase"] == "planning"
        assert loaded["current_section_index"] == 2
        assert loaded["plan"]["blog_title"] == "Test Blog"

    def test_save_section_drafts(self, manager: JobManager):
        """Section drafts are saved to files."""
        job_id = manager.create_job(
            title="Test Blog",
            context="Context",
        )

        state: BlogAgentState = {
            "current_phase": Phase.WRITING.value,
            "section_drafts": {
                "intro": "# Introduction\n\nThis is the intro.",
                "conclusion": "# Conclusion\n\nThis is the end.",
            },
        }

        manager.save_state(job_id, state)

        job_dir = manager.get_job_dir(job_id)
        intro_file = job_dir / "drafts" / "sections" / "intro.md"
        assert intro_file.exists()
        assert "Introduction" in intro_file.read_text()

    def test_load_section_drafts(self, manager: JobManager):
        """Section drafts are loaded from files."""
        job_id = manager.create_job(
            title="Test Blog",
            context="Context",
        )

        # Manually create draft file
        job_dir = manager.get_job_dir(job_id)
        drafts_dir = job_dir / "drafts" / "sections"
        (drafts_dir / "intro.md").write_text("# My Intro")

        loaded = manager.load_state(job_id)

        assert loaded is not None
        assert "section_drafts" in loaded
        assert loaded["section_drafts"]["intro"] == "# My Intro"

    def test_save_combined_draft(self, manager: JobManager):
        """Combined draft is saved."""
        job_id = manager.create_job(
            title="Test Blog",
            context="Context",
        )

        state: BlogAgentState = {
            "current_phase": Phase.ASSEMBLING.value,
            "combined_draft": "# Full Blog\n\nContent here.",
        }

        manager.save_state(job_id, state)

        job_dir = manager.get_job_dir(job_id)
        v1_file = job_dir / "drafts" / "v1.md"
        assert v1_file.exists()
        assert "Full Blog" in v1_file.read_text()

    def test_save_final_markdown(self, manager: JobManager):
        """Final markdown is saved."""
        job_id = manager.create_job(
            title="Test Blog",
            context="Context",
        )

        state: BlogAgentState = {
            "current_phase": Phase.DONE.value,
            "final_markdown": "# Final Blog\n\nPolished content.",
        }

        manager.save_state(job_id, state)

        job_dir = manager.get_job_dir(job_id)
        final_file = job_dir / "final.md"
        assert final_file.exists()
        assert "Final Blog" in final_file.read_text()

    def test_load_nonexistent_job(self, manager: JobManager):
        """Loading nonexistent job returns None."""
        result = manager.load_state("nonexistent-job")
        assert result is None

    def test_save_nonexistent_job_raises(self, manager: JobManager):
        """Saving to nonexistent job raises error."""
        with pytest.raises(ValueError, match="Job not found"):
            manager.save_state("nonexistent-job", {})

    def test_save_topic_context(self, manager: JobManager):
        """Topic context is saved."""
        job_id = manager.create_job(
            title="Test Blog",
            context="Context",
        )

        state: BlogAgentState = {
            "current_phase": Phase.TOPIC_DISCOVERY.value,
            "discovery_queries": ["query1", "query2", "query3"],
            "topic_context": [
                {"title": "Article 1", "url": "https://a.com", "snippet": "Text 1"},
                {"title": "Article 2", "url": "https://b.com", "snippet": "Text 2"},
            ],
        }

        manager.save_state(job_id, state)

        job_dir = manager.get_job_dir(job_id)
        tc_file = job_dir / "topic_context.json"
        assert tc_file.exists()

        with open(tc_file) as f:
            data = json.load(f)

        assert len(data["queries_used"]) == 3
        assert len(data["results"]) == 2


class TestJobManagerListJobs:
    """Tests for JobManager.list_jobs method."""

    def test_list_empty(self, manager: JobManager):
        """Empty jobs list."""
        jobs = manager.list_jobs()
        assert jobs == []

    def test_list_all_jobs(self, manager: JobManager):
        """Lists all jobs."""
        manager.create_job(title="Blog One", context="C1")
        manager.create_job(title="Blog Two", context="C2")

        jobs = manager.list_jobs()
        assert len(jobs) == 2

    def test_list_filters_by_complete(self, manager: JobManager):
        """Filters by complete status."""
        job1 = manager.create_job(title="Blog One", context="C1")
        job2 = manager.create_job(title="Blog Two", context="C2")

        # Mark job1 as done
        state: BlogAgentState = {"current_phase": Phase.DONE.value}
        manager.save_state(job1, state)

        complete = manager.list_jobs(status="complete")
        incomplete = manager.list_jobs(status="incomplete")

        assert len(complete) == 1
        assert complete[0]["job_id"] == "blog-one"

        assert len(incomplete) == 1
        assert incomplete[0]["job_id"] == "blog-two"

    def test_list_jobs_sorted_by_last_updated(self, manager: JobManager):
        """Jobs sorted by last_updated descending."""
        job1 = manager.create_job(title="First", context="C1")
        job2 = manager.create_job(title="Second", context="C2")

        # Update job1 to be more recent
        state: BlogAgentState = {"current_phase": Phase.PLANNING.value}
        manager.save_state(job1, state)

        jobs = manager.list_jobs()

        # Most recently updated should be first
        assert jobs[0]["job_id"] == "first"


class TestBlogAgentStateTypedDict:
    """Tests for BlogAgentState TypedDict structure."""

    def test_create_minimal_state(self):
        """Minimal state can be created."""
        state: BlogAgentState = {
            "job_id": "test-job",
            "title": "Test Blog",
        }
        assert state["job_id"] == "test-job"

    def test_create_full_state(self):
        """Full state with all fields."""
        state: BlogAgentState = {
            "job_id": "test-job",
            "title": "Test Blog",
            "context": "Some context",
            "target_length": "medium",
            "flags": {"review_sections": True},
            "current_phase": Phase.WRITING.value,
            "current_section_index": 3,
            "discovery_queries": ["q1", "q2", "q3"],
            "topic_context": [{"title": "A", "url": "http://a.com", "snippet": "S"}],
            "plan": {"blog_title": "Test", "sections": []},
            "research_cache": {},
            "validated_sources": {},
            "section_drafts": {"intro": "# Intro"},
            "section_reviews": {},
            "fact_check_items": [],
            "combined_draft": "# Blog",
            "final_review": {},
            "rendered_diagrams": {},
            "human_review_decision": "approve",
            "human_feedback": "",
            "final_markdown": "# Final",
            "metadata": {"word_count": 2000},
            "error_message": None,
        }
        assert state["current_phase"] == "writing"
        assert state["section_drafts"]["intro"] == "# Intro"

    def test_state_is_json_serializable(self):
        """State can be serialized to JSON."""
        state: BlogAgentState = {
            "job_id": "test",
            "title": "Test",
            "current_phase": Phase.PLANNING.value,
            "plan": {"sections": [{"id": "s1"}]},
        }
        json_str = json.dumps(state)
        loaded = json.loads(json_str)
        assert loaded["job_id"] == "test"
