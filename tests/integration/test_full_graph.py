"""
Integration tests for the full blog agent graph.

These tests run the complete pipeline with real LLM calls.
Run with: PYTHONPATH=. pytest tests/integration/test_full_graph.py -v

WARNING: These tests make real API calls and will consume API quota.
Set environment variable SKIP_EXPENSIVE_TESTS=1 to skip.
"""

import os
import pytest

from src.agent.graph import build_blog_agent_graph
from src.agent.state import JobManager, Phase


# Skip expensive tests if environment variable is set
skip_expensive = pytest.mark.skipif(
    os.getenv("SKIP_EXPENSIVE_TESTS") == "1",
    reason="SKIP_EXPENSIVE_TESTS is set",
)


class TestFullGraphExecution:
    """Integration tests for complete graph execution."""

    @pytest.fixture
    def job_manager(self, tmp_path):
        """Create JobManager with temporary directory."""
        return JobManager(base_dir=tmp_path)

    @pytest.fixture
    def initial_state(self, job_manager):
        """Create initial state for graph execution."""
        job_id = job_manager.create_job(
            title="Redis Semantic Caching",
            context="Using Redis for LLM response caching with vector similarity",
            target_length="short",
        )

        return {
            "job_id": job_id,
            "title": "Redis Semantic Caching",
            "context": "Using Redis for LLM response caching with vector similarity",
            "target_length": "short",
            "current_phase": Phase.TOPIC_DISCOVERY.value,
            "current_section_index": 0,
            "section_drafts": {},
            "flags": {},
        }

    @skip_expensive
    @pytest.mark.asyncio
    async def test_full_pipeline_execution(self, initial_state, job_manager, tmp_path):
        """Run complete pipeline from discovery to assembly."""
        graph = build_blog_agent_graph()

        # Execute the graph
        result = await graph.ainvoke(initial_state)

        # Verify final state
        assert result["current_phase"] == Phase.REVIEWING.value
        assert "final_markdown" in result
        assert len(result["final_markdown"]) > 500  # Should have substantial content
        assert result.get("metadata", {}).get("word_count", 0) > 100

        # Verify files were created
        job_dir = job_manager.get_job_dir(initial_state["job_id"])
        assert (job_dir / "final.md").exists()
        assert (job_dir / "metadata.json").exists()

        # Verify final.md has content
        final_content = (job_dir / "final.md").read_text()
        assert "# Redis Semantic Caching" in final_content or "Redis" in final_content

    @skip_expensive
    @pytest.mark.asyncio
    async def test_pipeline_produces_valid_markdown(self, initial_state, job_manager):
        """Verify produced markdown is valid."""
        graph = build_blog_agent_graph()

        result = await graph.ainvoke(initial_state)

        final_markdown = result.get("final_markdown", "")

        # Check markdown structure
        assert final_markdown.startswith("#")  # Should start with H1
        assert "##" in final_markdown  # Should have H2 sections

        # Check no unresolved placeholders
        assert "{{" not in final_markdown
        assert "}}" not in final_markdown
        assert "[PLACEHOLDER]" not in final_markdown.upper()

    @skip_expensive
    @pytest.mark.asyncio
    async def test_pipeline_handles_short_content(self, job_manager):
        """Test pipeline with minimal context."""
        job_id = job_manager.create_job(
            title="Quick Python Tip",
            context="List comprehensions",
            target_length="short",
        )

        initial_state = {
            "job_id": job_id,
            "title": "Quick Python Tip",
            "context": "List comprehensions",
            "target_length": "short",
            "current_phase": Phase.TOPIC_DISCOVERY.value,
            "current_section_index": 0,
            "section_drafts": {},
        }

        graph = build_blog_agent_graph()
        result = await graph.ainvoke(initial_state)

        assert result["current_phase"] == Phase.REVIEWING.value
        assert result.get("metadata", {}).get("word_count", 0) > 50


class TestGraphPhaseTransitions:
    """Tests for phase transitions during graph execution."""

    @pytest.fixture
    def job_manager(self, tmp_path):
        return JobManager(base_dir=tmp_path)

    @skip_expensive
    @pytest.mark.asyncio
    async def test_discovery_to_planning_transition(self, job_manager):
        """Test discovery phase transitions to planning."""
        from src.agent.nodes import topic_discovery_node

        job_id = job_manager.create_job("Test Topic", "Test context")

        state = {
            "job_id": job_id,
            "title": "Test Topic",
            "context": "Test context",
            "current_phase": Phase.TOPIC_DISCOVERY.value,
        }

        result = await topic_discovery_node(state)

        assert result["current_phase"] == Phase.PLANNING.value
        assert "discovery_queries" in result
        assert "topic_context" in result
        assert len(result["discovery_queries"]) >= 3

    @skip_expensive
    @pytest.mark.asyncio
    async def test_planning_produces_valid_plan(self, job_manager):
        """Test planning phase produces valid plan structure."""
        from src.agent.nodes import planning_node

        job_id = job_manager.create_job("Test API Caching", "Redis, performance")

        state = {
            "job_id": job_id,
            "title": "Test API Caching",
            "context": "Redis, performance",
            "target_length": "short",
            "topic_context": [
                {"title": "Redis Caching", "url": "https://redis.io", "snippet": "Fast caching"}
            ],
            "current_phase": Phase.PLANNING.value,
        }

        result = await planning_node(state)

        assert result["current_phase"] == Phase.RESEARCHING.value
        assert "plan" in result

        plan = result["plan"]
        assert "sections" in plan
        assert len(plan["sections"]) >= 3

        # Check section structure
        for section in plan["sections"]:
            assert "id" in section
            assert "role" in section


class TestGraphErrorHandling:
    """Tests for error handling in graph execution."""

    @pytest.fixture
    def job_manager(self, tmp_path):
        return JobManager(base_dir=tmp_path)

    @pytest.mark.asyncio
    async def test_missing_title_fails_gracefully(self, job_manager):
        """Graph handles missing title gracefully."""
        from src.agent.nodes import topic_discovery_node

        state = {
            "job_id": "test",
            "title": "",  # Empty title
            "context": "Some context",
        }

        result = await topic_discovery_node(state)

        assert result["current_phase"] == Phase.FAILED.value
        assert "error_message" in result

    @pytest.mark.asyncio
    async def test_empty_plan_handled(self, job_manager):
        """Write section handles empty plan gracefully."""
        from src.agent.nodes import write_section_node

        state = {
            "job_id": "test",
            "plan": {"sections": []},
            "current_section_index": 0,
            "section_drafts": {},
        }

        result = await write_section_node(state)

        # Should transition to assembly (nothing to write)
        assert result["current_phase"] == Phase.ASSEMBLING.value

    @pytest.mark.asyncio
    async def test_empty_drafts_assembly_fails(self):
        """Assembly fails gracefully with no drafts."""
        from src.agent.nodes import final_assembly_node

        state = {
            "job_id": "",
            "plan": {"sections": []},
            "section_drafts": {},
        }

        result = await final_assembly_node(state)

        assert result["current_phase"] == Phase.FAILED.value
        assert "error_message" in result


class TestGraphSectionLoop:
    """Tests for section writing loop behavior."""

    @skip_expensive
    @pytest.mark.asyncio
    async def test_section_loop_writes_all_required(self, tmp_path):
        """Section loop writes all required sections."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from src.agent.nodes import write_section_node
        from src.agent.state import JobManager

        job_manager = JobManager(base_dir=tmp_path)
        job_id = job_manager.create_job("Test", "Context")

        plan = {
            "blog_title": "Test Blog",
            "sections": [
                {"id": "hook", "role": "hook", "target_words": 50, "optional": False},
                {"id": "problem", "title": "Problem", "role": "problem", "target_words": 100, "optional": False},
                {"id": "optional_dive", "title": "Deep Dive", "role": "deep_dive", "optional": True},
                {"id": "conclusion", "title": "Conclusion", "role": "conclusion", "target_words": 50, "optional": False},
            ],
        }

        state = {
            "job_id": job_id,
            "title": "Test Blog",
            "plan": plan,
            "validated_sources": {"hook": [], "problem": [], "conclusion": []},
            "current_section_index": 0,
            "section_drafts": {},
        }

        # Mock the LLM call
        mock_response = MagicMock()
        mock_response.content = "Generated section content here."

        with patch("src.agent.nodes.KeyManager") as MockKeyManager, \
             patch("src.agent.nodes.ChatGoogleGenerativeAI") as MockLLM:

            mock_km = MagicMock()
            mock_km.get_best_key.return_value = "test-key"
            MockKeyManager.from_env.return_value = mock_km
            MockLLM.return_value.invoke.return_value = mock_response

            # Write first section
            result1 = await write_section_node(state)
            assert result1["current_section_index"] == 1
            assert "hook" in result1["section_drafts"]

            # Write second section
            state.update(result1)
            result2 = await write_section_node(state)
            assert result2["current_section_index"] == 2
            assert "problem" in result2["section_drafts"]

            # Write third section (skips optional, goes to conclusion)
            state.update(result2)
            result3 = await write_section_node(state)
            assert result3["current_section_index"] == 3
            assert "conclusion" in result3["section_drafts"]

            # Fourth call should transition to assembly
            state.update(result3)
            result4 = await write_section_node(state)
            assert result4["current_phase"] == Phase.ASSEMBLING.value
