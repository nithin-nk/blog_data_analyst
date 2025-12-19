"""
Integration tests for the CLI module.

These tests verify CLI commands with real (or mocked) graph execution.
Run with: PYTHONPATH=. pytest tests/integration/test_cli_integration.py -v

WARNING: Some tests make real API calls and will consume API quota.
Set environment variable SKIP_EXPENSIVE_TESTS=1 to skip.
"""

import os
import pytest
from click.testing import CliRunner
from unittest.mock import AsyncMock, MagicMock, patch

from src.agent.__main__ import cli
from src.agent.state import Phase


# Skip expensive tests if environment variable is set
skip_expensive = pytest.mark.skipif(
    os.getenv("SKIP_EXPENSIVE_TESTS") == "1",
    reason="SKIP_EXPENSIVE_TESTS is set",
)


@pytest.fixture
def runner():
    """Create CLI test runner."""
    return CliRunner()


class TestStartIntegration:
    """Integration tests for the start command."""

    @pytest.fixture
    def mock_graph(self):
        """Create mock graph that simulates successful completion."""
        mock = MagicMock()
        mock.ainvoke = AsyncMock(return_value={
            "current_phase": Phase.REVIEWING.value,
            "final_markdown": "# Test Blog\n\nContent here.",
            "metadata": {
                "word_count": 500,
                "reading_time_minutes": 2,
                "section_count": 3,
            },
        })
        return mock

    def test_start_with_mocked_graph(self, runner, tmp_path, mock_graph):
        """Start command runs successfully with mocked graph."""
        with patch("src.agent.__main__.JobManager") as MockJM, \
             patch("src.agent.__main__.build_blog_agent_graph", return_value=mock_graph):

            mock_jm = MagicMock()
            mock_jm.create_job.return_value = "test-blog"
            mock_jm.get_job_dir.return_value = tmp_path

            # Create the final.md file
            (tmp_path / "final.md").write_text("# Test\n\nContent")

            MockJM.return_value = mock_jm

            result = runner.invoke(cli, [
                "start",
                "--title", "Test Blog",
                "--context", "Test context",
                "--length", "short",
            ])

            assert result.exit_code == 0
            assert "successfully" in result.output.lower() or "generated" in result.output.lower()

    def test_start_shows_stats_on_success(self, runner, tmp_path, mock_graph):
        """Start command displays stats after successful generation."""
        with patch("src.agent.__main__.JobManager") as MockJM, \
             patch("src.agent.__main__.build_blog_agent_graph", return_value=mock_graph):

            mock_jm = MagicMock()
            mock_jm.create_job.return_value = "test-blog"
            mock_jm.get_job_dir.return_value = tmp_path
            (tmp_path / "final.md").write_text("content")
            MockJM.return_value = mock_jm

            result = runner.invoke(cli, [
                "start",
                "--title", "Test",
                "--context", "Test",
            ])

            assert "500" in result.output  # Word count
            assert "2 min" in result.output  # Reading time

    def test_start_handles_failure(self, runner, tmp_path):
        """Start command handles graph failure gracefully."""
        mock_graph = MagicMock()
        mock_graph.ainvoke = AsyncMock(return_value={
            "current_phase": Phase.FAILED.value,
            "error_message": "Test error",
        })

        with patch("src.agent.__main__.JobManager") as MockJM, \
             patch("src.agent.__main__.build_blog_agent_graph", return_value=mock_graph):

            mock_jm = MagicMock()
            mock_jm.create_job.return_value = "test-blog"
            mock_jm.get_job_dir.return_value = tmp_path
            MockJM.return_value = mock_jm

            result = runner.invoke(cli, [
                "start",
                "--title", "Test",
                "--context", "Test",
            ])

            assert result.exit_code != 0
            assert "failed" in result.output.lower()


class TestResumeIntegration:
    """Integration tests for the resume command."""

    def test_resume_not_found(self, runner):
        """Resume handles non-existent job."""
        with patch("src.agent.__main__.JobManager") as MockJM:
            mock_jm = MagicMock()
            mock_jm.load_state.return_value = None
            MockJM.return_value = mock_jm

            result = runner.invoke(cli, ["resume", "nonexistent-job"])

            assert result.exit_code != 0
            assert "not found" in result.output.lower()

    def test_resume_completed_job(self, runner, tmp_path):
        """Resume handles already completed job."""
        with patch("src.agent.__main__.JobManager") as MockJM:
            mock_jm = MagicMock()
            mock_jm.load_state.return_value = {
                "job_id": "test-job",
                "current_phase": Phase.DONE.value,
            }
            mock_jm.get_job_dir.return_value = tmp_path
            MockJM.return_value = mock_jm

            result = runner.invoke(cli, ["resume", "test-job"])

            assert result.exit_code == 0
            assert "complete" in result.output.lower()

    def test_resume_continues_from_writing(self, runner, tmp_path):
        """Resume continues from writing phase."""
        mock_graph = MagicMock()
        mock_graph.ainvoke = AsyncMock(return_value={
            "current_phase": Phase.REVIEWING.value,
        })

        with patch("src.agent.__main__.JobManager") as MockJM, \
             patch("src.agent.__main__.build_blog_agent_graph", return_value=mock_graph):

            mock_jm = MagicMock()
            mock_jm.load_state.return_value = {
                "job_id": "test-job",
                "title": "Test",
                "current_phase": Phase.WRITING.value,
                "current_section_index": 1,
                "section_drafts": {"hook": "content"},
            }
            mock_jm.get_job_dir.return_value = tmp_path
            (tmp_path / "final.md").write_text("content")
            MockJM.return_value = mock_jm

            result = runner.invoke(cli, ["resume", "test-job"])

            assert result.exit_code == 0
            mock_graph.ainvoke.assert_called_once()


class TestJobsIntegration:
    """Integration tests for the jobs command."""

    def test_jobs_with_real_job_manager(self, runner, tmp_path):
        """Jobs command works with real JobManager."""
        from src.agent.state import JobManager

        job_manager = JobManager(base_dir=tmp_path)

        # Create some test jobs
        job_manager.create_job("Test Blog 1", "Context 1")
        job_manager.create_job("Test Blog 2", "Context 2")

        with patch("src.agent.__main__.JobManager", return_value=job_manager):
            result = runner.invoke(cli, ["jobs"])

            assert result.exit_code == 0
            assert "test-blog-1" in result.output
            assert "test-blog-2" in result.output

    def test_jobs_filter_complete(self, runner, tmp_path):
        """Jobs command filters by complete status."""
        from src.agent.state import JobManager

        job_manager = JobManager(base_dir=tmp_path)
        job_id = job_manager.create_job("Test Blog", "Context")

        # Mark as complete
        job_manager.save_state(job_id, {"current_phase": Phase.DONE.value})

        with patch("src.agent.__main__.JobManager", return_value=job_manager):
            result = runner.invoke(cli, ["jobs", "--status", "complete"])

            assert result.exit_code == 0
            assert "test-blog" in result.output


class TestShowIntegration:
    """Integration tests for the show command."""

    def test_show_with_real_job_manager(self, runner, tmp_path):
        """Show command works with real JobManager."""
        from src.agent.state import JobManager

        job_manager = JobManager(base_dir=tmp_path)
        job_id = job_manager.create_job("Test Blog Title", "Test context here")

        # Add some state
        job_manager.save_state(job_id, {
            "current_phase": Phase.WRITING.value,
            "plan": {"sections": [{"id": "hook"}, {"id": "problem"}]},
            "section_drafts": {"hook": "hook content"},
        })

        with patch("src.agent.__main__.JobManager", return_value=job_manager):
            result = runner.invoke(cli, ["show", job_id])

            assert result.exit_code == 0
            assert "Test Blog Title" in result.output
            assert "writing" in result.output.lower()


class TestEndToEnd:
    """End-to-end tests with real graph execution."""

    @skip_expensive
    @pytest.mark.asyncio
    async def test_full_cli_run(self, runner, tmp_path):
        """Full CLI run from start to completion."""
        from src.agent.state import JobManager

        # This test uses real LLM calls
        job_manager = JobManager(base_dir=tmp_path)

        with patch("src.agent.__main__.JobManager", return_value=job_manager):
            result = runner.invoke(cli, [
                "start",
                "--title", "Quick Redis Tip",
                "--context", "Caching basics",
                "--length", "short",
            ])

            if result.exit_code != 0:
                print(f"Output: {result.output}")
                print(f"Exception: {result.exception}")

            assert result.exit_code == 0
            assert "successfully" in result.output.lower() or "generated" in result.output.lower()

            # Verify job was created
            jobs = job_manager.list_jobs()
            assert len(jobs) == 1

            # Verify final.md was created
            job_dir = job_manager.get_job_dir(jobs[0]["job_id"])
            assert (job_dir / "final.md").exists()
