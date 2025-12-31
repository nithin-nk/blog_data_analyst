"""
Unit tests for the CLI module.

Tests cover:
- Command argument validation
- Help text
- Error handling for invalid inputs
"""

import pytest
from click.testing import CliRunner
from unittest.mock import MagicMock, patch

from src.agent.__main__ import cli, start, resume, jobs, show


@pytest.fixture
def runner():
    """Create CLI test runner."""
    return CliRunner()


class TestCliGroup:
    """Tests for the main CLI group."""

    def test_help_shows_commands(self, runner):
        """Help text shows all available commands."""
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "start" in result.output
        assert "resume" in result.output
        assert "jobs" in result.output
        assert "show" in result.output

    def test_no_command_shows_usage(self, runner):
        """Running without command shows usage info."""
        result = runner.invoke(cli)
        # Click returns exit code 0 or 2 depending on version when no command given
        assert "Usage:" in result.output


class TestStartCommand:
    """Tests for the start command."""

    def test_start_help(self, runner):
        """Start command shows help."""
        result = runner.invoke(cli, ["start", "--help"])
        assert result.exit_code == 0
        assert "--title" in result.output
        assert "--context" in result.output
        assert "--length" in result.output

    def test_start_requires_title(self, runner):
        """Start command requires --title option."""
        result = runner.invoke(cli, ["start", "--context", "test"])
        assert result.exit_code != 0
        assert "Missing option" in result.output or "required" in result.output.lower()

    def test_start_requires_context(self, runner):
        """Start command requires --context option."""
        result = runner.invoke(cli, ["start", "--title", "Test"])
        assert result.exit_code != 0
        assert "Missing option" in result.output or "required" in result.output.lower()

    def test_start_length_choices(self, runner):
        """Start command length option has valid choices."""
        result = runner.invoke(cli, ["start", "--title", "Test", "--context", "Test", "--length", "invalid"])
        assert result.exit_code != 0
        assert "Invalid value" in result.output or "invalid choice" in result.output.lower()

    @patch("src.agent.__main__._run_start")
    def test_start_calls_run_start(self, mock_run_start, runner):
        """Start command calls _run_start with correct args."""
        mock_run_start.return_value = None

        result = runner.invoke(cli, [
            "start",
            "--title", "Test Title",
            "--context", "Test context",
            "--length", "short",
        ])

        # Note: asyncio.run wraps _run_start, so we check it was called
        mock_run_start.assert_called_once_with("Test Title", "Test context", "short", False)

    @patch("src.agent.__main__._run_start")
    def test_start_default_length_medium(self, mock_run_start, runner):
        """Start command defaults to medium length."""
        mock_run_start.return_value = None

        runner.invoke(cli, [
            "start",
            "--title", "Test",
            "--context", "Test",
        ])

        mock_run_start.assert_called_once_with("Test", "Test", "medium", False)


class TestResumeCommand:
    """Tests for the resume command."""

    def test_resume_help(self, runner):
        """Resume command shows help."""
        result = runner.invoke(cli, ["resume", "--help"])
        assert result.exit_code == 0
        assert "JOB_ID" in result.output

    def test_resume_requires_job_id(self, runner):
        """Resume command requires job_id argument."""
        result = runner.invoke(cli, ["resume"])
        assert result.exit_code != 0
        assert "Missing argument" in result.output or "JOB_ID" in result.output

    @patch("src.agent.__main__._run_resume")
    def test_resume_calls_run_resume(self, mock_run_resume, runner):
        """Resume command calls _run_resume with job_id."""
        mock_run_resume.return_value = None

        runner.invoke(cli, ["resume", "test-job-id"])

        mock_run_resume.assert_called_once_with("test-job-id")


class TestJobsCommand:
    """Tests for the jobs command."""

    def test_jobs_help(self, runner):
        """Jobs command shows help."""
        result = runner.invoke(cli, ["jobs", "--help"])
        assert result.exit_code == 0
        assert "--status" in result.output

    def test_jobs_status_choices(self, runner):
        """Jobs command status option has valid choices."""
        result = runner.invoke(cli, ["jobs", "--status", "invalid"])
        assert result.exit_code != 0

    @patch("src.agent.__main__.JobManager")
    def test_jobs_lists_all(self, MockJobManager, runner):
        """Jobs command lists all jobs."""
        mock_jm = MagicMock()
        mock_jm.list_jobs.return_value = [
            {"job_id": "test-1", "title": "Test 1", "phase": "done", "complete": True},
            {"job_id": "test-2", "title": "Test 2", "phase": "writing", "complete": False},
        ]
        MockJobManager.return_value = mock_jm

        result = runner.invoke(cli, ["jobs"])

        assert result.exit_code == 0
        assert "test-1" in result.output
        assert "test-2" in result.output
        mock_jm.list_jobs.assert_called_once_with(None)

    @patch("src.agent.__main__.JobManager")
    def test_jobs_filters_by_status(self, MockJobManager, runner):
        """Jobs command filters by status."""
        mock_jm = MagicMock()
        mock_jm.list_jobs.return_value = []
        MockJobManager.return_value = mock_jm

        runner.invoke(cli, ["jobs", "--status", "complete"])

        mock_jm.list_jobs.assert_called_once_with("complete")

    @patch("src.agent.__main__.JobManager")
    def test_jobs_shows_no_jobs_message(self, MockJobManager, runner):
        """Jobs command shows message when no jobs found."""
        mock_jm = MagicMock()
        mock_jm.list_jobs.return_value = []
        MockJobManager.return_value = mock_jm

        result = runner.invoke(cli, ["jobs"])

        assert "No jobs found" in result.output


class TestShowCommand:
    """Tests for the show command."""

    def test_show_help(self, runner):
        """Show command shows help."""
        result = runner.invoke(cli, ["show", "--help"])
        assert result.exit_code == 0
        assert "JOB_ID" in result.output

    def test_show_requires_job_id(self, runner):
        """Show command requires job_id argument."""
        result = runner.invoke(cli, ["show"])
        assert result.exit_code != 0

    @patch("src.agent.__main__.JobManager")
    def test_show_displays_job_details(self, MockJobManager, runner):
        """Show command displays job details."""
        mock_jm = MagicMock()
        mock_jm.load_state.return_value = {
            "title": "Test Blog",
            "current_phase": "writing",
            "target_length": "medium",
            "plan": {"sections": [{"id": "hook"}, {"id": "problem"}]},
            "section_drafts": {"hook": "content"},
        }
        mock_jm.get_job_dir.return_value = MagicMock()
        mock_jm.get_job_dir.return_value.__truediv__ = lambda self, x: MagicMock(exists=lambda: False)
        MockJobManager.return_value = mock_jm

        result = runner.invoke(cli, ["show", "test-job"])

        assert result.exit_code == 0
        assert "Test Blog" in result.output
        assert "writing" in result.output
        assert "medium" in result.output

    @patch("src.agent.__main__.JobManager")
    def test_show_handles_not_found(self, MockJobManager, runner):
        """Show command handles job not found."""
        mock_jm = MagicMock()
        mock_jm.load_state.return_value = None
        MockJobManager.return_value = mock_jm

        result = runner.invoke(cli, ["show", "nonexistent"])

        assert result.exit_code != 0
        assert "not found" in result.output.lower()


def create_mock_astream(final_state):
    """Create a mock astream that yields the final state."""
    async def mock_astream(_state):
        yield {"final_node": final_state}
    return mock_astream


class TestRunStart:
    """Tests for _run_start async function."""

    @pytest.mark.asyncio
    async def test_run_start_creates_job(self, tmp_path):
        """_run_start creates a new job."""
        from src.agent.__main__ import _run_start
        from src.agent.state import Phase

        with patch("src.agent.__main__.JobManager") as MockJM, \
             patch("src.agent.__main__.build_blog_agent_graph") as MockGraph:

            mock_jm = MagicMock()
            mock_jm.create_job.return_value = "test-job-id"
            mock_jm.get_job_dir.return_value = tmp_path
            mock_jm.load_state.return_value = None  # Force new job creation (not resume)
            MockJM.return_value = mock_jm

            mock_graph = MagicMock()
            mock_graph.astream = create_mock_astream({
                "current_phase": Phase.REVIEWING.value,
                "metadata": {"word_count": 500, "reading_time_minutes": 2, "section_count": 3},
            })
            MockGraph.return_value = mock_graph

            # Create mock final.md
            (tmp_path / "final.md").write_text("test")

            await _run_start("Test Title", "Test context", "short", False)

            mock_jm.create_job.assert_called_once_with("Test Title", "Test context", "short")


class TestRunResume:
    """Tests for _run_resume async function."""

    @pytest.mark.asyncio
    async def test_run_resume_loads_state(self, tmp_path):
        """_run_resume loads existing job state."""
        from src.agent.__main__ import _run_resume
        from src.agent.state import Phase

        with patch("src.agent.__main__.JobManager") as MockJM, \
             patch("src.agent.__main__.build_blog_agent_graph") as MockGraph:

            mock_jm = MagicMock()
            mock_jm.load_state.return_value = {
                "job_id": "test-job",
                "title": "Test",
                "current_phase": Phase.WRITING.value,
                "current_section_index": 1,
                "section_drafts": {},
            }
            mock_jm.get_job_dir.return_value = tmp_path
            MockJM.return_value = mock_jm

            mock_graph = MagicMock()
            mock_graph.astream = create_mock_astream({
                "current_phase": Phase.REVIEWING.value,
            })
            MockGraph.return_value = mock_graph

            await _run_resume("test-job")

            mock_jm.load_state.assert_called_once_with("test-job")
