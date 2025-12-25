"""Integration tests for section selection flow."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.agent.nodes import section_selection_node
from src.agent.state import Phase


@pytest.mark.asyncio
async def test_section_selection_with_user_input(tmp_path):
    """Test section selection with mocked user input selecting specific sections."""
    # Setup state with plan containing required and optional sections
    state = {
        "job_id": "test-job",
        "plan": {
            "sections": [
                {
                    "id": "hook",
                    "title": "Hook",
                    "optional": False,
                    "role": "hook",
                    "target_words": 100,
                },
                {
                    "id": "problem",
                    "title": "Problem",
                    "optional": False,
                    "role": "problem",
                    "target_words": 200,
                },
                {
                    "id": "opt1",
                    "title": "Optional Deep Dive",
                    "optional": True,
                    "role": "deep_dive",
                    "target_words": 300,
                    "gap_addressed": "Advanced patterns",
                    "gap_justification": "Covers advanced use cases and patterns for production",
                },
                {
                    "id": "opt2",
                    "title": "Optional Comparison",
                    "optional": True,
                    "role": "deep_dive",
                    "target_words": 300,
                    "gap_addressed": "Alternatives",
                    "gap_justification": "Compares with alternative approaches",
                },
                {
                    "id": "conclusion",
                    "title": "Conclusion",
                    "optional": False,
                    "role": "conclusion",
                    "target_words": 100,
                },
            ]
        },
        "metrics": [],
    }

    # Mock JobManager to avoid file I/O
    with patch("src.agent.nodes.JobManager") as mock_job_manager:
        mock_instance = MagicMock()
        mock_instance.get_job_dir.return_value = tmp_path
        mock_job_manager.return_value = mock_instance

        # Mock user input: select option 1 (first optional section)
        with patch("rich.prompt.Prompt.ask", return_value="1"), patch(
            "rich.prompt.Confirm.ask", return_value=True
        ):
            result = await section_selection_node(state)

    # Verify results
    assert result["current_phase"] == Phase.RESEARCHING.value
    assert len(result["selected_section_ids"]) == 4  # 3 required + 1 optional
    assert "hook" in result["selected_section_ids"]
    assert "problem" in result["selected_section_ids"]
    assert "conclusion" in result["selected_section_ids"]
    assert "opt1" in result["selected_section_ids"]
    assert "opt2" not in result["selected_section_ids"]
    assert result["section_selection_skipped"] is False

    # Verify checkpoint was saved
    checkpoint_file = tmp_path / "selected_sections.json"
    assert checkpoint_file.exists()
    with open(checkpoint_file) as f:
        checkpoint_data = json.load(f)
        assert len(checkpoint_data["selected_section_ids"]) == 4
        assert checkpoint_data["selection_skipped"] is False


@pytest.mark.asyncio
async def test_section_selection_skip_all(tmp_path):
    """Test section selection when user chooses to skip (include all)."""
    state = {
        "job_id": "test-job",
        "plan": {
            "sections": [
                {"id": "a", "title": "A", "optional": False, "role": "hook", "target_words": 100},
                {"id": "b", "title": "B", "optional": True, "role": "deep_dive", "target_words": 200, "gap_justification": "Extra detail"},
                {"id": "c", "title": "C", "optional": True, "role": "deep_dive", "target_words": 200, "gap_justification": "More context"},
            ]
        },
        "metrics": [],
    }

    with patch("src.agent.nodes.JobManager") as mock_job_manager:
        mock_instance = MagicMock()
        mock_instance.get_job_dir.return_value = tmp_path
        mock_job_manager.return_value = mock_instance

        # Mock user input: press enter (skip), then confirm
        with patch("rich.prompt.Prompt.ask", return_value=""), patch(
            "rich.prompt.Confirm.ask", return_value=True
        ):
            result = await section_selection_node(state)

    # Verify all sections are selected
    assert result["section_selection_skipped"] is True
    assert len(result["selected_section_ids"]) == 3
    assert set(result["selected_section_ids"]) == {"a", "b", "c"}


@pytest.mark.asyncio
async def test_section_selection_all_keyword(tmp_path):
    """Test section selection when user types 'all'."""
    state = {
        "job_id": "test-job",
        "plan": {
            "sections": [
                {"id": "a", "title": "A", "optional": False, "role": "hook", "target_words": 100},
                {"id": "b", "title": "B", "optional": True, "role": "deep_dive", "target_words": 200, "gap_justification": "Detail"},
            ]
        },
        "metrics": [],
    }

    with patch("src.agent.nodes.JobManager") as mock_job_manager:
        mock_instance = MagicMock()
        mock_instance.get_job_dir.return_value = tmp_path
        mock_job_manager.return_value = mock_instance

        # Mock user input: type "all"
        with patch("rich.prompt.Prompt.ask", return_value="all"):
            result = await section_selection_node(state)

    # Verify all sections are selected
    assert result["section_selection_skipped"] is True
    assert len(result["selected_section_ids"]) == 2


@pytest.mark.asyncio
async def test_section_selection_no_optional_sections(tmp_path):
    """Test section selection when there are no optional sections."""
    state = {
        "job_id": "test-job",
        "plan": {
            "sections": [
                {"id": "hook", "title": "Hook", "optional": False, "role": "hook", "target_words": 100},
                {"id": "problem", "title": "Problem", "optional": False, "role": "problem", "target_words": 200},
            ]
        },
        "metrics": [],
    }

    with patch("src.agent.nodes.JobManager") as mock_job_manager:
        mock_instance = MagicMock()
        mock_instance.get_job_dir.return_value = tmp_path
        mock_job_manager.return_value = mock_instance

        result = await section_selection_node(state)

    # Should auto-proceed with all required sections
    assert result["section_selection_skipped"] is True
    assert len(result["selected_section_ids"]) == 2
    assert set(result["selected_section_ids"]) == {"hook", "problem"}
    assert result["current_phase"] == Phase.RESEARCHING.value


@pytest.mark.asyncio
async def test_section_selection_multiple_selections(tmp_path):
    """Test selecting multiple optional sections."""
    state = {
        "job_id": "test-job",
        "plan": {
            "sections": [
                {"id": "req1", "title": "Required 1", "optional": False, "role": "hook", "target_words": 100},
                {"id": "opt1", "title": "Optional 1", "optional": True, "role": "deep_dive", "target_words": 200, "gap_justification": "A"},
                {"id": "opt2", "title": "Optional 2", "optional": True, "role": "deep_dive", "target_words": 200, "gap_justification": "B"},
                {"id": "opt3", "title": "Optional 3", "optional": True, "role": "deep_dive", "target_words": 200, "gap_justification": "C"},
                {"id": "req2", "title": "Required 2", "optional": False, "role": "conclusion", "target_words": 100},
            ]
        },
        "metrics": [],
    }

    with patch("src.agent.nodes.JobManager") as mock_job_manager:
        mock_instance = MagicMock()
        mock_instance.get_job_dir.return_value = tmp_path
        mock_job_manager.return_value = mock_instance

        # Mock user input: select options 1 and 3
        with patch("rich.prompt.Prompt.ask", return_value="1,3"), patch(
            "rich.prompt.Confirm.ask", return_value=True
        ):
            result = await section_selection_node(state)

    # Verify selected sections (2 required + 2 optional)
    assert len(result["selected_section_ids"]) == 4
    assert "req1" in result["selected_section_ids"]
    assert "req2" in result["selected_section_ids"]
    assert "opt1" in result["selected_section_ids"]
    assert "opt3" in result["selected_section_ids"]
    assert "opt2" not in result["selected_section_ids"]


@pytest.mark.asyncio
async def test_section_selection_range_input(tmp_path):
    """Test selecting sections using range notation."""
    state = {
        "job_id": "test-job",
        "plan": {
            "sections": [
                {"id": "req", "title": "Required", "optional": False, "role": "hook", "target_words": 100},
                {"id": "opt1", "title": "Opt 1", "optional": True, "role": "deep_dive", "target_words": 200, "gap_justification": "A"},
                {"id": "opt2", "title": "Opt 2", "optional": True, "role": "deep_dive", "target_words": 200, "gap_justification": "B"},
                {"id": "opt3", "title": "Opt 3", "optional": True, "role": "deep_dive", "target_words": 200, "gap_justification": "C"},
            ]
        },
        "metrics": [],
    }

    with patch("src.agent.nodes.JobManager") as mock_job_manager:
        mock_instance = MagicMock()
        mock_instance.get_job_dir.return_value = tmp_path
        mock_job_manager.return_value = mock_instance

        # Mock user input: select range 1-2
        with patch("rich.prompt.Prompt.ask", return_value="1-2"), patch(
            "rich.prompt.Confirm.ask", return_value=True
        ):
            result = await section_selection_node(state)

    # Verify selected sections (1 required + opt1 and opt2)
    assert len(result["selected_section_ids"]) == 3
    assert "req" in result["selected_section_ids"]
    assert "opt1" in result["selected_section_ids"]
    assert "opt2" in result["selected_section_ids"]
    assert "opt3" not in result["selected_section_ids"]


@pytest.mark.asyncio
async def test_section_selection_invalid_then_valid_input(tmp_path):
    """Test recovery from invalid input."""
    state = {
        "job_id": "test-job",
        "plan": {
            "sections": [
                {"id": "req", "title": "Required", "optional": False, "role": "hook", "target_words": 100},
                {"id": "opt1", "title": "Opt 1", "optional": True, "role": "deep_dive", "target_words": 200, "gap_justification": "A"},
            ]
        },
        "metrics": [],
    }

    with patch("src.agent.nodes.JobManager") as mock_job_manager:
        mock_instance = MagicMock()
        mock_instance.get_job_dir.return_value = tmp_path
        mock_job_manager.return_value = mock_instance

        # Mock user input: first invalid (out of range), then valid
        with patch("rich.prompt.Prompt.ask", side_effect=["99", "1"]), patch(
            "rich.prompt.Confirm.ask", return_value=True
        ):
            result = await section_selection_node(state)

    # Should eventually succeed with valid input
    assert len(result["selected_section_ids"]) == 2
    assert "req" in result["selected_section_ids"]
    assert "opt1" in result["selected_section_ids"]


@pytest.mark.asyncio
async def test_section_selection_groups_required_first(tmp_path):
    """Test that selected sections group required first, then optional."""
    state = {
        "job_id": "test-job",
        "plan": {
            "sections": [
                {"id": "a", "title": "A", "optional": False, "role": "hook", "target_words": 100},
                {"id": "b", "title": "B", "optional": True, "role": "deep_dive", "target_words": 200, "gap_justification": "B"},
                {"id": "c", "title": "C", "optional": False, "role": "problem", "target_words": 100},
                {"id": "d", "title": "D", "optional": True, "role": "deep_dive", "target_words": 200, "gap_justification": "D"},
                {"id": "e", "title": "E", "optional": False, "role": "conclusion", "target_words": 100},
            ]
        },
        "metrics": [],
    }

    with patch("src.agent.nodes.JobManager") as mock_job_manager:
        mock_instance = MagicMock()
        mock_instance.get_job_dir.return_value = tmp_path
        mock_job_manager.return_value = mock_instance

        # Select optional sections in reverse order (2, 1)
        with patch("rich.prompt.Prompt.ask", return_value="2,1"), patch(
            "rich.prompt.Confirm.ask", return_value=True
        ):
            result = await section_selection_node(state)

    # Verify required sections come first, then optional (in plan order)
    selected = result["selected_section_ids"]
    assert selected == ["a", "c", "e", "b", "d"]  # Required first, then optional


@pytest.mark.asyncio
async def test_section_selection_skips_when_phase_past(tmp_path):
    """Test that section_selection_node skips execution when resuming from a later phase."""
    state = {
        "job_id": "test-job",
        "current_phase": Phase.VALIDATING_SOURCES.value,  # Phase is past section selection
        "plan": {
            "sections": [
                {"id": "a", "title": "A", "optional": False, "role": "hook", "target_words": 100},
                {"id": "b", "title": "B", "optional": True, "role": "deep_dive", "target_words": 200, "gap_justification": "B"},
            ]
        },
        "current_section_index": 0,
        "metrics": [],
    }

    with patch("src.agent.nodes.JobManager") as mock_job_manager:
        mock_instance = MagicMock()
        mock_instance.get_job_dir.return_value = tmp_path
        mock_job_manager.return_value = mock_instance

        # Should NOT prompt for user input - just skip
        result = await section_selection_node(state)

    # Should return minimal state update without changing phase or selection
    assert "selected_section_ids" not in result or result.get("selected_section_ids") is None
    assert result.get("current_phase") != Phase.RESEARCHING.value  # Should not advance phase
