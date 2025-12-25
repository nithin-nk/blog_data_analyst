"""Unit tests for section selection logic."""

import pytest

from src.agent.nodes import _parse_selection_input


class TestParseSelectionInput:
    """Tests for the selection input parser."""

    def test_single_numbers(self):
        """Test parsing comma-separated single numbers."""
        assert _parse_selection_input("1,3,5", 10) == [1, 3, 5]
        assert _parse_selection_input("2,4,6,8", 10) == [2, 4, 6, 8]
        assert _parse_selection_input("1", 10) == [1]

    def test_ranges(self):
        """Test parsing ranges."""
        assert _parse_selection_input("1-4", 10) == [1, 2, 3, 4]
        assert _parse_selection_input("5-7", 10) == [5, 6, 7]
        assert _parse_selection_input("1-1", 10) == [1]

    def test_mixed_format(self):
        """Test parsing mixed numbers and ranges."""
        assert _parse_selection_input("1,3-5,7", 10) == [1, 3, 4, 5, 7]
        assert _parse_selection_input("1-3,5,7-9", 10) == [1, 2, 3, 5, 7, 8, 9]
        assert _parse_selection_input("2,4-6,8,10", 10) == [2, 4, 5, 6, 8, 10]

    def test_whitespace_handling(self):
        """Test that whitespace is handled correctly."""
        assert _parse_selection_input(" 1 , 3 , 5 ", 10) == [1, 3, 5]
        assert _parse_selection_input("1 - 4", 10) == [1, 2, 3, 4]
        assert _parse_selection_input(" 1-3 , 5 - 7 ", 10) == [1, 2, 3, 5, 6, 7]

    def test_duplicate_handling(self):
        """Test that duplicates are removed."""
        # Overlapping ranges
        assert _parse_selection_input("1-3,2-4", 10) == [1, 2, 3, 4]
        # Duplicate numbers
        assert _parse_selection_input("1,1,2,2,3", 10) == [1, 2, 3]

    def test_sorting(self):
        """Test that output is sorted."""
        assert _parse_selection_input("5,1,3", 10) == [1, 3, 5]
        assert _parse_selection_input("7-9,1-3", 10) == [1, 2, 3, 7, 8, 9]

    def test_invalid_range_too_high(self):
        """Test error when range exceeds max_index."""
        with pytest.raises(ValueError, match="Invalid range"):
            _parse_selection_input("1-20", 10)

    def test_invalid_range_too_low(self):
        """Test error when range starts below 1."""
        with pytest.raises(ValueError, match="Invalid range"):
            _parse_selection_input("0-5", 10)

    def test_invalid_range_reversed(self):
        """Test error when range is reversed."""
        with pytest.raises(ValueError, match="Invalid range"):
            _parse_selection_input("5-1", 10)

    def test_invalid_number_too_high(self):
        """Test error when number exceeds max_index."""
        with pytest.raises(ValueError, match="Invalid number"):
            _parse_selection_input("15", 10)

    def test_invalid_number_too_low(self):
        """Test error when number is below 1."""
        with pytest.raises(ValueError, match="Invalid number"):
            _parse_selection_input("0", 10)

    def test_invalid_format_letters(self):
        """Test error with invalid text."""
        with pytest.raises(ValueError, match="Invalid number"):
            _parse_selection_input("abc", 10)

    def test_invalid_format_special_chars(self):
        """Test error with special characters."""
        with pytest.raises(ValueError, match="Invalid"):
            _parse_selection_input("1,2#3", 10)


class TestSectionFiltering:
    """Tests for section filtering logic."""

    def test_filter_with_selection(self):
        """Test filtering sections based on selected_section_ids."""
        sections = [
            {"id": "hook", "title": "Hook", "optional": False},
            {"id": "problem", "title": "Problem", "optional": False},
            {"id": "impl", "title": "Implementation", "optional": False},
            {"id": "opt1", "title": "Optional 1", "optional": True},
            {"id": "opt2", "title": "Optional 2", "optional": True},
        ]
        selected_ids = ["hook", "problem", "impl", "opt1"]

        # Simulate the filtering logic used in nodes
        result = [s for s in sections if s["id"] in selected_ids]

        assert len(result) == 4
        assert result[0]["id"] == "hook"
        assert result[1]["id"] == "problem"
        assert result[2]["id"] == "impl"
        assert result[3]["id"] == "opt1"
        assert "opt2" not in [s["id"] for s in result]

    def test_filter_preserves_order(self):
        """Test that filtering preserves plan order."""
        sections = [
            {"id": "a", "optional": False},
            {"id": "b", "optional": True},
            {"id": "c", "optional": False},
            {"id": "d", "optional": True},
        ]
        selected_ids = ["a", "c", "d"]  # Note: d comes after c in plan

        result = [s for s in sections if s["id"] in selected_ids]

        assert len(result) == 3
        assert [s["id"] for s in result] == ["a", "c", "d"]

    def test_fallback_without_selection(self):
        """Test backward compatibility - no selected_ids filters out optional."""
        sections = [
            {"id": "a", "optional": False},
            {"id": "b", "optional": True},
            {"id": "c", "optional": False},
        ]
        selected_ids = []

        # Simulate the fallback logic
        if selected_ids:
            result = [s for s in sections if s["id"] in selected_ids]
        else:
            result = [s for s in sections if not s.get("optional")]

        assert len(result) == 2
        assert result[0]["id"] == "a"
        assert result[1]["id"] == "c"

    def test_all_required_sections_selected(self):
        """Test when only required sections are selected."""
        sections = [
            {"id": "req1", "optional": False},
            {"id": "req2", "optional": False},
            {"id": "opt1", "optional": True},
        ]
        selected_ids = ["req1", "req2"]

        result = [s for s in sections if s["id"] in selected_ids]

        assert len(result) == 2
        assert all(s["id"].startswith("req") for s in result)

    def test_all_sections_selected(self):
        """Test when all sections (required + optional) are selected."""
        sections = [
            {"id": "req1", "optional": False},
            {"id": "opt1", "optional": True},
            {"id": "opt2", "optional": True},
        ]
        selected_ids = ["req1", "opt1", "opt2"]

        result = [s for s in sections if s["id"] in selected_ids]

        assert len(result) == 3

    def test_empty_sections_list(self):
        """Test with empty sections list."""
        sections = []
        selected_ids = ["anything"]

        result = [s for s in sections if s["id"] in selected_ids]

        assert len(result) == 0

    def test_no_matching_ids(self):
        """Test when selected_ids don't match any sections."""
        sections = [
            {"id": "a", "optional": False},
            {"id": "b", "optional": True},
        ]
        selected_ids = ["x", "y", "z"]

        result = [s for s in sections if s["id"] in selected_ids]

        assert len(result) == 0
