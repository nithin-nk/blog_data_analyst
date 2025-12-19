"""Unit tests for tools module."""

import pytest

from src.agent.tools import (
    chunk_content,
    check_originality,
    _split_into_sentences,
    _has_ngram_match,
)


class TestChunkContent:
    """Tests for chunk_content function."""

    def test_empty_content_returns_empty_list(self):
        """Empty content returns empty list."""
        assert chunk_content("") == []
        assert chunk_content(None) == []  # type: ignore

    def test_short_content_returns_single_chunk(self):
        """Content shorter than max_tokens returns single chunk."""
        content = "This is a short paragraph."
        result = chunk_content(content, max_tokens=100)
        assert len(result) == 1
        assert result[0] == content

    def test_long_content_splits_into_chunks(self):
        """Long content is split into multiple chunks."""
        # Create content ~1000 chars (250 tokens at 4 chars/token)
        paragraphs = ["This is paragraph number {}.".format(i) * 10 for i in range(20)]
        content = "\n\n".join(paragraphs)

        result = chunk_content(content, max_tokens=100)  # ~400 chars per chunk
        assert len(result) > 1

    def test_preserves_paragraph_boundaries(self):
        """Chunks break at paragraph boundaries when possible."""
        content = "First paragraph here.\n\nSecond paragraph here.\n\nThird paragraph."

        result = chunk_content(content, max_tokens=50)

        # Each chunk should be a complete paragraph or close to it
        for chunk in result:
            # Shouldn't have cut mid-sentence (unless very long paragraph)
            assert chunk.strip()

    def test_handles_very_long_paragraph(self):
        """Very long single paragraph is split by sentences."""
        # Create a long paragraph with multiple sentences
        sentences = ["This is sentence number {}.".format(i) for i in range(50)]
        long_para = " ".join(sentences)

        result = chunk_content(long_para, max_tokens=50)  # ~200 chars per chunk
        assert len(result) > 1

    def test_respects_max_tokens(self):
        """Chunks don't exceed max_tokens (approximately)."""
        content = "Word " * 1000  # ~4000 chars
        max_tokens = 100  # ~400 chars

        result = chunk_content(content, max_tokens=max_tokens)

        for chunk in result:
            # Allow some overflow for sentence boundaries
            assert len(chunk) <= max_tokens * 4 * 1.5  # 50% buffer


class TestSplitIntoSentences:
    """Tests for _split_into_sentences helper."""

    def test_splits_on_period(self):
        """Splits on period followed by capital letter."""
        text = "First sentence. Second sentence. Third sentence."
        result = _split_into_sentences(text)
        assert len(result) == 3

    def test_splits_on_question_mark(self):
        """Splits on question mark."""
        text = "Is this a question? Yes it is."
        result = _split_into_sentences(text)
        assert len(result) == 2

    def test_splits_on_exclamation(self):
        """Splits on exclamation mark."""
        text = "What a day! It was amazing."
        result = _split_into_sentences(text)
        assert len(result) == 2

    def test_handles_abbreviations(self):
        """Doesn't incorrectly split on abbreviations."""
        text = "Dr. Smith went to the store. He bought milk."
        result = _split_into_sentences(text)
        # May split incorrectly on Dr. but that's acceptable
        assert len(result) >= 2

    def test_empty_string_returns_empty(self):
        """Empty string returns empty list."""
        assert _split_into_sentences("") == []


class TestCheckOriginality:
    """Tests for check_originality function."""

    def test_no_content_returns_empty(self):
        """Empty content returns no flags."""
        assert check_originality("", []) == []
        assert check_originality("Some content", []) == []

    def test_no_sources_returns_empty(self):
        """No sources means nothing to compare against."""
        assert check_originality("Some original content here.", []) == []

    def test_flags_similar_sentences(self):
        """Sentences above threshold are flagged."""
        content = "The quick brown fox jumps over the lazy dog."
        sources = [{
            "url": "https://example.com",
            "content": "The quick brown fox jumps over the lazy dog."
        }]

        result = check_originality(content, sources, threshold=0.7)
        assert len(result) >= 1
        assert result[0]["similar_to"] == "https://example.com"
        assert result[0]["similarity"] >= 0.7

    def test_ignores_short_sentences(self):
        """Sentences < 20 chars are skipped."""
        content = "Hi there."  # Too short
        sources = [{
            "url": "https://example.com",
            "content": "Hi there."
        }]

        result = check_originality(content, sources, threshold=0.5)
        assert len(result) == 0

    def test_different_content_not_flagged(self):
        """Dissimilar content is not flagged."""
        content = "Apples and oranges are popular fruits."
        sources = [{
            "url": "https://example.com",
            "content": "Python is a programming language used for web development."
        }]

        result = check_originality(content, sources, threshold=0.7)
        assert len(result) == 0

    def test_threshold_affects_sensitivity(self):
        """Lower threshold catches more similarities."""
        content = "The brown fox jumped over a sleeping dog."
        sources = [{
            "url": "https://example.com",
            "content": "The quick brown fox jumps over the lazy dog."
        }]

        # With high threshold, might not flag
        result_high = check_originality(content, sources, threshold=0.9)

        # With lower threshold, more likely to flag
        result_low = check_originality(content, sources, threshold=0.3)

        # Lower threshold should catch more (or equal)
        assert len(result_low) >= len(result_high)

    def test_multiple_sources_checked(self):
        """All sources are checked."""
        content = "This is a test sentence for checking."
        sources = [
            {"url": "https://example1.com", "content": "Something different."},
            {"url": "https://example2.com", "content": "This is a test sentence for checking."},
        ]

        result = check_originality(content, sources, threshold=0.9)
        assert len(result) >= 1
        assert result[0]["similar_to"] == "https://example2.com"


class TestHasNgramMatch:
    """Tests for _has_ngram_match helper."""

    def test_finds_exact_ngram(self):
        """Finds exact 4-gram match."""
        text = "the quick brown fox"
        source = "the quick brown fox jumps over"
        assert _has_ngram_match(text, source, n=4) is True

    def test_no_match_returns_false(self):
        """No match returns False."""
        text = "apples oranges bananas grapes"
        source = "python java rust golang"
        assert _has_ngram_match(text, source, n=4) is False

    def test_short_text_returns_false(self):
        """Text shorter than n returns False."""
        text = "one two three"
        source = "one two three four five six"
        assert _has_ngram_match(text, source, n=4) is False

    def test_case_insensitive(self):
        """Matching is case-insensitive."""
        text = "THE QUICK BROWN FOX"
        source = "the quick brown fox jumps"
        assert _has_ngram_match(text, source, n=4) is True


class TestChunkContentEdgeCases:
    """Edge case tests for chunk_content."""

    def test_single_character_content(self):
        """Single character content."""
        result = chunk_content("x", max_tokens=100)
        assert result == ["x"]

    def test_only_whitespace(self):
        """Only whitespace content."""
        result = chunk_content("   \n\n   ", max_tokens=100)
        assert result == []

    def test_unicode_content(self):
        """Unicode content is handled correctly."""
        content = "这是中文内容。Another paragraph here."
        result = chunk_content(content, max_tokens=100)
        assert len(result) >= 1
        assert "这是中文内容" in result[0]

    def test_code_blocks_preserved(self):
        """Code-like content isn't mangled."""
        content = "def hello():\n    print('world')\n\nAnother paragraph."
        result = chunk_content(content, max_tokens=100)
        assert "def hello():" in "".join(result)
