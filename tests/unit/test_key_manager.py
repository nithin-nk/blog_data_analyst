"""Unit tests for KeyManager."""

import json
import os
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

from src.agent.key_manager import KeyManager, KeyUsage


@pytest.fixture
def temp_usage_dir(tmp_path: Path) -> Path:
    """Create a temporary usage directory."""
    usage_dir = tmp_path / "usage"
    usage_dir.mkdir()
    return usage_dir


@pytest.fixture
def sample_keys() -> list[str]:
    """Sample API keys for testing."""
    return ["key_1", "key_2", "key_3", "key_4"]


@pytest.fixture
def manager(sample_keys: list[str], temp_usage_dir: Path) -> KeyManager:
    """Create a KeyManager with temp storage."""
    mgr = KeyManager(keys=sample_keys)
    mgr._usage_dir = temp_usage_dir
    mgr._load_usage()  # Reinitialize with temp dir
    return mgr


class TestKeyManagerInit:
    """Tests for KeyManager initialization."""

    def test_init_with_keys(self, sample_keys: list[str], temp_usage_dir: Path):
        """KeyManager initializes with provided keys."""
        manager = KeyManager(keys=sample_keys)
        manager._usage_dir = temp_usage_dir
        assert len(manager.keys) == 4
        assert manager.rpd_limit == 250

    def test_init_empty_keys_raises(self):
        """KeyManager raises error with empty keys list."""
        with pytest.raises(ValueError, match="At least one API key"):
            KeyManager(keys=[])

    def test_from_env_loads_keys(self, temp_usage_dir: Path):
        """KeyManager.from_env() loads keys from environment."""
        # Clear any existing GOOGLE_API_KEY_* vars, then set only our test keys
        env_copy = {k: v for k, v in os.environ.items()
                   if not k.startswith("GOOGLE_API_KEY_")}
        env_copy["GOOGLE_API_KEY_1"] = "env_key_1"
        env_copy["GOOGLE_API_KEY_2"] = "env_key_2"
        with patch.dict(os.environ, env_copy, clear=True):
            manager = KeyManager.from_env()
            manager._usage_dir = temp_usage_dir
            assert len(manager.keys) == 2
            assert "env_key_1" in manager.keys

    def test_from_env_no_keys_raises(self):
        """KeyManager.from_env() raises if no keys in environment."""
        with patch.dict(os.environ, {}, clear=True):
            # Clear any existing GOOGLE_API_KEY_* vars
            env_copy = {k: v for k, v in os.environ.items()
                       if not k.startswith("GOOGLE_API_KEY_")}
            with patch.dict(os.environ, env_copy, clear=True):
                with pytest.raises(ValueError, match="No API keys found"):
                    KeyManager.from_env()


class TestGetBestKey:
    """Tests for get_best_key method."""

    def test_get_best_key_returns_unused_key(self, manager: KeyManager):
        """Best key is one with most remaining quota."""
        key = manager.get_best_key()
        assert key in manager.keys

    def test_get_best_key_prefers_least_used(self, manager: KeyManager):
        """Key with fewer requests is preferred."""
        # Use first key 100 times
        for _ in range(100):
            manager.record_usage(manager.keys[0], tokens_in=10, tokens_out=5)

        # Best key should NOT be the heavily used one
        best = manager.get_best_key()
        assert best != manager.keys[0]

    def test_get_best_key_skips_rate_limited(self, manager: KeyManager):
        """Rate-limited keys are skipped."""
        # Rate limit first 3 keys
        for i in range(3):
            manager.mark_rate_limited(manager.keys[i])

        # Only 4th key should be returned
        best = manager.get_best_key()
        assert best == manager.keys[3]

    def test_get_best_key_raises_when_all_exhausted(self, manager: KeyManager):
        """Raises RuntimeError when all keys are rate-limited."""
        for key in manager.keys:
            manager.mark_rate_limited(key)

        with pytest.raises(RuntimeError, match="All API keys exhausted"):
            manager.get_best_key()


class TestRecordUsage:
    """Tests for record_usage method."""

    def test_record_usage_increments_requests(self, manager: KeyManager):
        """Usage recording increments request count."""
        key = manager.keys[0]
        manager.record_usage(key, tokens_in=100, tokens_out=50)

        usage = manager.usage[key]
        assert usage.requests == 1
        assert usage.tokens_in == 100
        assert usage.tokens_out == 50

    def test_record_usage_accumulates(self, manager: KeyManager):
        """Multiple usage records accumulate."""
        key = manager.keys[0]
        manager.record_usage(key, tokens_in=100, tokens_out=50)
        manager.record_usage(key, tokens_in=200, tokens_out=100)

        usage = manager.usage[key]
        assert usage.requests == 2
        assert usage.tokens_in == 300
        assert usage.tokens_out == 150

    def test_record_usage_sets_last_used(self, manager: KeyManager):
        """Last used timestamp is set."""
        key = manager.keys[0]
        manager.record_usage(key)

        assert manager.usage[key].last_used is not None


class TestMarkRateLimited:
    """Tests for mark_rate_limited method."""

    def test_mark_rate_limited_sets_flag(self, manager: KeyManager):
        """Rate limited flag is set."""
        key = manager.keys[0]
        manager.mark_rate_limited(key)

        assert manager.usage[key].rate_limited is True

    def test_mark_rate_limited_persists(
        self,
        sample_keys: list[str],
        temp_usage_dir: Path
    ):
        """Rate limited status persists to disk."""
        manager1 = KeyManager(keys=sample_keys)
        manager1._usage_dir = temp_usage_dir
        manager1._load_usage()

        manager1.mark_rate_limited(sample_keys[0])

        # Create new manager instance
        manager2 = KeyManager(keys=sample_keys)
        manager2._usage_dir = temp_usage_dir
        manager2._load_usage()

        assert manager2.usage[sample_keys[0]].rate_limited is True


class TestGetNextKey:
    """Tests for get_next_key method."""

    def test_get_next_key_returns_alternative(self, manager: KeyManager):
        """Next key returns an alternative after rate limit."""
        first_key = manager.keys[0]
        next_key = manager.get_next_key(first_key)

        assert next_key is not None
        assert next_key != first_key

    def test_get_next_key_marks_current_limited(self, manager: KeyManager):
        """Current key is marked rate-limited."""
        first_key = manager.keys[0]
        manager.get_next_key(first_key)

        assert manager.usage[first_key].rate_limited is True

    def test_get_next_key_returns_none_when_exhausted(self, manager: KeyManager):
        """Returns None when all keys exhausted."""
        # Rate limit all but one
        for key in manager.keys[:-1]:
            manager.mark_rate_limited(key)

        # Get next for last key should return None
        last_key = manager.keys[-1]
        next_key = manager.get_next_key(last_key)

        assert next_key is None


class TestUsagePersistence:
    """Tests for usage persistence to disk."""

    def test_usage_persists_to_disk(
        self,
        sample_keys: list[str],
        temp_usage_dir: Path
    ):
        """Usage data is saved to JSON file."""
        manager = KeyManager(keys=sample_keys)
        manager._usage_dir = temp_usage_dir
        manager._load_usage()

        manager.record_usage(sample_keys[0], tokens_in=100, tokens_out=50)

        # Check file exists
        usage_file = temp_usage_dir / f"{date.today().isoformat()}.json"
        assert usage_file.exists()

        # Check content
        with open(usage_file) as f:
            data = json.load(f)
        assert sample_keys[0] in data
        assert data[sample_keys[0]]["requests"] == 1

    def test_usage_loads_from_disk(
        self,
        sample_keys: list[str],
        temp_usage_dir: Path
    ):
        """Usage data is loaded from existing file."""
        # Pre-create usage file
        usage_file = temp_usage_dir / f"{date.today().isoformat()}.json"
        with open(usage_file, "w") as f:
            json.dump({
                sample_keys[0]: {
                    "requests": 50,
                    "tokens_in": 5000,
                    "tokens_out": 2500,
                    "rate_limited": False,
                    "last_used": None,
                }
            }, f)

        manager = KeyManager(keys=sample_keys)
        manager._usage_dir = temp_usage_dir
        manager._load_usage()

        assert manager.usage[sample_keys[0]].requests == 50


class TestDateRollover:
    """Tests for date rollover behavior."""

    def test_usage_resets_on_new_day(
        self,
        sample_keys: list[str],
        temp_usage_dir: Path
    ):
        """Usage resets when date changes."""
        manager = KeyManager(keys=sample_keys)
        manager._usage_dir = temp_usage_dir
        manager._load_usage()

        # Record some usage
        manager.record_usage(sample_keys[0], tokens_in=100, tokens_out=50)
        assert manager.usage[sample_keys[0]].requests == 1

        # Simulate next day
        manager._current_date = date.today() - timedelta(days=1)
        manager._check_date_rollover()

        # Usage should be reset
        assert manager.usage[sample_keys[0]].requests == 0


class TestGetUsageStats:
    """Tests for get_usage_stats method."""

    def test_get_usage_stats_returns_summary(self, manager: KeyManager):
        """Usage stats returns summary for all keys."""
        manager.record_usage(manager.keys[0], tokens_in=100, tokens_out=50)

        stats = manager.get_usage_stats()

        assert "date" in stats
        assert "rpd_limit" in stats
        assert stats["rpd_limit"] == 250
        assert "keys" in stats
        assert "KEY_1" in stats["keys"]  # Masked key name
        assert stats["keys"]["KEY_1"]["requests"] == 1

    def test_get_usage_stats_masks_keys(self, manager: KeyManager):
        """Actual API keys are not exposed in stats."""
        stats = manager.get_usage_stats()

        # Check that actual keys are not in output
        for key in manager.keys:
            assert key not in str(stats)


class TestResetRateLimits:
    """Tests for reset_rate_limits method."""

    def test_reset_rate_limits_clears_all(self, manager: KeyManager):
        """All rate limits are cleared."""
        # Rate limit all keys
        for key in manager.keys:
            manager.mark_rate_limited(key)

        # Reset
        manager.reset_rate_limits()

        # All should be usable
        for key in manager.keys:
            assert manager.usage[key].rate_limited is False


class TestPaidKeyFallback:
    """Tests for GOOGLE_PAID_KEY fallback."""

    def test_from_env_loads_paid_key(self, temp_usage_dir: Path):
        """KeyManager.from_env() loads GOOGLE_PAID_KEY if present."""
        env_copy = {k: v for k, v in os.environ.items()
                   if not k.startswith("GOOGLE_")}
        env_copy["GOOGLE_API_KEY_1"] = "free_key_1"
        env_copy["GOOGLE_PAID_KEY"] = "paid_key"
        with patch.dict(os.environ, env_copy, clear=True):
            manager = KeyManager.from_env()
            manager._usage_dir = temp_usage_dir
            assert manager.paid_key == "paid_key"

    def test_from_env_no_paid_key(self, temp_usage_dir: Path):
        """KeyManager.from_env() sets paid_key to None if not present."""
        env_copy = {k: v for k, v in os.environ.items()
                   if not k.startswith("GOOGLE_")}
        env_copy["GOOGLE_API_KEY_1"] = "free_key_1"
        with patch.dict(os.environ, env_copy, clear=True):
            manager = KeyManager.from_env()
            manager._usage_dir = temp_usage_dir
            assert manager.paid_key is None

    def test_get_best_key_falls_back_to_paid_key(
        self,
        sample_keys: list[str],
        temp_usage_dir: Path
    ):
        """get_best_key returns paid_key when all free keys exhausted."""
        manager = KeyManager(keys=sample_keys, paid_key="paid_key")
        manager._usage_dir = temp_usage_dir
        manager._load_usage()

        # Rate limit all free keys
        for key in manager.keys:
            manager.mark_rate_limited(key)

        # Should return paid key instead of raising
        best = manager.get_best_key()
        assert best == "paid_key"

    def test_get_best_key_prefers_free_keys(
        self,
        sample_keys: list[str],
        temp_usage_dir: Path
    ):
        """get_best_key uses free keys before paid key."""
        manager = KeyManager(keys=sample_keys, paid_key="paid_key")
        manager._usage_dir = temp_usage_dir
        manager._load_usage()

        # Don't rate limit any keys
        best = manager.get_best_key()
        assert best in sample_keys
        assert best != "paid_key"

    def test_get_best_key_raises_when_no_paid_key(
        self,
        sample_keys: list[str],
        temp_usage_dir: Path
    ):
        """get_best_key raises if no paid_key and all free keys exhausted."""
        manager = KeyManager(keys=sample_keys, paid_key=None)
        manager._usage_dir = temp_usage_dir
        manager._load_usage()

        # Rate limit all free keys
        for key in manager.keys:
            manager.mark_rate_limited(key)

        with pytest.raises(RuntimeError, match="All API keys exhausted"):
            manager.get_best_key()

    def test_get_next_key_falls_back_to_paid_key(
        self,
        sample_keys: list[str],
        temp_usage_dir: Path
    ):
        """get_next_key returns paid_key when all free keys exhausted."""
        manager = KeyManager(keys=sample_keys, paid_key="paid_key")
        manager._usage_dir = temp_usage_dir
        manager._load_usage()

        # Rate limit all but last key
        for key in sample_keys[:-1]:
            manager.mark_rate_limited(key)

        # Get next for last key should return paid key
        last_key = sample_keys[-1]
        next_key = manager.get_next_key(last_key)

        assert next_key == "paid_key"

    def test_get_next_key_returns_none_when_no_paid_key(
        self,
        sample_keys: list[str],
        temp_usage_dir: Path
    ):
        """get_next_key returns None if no paid_key and all exhausted."""
        manager = KeyManager(keys=sample_keys, paid_key=None)
        manager._usage_dir = temp_usage_dir
        manager._load_usage()

        # Rate limit all but last key
        for key in sample_keys[:-1]:
            manager.mark_rate_limited(key)

        # Get next for last key should return None
        last_key = sample_keys[-1]
        next_key = manager.get_next_key(last_key)

        assert next_key is None
