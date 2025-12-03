"""
Unit tests for the KeyManager module.
"""

import json
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from src.utils.key_manager import KeyManager, get_key_manager


class TestKeyManager:
    """Tests for KeyManager class."""

    def setup_method(self):
        """Reset the singleton before each test."""
        KeyManager._instance = None

    def test_initialization_with_keys(self, tmp_path):
        """Test KeyManager initializes correctly with keys."""
        state_file = tmp_path / "state.json"
        keys = ["key1", "key2", "key3"]
        
        manager = KeyManager(keys=keys, state_file=state_file)
        
        assert len(manager._keys) == 3
        assert manager._current_index == 0
        print("✓ KeyManager initialized with 3 keys")

    def test_filters_empty_keys(self, tmp_path):
        """Test that empty/None keys are filtered out."""
        state_file = tmp_path / "state.json"
        keys = ["key1", "", None, "key2", ""]
        
        manager = KeyManager(keys=keys, state_file=state_file)
        
        assert len(manager._keys) == 2
        assert manager._keys == ["key1", "key2"]
        print("✓ Empty keys filtered out correctly")

    def test_round_robin_selection(self, tmp_path):
        """Test that keys are selected in round-robin order."""
        state_file = tmp_path / "state.json"
        keys = ["key1", "key2", "key3"]
        
        manager = KeyManager(keys=keys, state_file=state_file)
        
        # First round
        assert manager.get_next_key() == "key1"
        assert manager.get_next_key() == "key2"
        assert manager.get_next_key() == "key3"
        
        # Second round - should wrap around
        assert manager.get_next_key() == "key1"
        assert manager.get_next_key() == "key2"
        
        print("✓ Round-robin selection works correctly")

    def test_cooldown_tracking(self, tmp_path):
        """Test that rate-limited keys are put in cooldown."""
        state_file = tmp_path / "state.json"
        keys = ["key1", "key2", "key3"]
        
        manager = KeyManager(keys=keys, cooldown_seconds=60, state_file=state_file)
        
        # Get first key and mark it rate-limited
        key1 = manager.get_next_key()
        assert key1 == "key1"
        manager.mark_key_rate_limited(key1)
        
        # Next call should skip key1 and return key2
        assert manager.get_next_key() == "key2"
        
        # Mark key2 as rate-limited
        manager.mark_key_rate_limited("key2")
        
        # Should return key3
        assert manager.get_next_key() == "key3"
        
        print("✓ Cooldown tracking works correctly")

    def test_all_keys_in_cooldown(self, tmp_path):
        """Test behavior when all keys are in cooldown."""
        state_file = tmp_path / "state.json"
        keys = ["key1", "key2"]
        
        manager = KeyManager(keys=keys, cooldown_seconds=60, state_file=state_file)
        
        # Mark all keys as rate-limited
        manager.mark_key_rate_limited("key1")
        manager.mark_key_rate_limited("key2")
        
        # Should return None
        result = manager.get_next_key()
        assert result is None
        
        print("✓ Returns None when all keys in cooldown")

    def test_cooldown_expiry(self, tmp_path):
        """Test that keys become available after cooldown expires."""
        state_file = tmp_path / "state.json"
        keys = ["key1", "key2"]
        
        # Use very short cooldown for testing
        manager = KeyManager(keys=keys, cooldown_seconds=1, state_file=state_file)
        
        manager.mark_key_rate_limited("key1")
        
        # key1 should be in cooldown, get key2
        assert manager.get_next_key() == "key2"
        
        # Wait for cooldown to expire
        time.sleep(1.1)
        
        # key1 should be available again
        assert manager.get_next_key() == "key1"
        
        print("✓ Keys become available after cooldown expires")

    def test_get_wait_time_for_next_key(self, tmp_path):
        """Test getting wait time for next available key."""
        state_file = tmp_path / "state.json"
        keys = ["key1", "key2"]
        
        manager = KeyManager(keys=keys, cooldown_seconds=60, state_file=state_file)
        
        # No cooldowns - should return 0
        assert manager.get_wait_time_for_next_key() == 0
        
        # Mark one key - should still return 0 (other key available)
        manager.mark_key_rate_limited("key1")
        assert manager.get_wait_time_for_next_key() == 0
        
        # Mark all keys
        manager.mark_key_rate_limited("key2")
        wait_time = manager.get_wait_time_for_next_key()
        assert 0 < wait_time <= 60
        
        print(f"✓ Wait time calculation works (wait_time={wait_time:.1f}s)")

    def test_get_cooldown_status(self, tmp_path):
        """Test getting cooldown status of all keys."""
        state_file = tmp_path / "state.json"
        keys = ["key1", "key2"]
        
        manager = KeyManager(keys=keys, cooldown_seconds=60, state_file=state_file)
        
        status = manager.get_cooldown_status()
        # Keys are short so they won't be truncated
        assert status["key1"]["status"] == "available"
        assert status["key2"]["status"] == "available"
        
        manager.mark_key_rate_limited("key1")
        status = manager.get_cooldown_status()
        assert status["key1"]["status"] == "cooldown"
        assert "remaining_seconds" in status["key1"]
        assert status["key2"]["status"] == "available"
        
        print("✓ Cooldown status reporting works correctly")

    def test_state_persistence(self, tmp_path):
        """Test that state is persisted and loaded correctly."""
        state_file = tmp_path / "state.json"
        keys = ["key1", "key2", "key3"]
        
        # Create first manager and advance state
        manager1 = KeyManager(keys=keys, cooldown_seconds=300, state_file=state_file)
        manager1.get_next_key()  # key1
        manager1.get_next_key()  # key2
        manager1.mark_key_rate_limited("key1")
        
        # Reset singleton
        KeyManager._instance = None
        
        # Create new manager - should load state
        manager2 = KeyManager(keys=keys, cooldown_seconds=300, state_file=state_file)
        
        # Index should be restored
        assert manager2._current_index == 2  # Was at index 2
        
        # Cooldown should be restored
        assert "key1..." in manager2._cooldowns or any(
            k.startswith("key1") for k in manager2._cooldowns
        )
        
        print("✓ State persistence works correctly")

    def test_reset(self, tmp_path):
        """Test reset clears all state."""
        state_file = tmp_path / "state.json"
        keys = ["key1", "key2"]
        
        manager = KeyManager(keys=keys, cooldown_seconds=60, state_file=state_file)
        
        manager.get_next_key()
        manager.mark_key_rate_limited("key1")
        
        manager.reset()
        
        assert manager._current_index == 0
        assert len(manager._cooldowns) == 0
        
        # First key should be key1 again
        assert manager.get_next_key() == "key1"
        
        print("✓ Reset works correctly")

    def test_singleton_pattern(self, tmp_path):
        """Test that KeyManager is a singleton."""
        state_file = tmp_path / "state.json"
        
        manager1 = KeyManager(keys=["key1"], state_file=state_file)
        manager2 = KeyManager()
        
        assert manager1 is manager2
        
        print("✓ Singleton pattern works correctly")

    def test_thread_safety(self, tmp_path):
        """Test thread-safe operations."""
        import threading
        
        state_file = tmp_path / "state.json"
        keys = ["key1", "key2", "key3", "key4", "key5"]
        
        # Reset singleton
        KeyManager._instance = None
        
        manager = KeyManager(keys=keys, state_file=state_file)
        results = []
        
        def get_keys(n):
            for _ in range(n):
                key = manager.get_next_key()
                results.append(key)
        
        threads = [threading.Thread(target=get_keys, args=(10,)) for _ in range(5)]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # Should have 50 results, all valid keys
        assert len(results) == 50
        assert all(k in keys for k in results)
        
        print("✓ Thread safety works correctly")


class TestGetKeyManager:
    """Tests for get_key_manager helper function."""

    def setup_method(self):
        """Reset the singleton before each test."""
        KeyManager._instance = None
        # Reset module-level variable
        import src.utils.key_manager as km
        km._key_manager = None

    def test_get_key_manager_with_settings(self, tmp_path):
        """Test get_key_manager initializes from settings."""
        from unittest.mock import MagicMock
        
        settings = MagicMock()
        settings.google_api_key = "key0"
        settings.google_api_key_1 = "key1"
        settings.google_api_key_2 = "key2"
        settings.google_api_key_3 = "key3"
        settings.google_api_key_4 = "key4"
        
        manager = get_key_manager(settings)
        
        assert len(manager._keys) == 5
        
        print("✓ get_key_manager loads keys from settings correctly")


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
