"""
Intelligent API Key Manager with Round-Robin and Cooldown Tracking.

Distributes load across multiple API keys evenly and handles rate limits
by placing keys in cooldown for a configurable period.
"""

import json
import threading
import time
from pathlib import Path
from typing import Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


class KeyManager:
    """
    Manages multiple API keys with round-robin selection and cooldown tracking.
    
    Features:
    - Round-robin distribution for even load balancing
    - Cooldown tracking when keys hit rate limits (default: 60 seconds)
    - Persistent state saved to file for recovery across restarts
    - Thread-safe operations
    """
    
    _instance: Optional["KeyManager"] = None
    _lock = threading.Lock()
    
    # Default cooldown period in seconds
    DEFAULT_COOLDOWN_SECONDS = 60
    
    # State file path
    STATE_FILE = Path("outputs/.key_manager_state.json")
    
    def __new__(cls, *args, **kwargs):
        """Singleton pattern to ensure single key manager instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(
        self,
        keys: Optional[list[str]] = None,
        cooldown_seconds: int = DEFAULT_COOLDOWN_SECONDS,
        state_file: Optional[Path] = None,
    ):
        """
        Initialize the key manager.
        
        Args:
            keys: List of API keys to manage
            cooldown_seconds: Seconds to wait before retrying a rate-limited key
            state_file: Path to persist state (default: outputs/.key_manager_state.json)
        """
        # Skip re-initialization if already set up (singleton)
        if self._initialized and keys is None:
            return
        
        self._keys: list[str] = []
        self._current_index = 0
        self._cooldowns: dict[str, float] = {}  # key -> cooldown_end_timestamp
        self._cooldown_seconds = cooldown_seconds
        self._state_file = state_file or self.STATE_FILE
        self._call_lock = threading.Lock()
        
        if keys:
            self.set_keys(keys)
        
        # Load persisted state
        self._load_state()
        self._initialized = True
    
    def set_keys(self, keys: list[str]) -> None:
        """
        Set the list of API keys to manage.
        
        Args:
            keys: List of API keys (filters out empty/None values)
        """
        with self._call_lock:
            self._keys = [k for k in keys if k]
            # Reset index if it's out of bounds
            if self._current_index >= len(self._keys):
                self._current_index = 0
            logger.info(f"KeyManager initialized with {len(self._keys)} keys")
    
    def get_next_key(self) -> Optional[str]:
        """
        Get the next available key using round-robin with cooldown awareness.
        
        Returns:
            The next available API key, or None if all keys are in cooldown
        """
        with self._call_lock:
            if not self._keys:
                logger.error("No API keys configured")
                return None
            
            current_time = time.time()
            checked_count = 0
            
            # Try to find an available key starting from current index
            while checked_count < len(self._keys):
                key = self._keys[self._current_index]
                key_id = self._get_key_id(key)
                
                # Check if key is in cooldown
                cooldown_end = self._cooldowns.get(key_id, 0)
                if current_time >= cooldown_end:
                    # Key is available - advance index for next call
                    selected_key = key
                    self._current_index = (self._current_index + 1) % len(self._keys)
                    
                    # Clear expired cooldown
                    if key_id in self._cooldowns:
                        del self._cooldowns[key_id]
                        self._save_state()
                    
                    logger.debug(f"Selected key {key_id} (index {self._current_index - 1})")
                    return selected_key
                else:
                    # Key in cooldown - try next
                    remaining = int(cooldown_end - current_time)
                    logger.debug(f"Key {key_id} in cooldown for {remaining}s more")
                    self._current_index = (self._current_index + 1) % len(self._keys)
                    checked_count += 1
            
            # All keys are in cooldown
            logger.warning("All API keys are in cooldown")
            return None
    
    def mark_key_rate_limited(self, key: str) -> None:
        """
        Mark a key as rate-limited, putting it in cooldown.
        
        Args:
            key: The API key that hit a rate limit
        """
        with self._call_lock:
            key_id = self._get_key_id(key)
            cooldown_end = time.time() + self._cooldown_seconds
            self._cooldowns[key_id] = cooldown_end
            self._save_state()
            logger.info(
                f"Key {key_id} marked rate-limited, cooldown for {self._cooldown_seconds}s"
            )
    
    def get_cooldown_status(self) -> dict[str, dict]:
        """
        Get the current cooldown status of all keys.
        
        Returns:
            Dict mapping key IDs to their status (available/cooldown with remaining time)
        """
        with self._call_lock:
            current_time = time.time()
            status = {}
            
            for key in self._keys:
                key_id = self._get_key_id(key)
                cooldown_end = self._cooldowns.get(key_id, 0)
                
                if current_time >= cooldown_end:
                    status[key_id] = {"status": "available"}
                else:
                    remaining = int(cooldown_end - current_time)
                    status[key_id] = {"status": "cooldown", "remaining_seconds": remaining}
            
            return status
    
    def get_wait_time_for_next_key(self) -> float:
        """
        Get the minimum wait time until a key becomes available.
        
        Returns:
            Seconds to wait, or 0 if a key is immediately available
        """
        with self._call_lock:
            if not self._keys:
                return 0
            
            current_time = time.time()
            min_wait = float("inf")
            
            for key in self._keys:
                key_id = self._get_key_id(key)
                cooldown_end = self._cooldowns.get(key_id, 0)
                wait_time = max(0, cooldown_end - current_time)
                
                if wait_time == 0:
                    return 0
                
                min_wait = min(min_wait, wait_time)
            
            return min_wait if min_wait != float("inf") else 0
    
    def reset(self) -> None:
        """Reset all cooldowns and the round-robin index."""
        with self._call_lock:
            self._current_index = 0
            self._cooldowns.clear()
            self._save_state()
            logger.info("KeyManager state reset")
    
    def _get_key_id(self, key: str) -> str:
        """Get a safe identifier for a key (first 8 chars for logging)."""
        return key[:8] + "..." if len(key) > 8 else key
    
    def _save_state(self) -> None:
        """Persist the current state to file."""
        try:
            self._state_file.parent.mkdir(parents=True, exist_ok=True)
            state = {
                "current_index": self._current_index,
                "cooldowns": self._cooldowns,
                "saved_at": time.time(),
            }
            with open(self._state_file, "w") as f:
                json.dump(state, f, indent=2)
            logger.debug(f"KeyManager state saved to {self._state_file}")
        except Exception as e:
            logger.warning(f"Failed to save KeyManager state: {e}")
    
    def _load_state(self) -> None:
        """Load persisted state from file."""
        try:
            if not self._state_file.exists():
                return
            
            with open(self._state_file, "r") as f:
                state = json.load(f)
            
            self._current_index = state.get("current_index", 0)
            
            # Only restore cooldowns that haven't expired
            current_time = time.time()
            loaded_cooldowns = state.get("cooldowns", {})
            self._cooldowns = {
                k: v for k, v in loaded_cooldowns.items() 
                if v > current_time
            }
            
            # Ensure index is valid
            if self._keys and self._current_index >= len(self._keys):
                self._current_index = 0
            
            logger.info(
                f"KeyManager state loaded: index={self._current_index}, "
                f"active_cooldowns={len(self._cooldowns)}"
            )
        except Exception as e:
            logger.warning(f"Failed to load KeyManager state: {e}")


# Global instance getter
_key_manager: Optional[KeyManager] = None


def get_key_manager(settings=None) -> KeyManager:
    """
    Get or create the global KeyManager instance.
    
    Args:
        settings: Optional settings object with API keys
        
    Returns:
        The KeyManager singleton instance
    """
    global _key_manager
    
    if _key_manager is None:
        _key_manager = KeyManager()
    
    if settings is not None:
        keys = [
            getattr(settings, "google_api_key", None),
            getattr(settings, "google_api_key_1", None),
            getattr(settings, "google_api_key_2", None),
            getattr(settings, "google_api_key_3", None),
            getattr(settings, "google_api_key_4", None),
        ]
        _key_manager.set_keys(keys)
    
    return _key_manager
