"""
KeyManager - Manages multiple Gemini API keys with usage tracking and rotation.

Features:
- Tracks requests per day (RPD) per key
- Rotates to next key on 429 error
- Selects key with most remaining quota
- Persists usage to ~/.blog_agent/usage/{date}.json
"""

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, date
from pathlib import Path
from typing import Any


# Default RPD (requests per day) limit per key
DEFAULT_RPD_LIMIT = 250

# Base directory for blog agent data
BLOG_AGENT_DIR = Path.home() / ".blog_agent"
USAGE_DIR = BLOG_AGENT_DIR / "usage"


@dataclass
class KeyUsage:
    """Usage statistics for a single API key."""

    requests: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    rate_limited: bool = False
    last_used: str | None = None


@dataclass
class KeyManager:
    """
    Manages multiple Gemini API keys with usage tracking and rotation.

    Example:
        keys = ["key1", "key2", "key3", "key4"]
        manager = KeyManager(keys)

        # Get best available key
        key = manager.get_best_key()

        # Record usage after API call
        manager.record_usage(key, tokens_in=100, tokens_out=50)

        # Handle rate limit
        if got_429_error:
            manager.mark_rate_limited(key)
            next_key = manager.get_next_key(key)
    """

    keys: list[str]
    rpd_limit: int = DEFAULT_RPD_LIMIT
    usage: dict[str, KeyUsage] = field(default_factory=dict)
    _current_date: date = field(default_factory=date.today)
    _usage_dir: Path = field(default_factory=lambda: USAGE_DIR)

    def __post_init__(self) -> None:
        """Initialize usage tracking for all keys."""
        if not self.keys:
            raise ValueError("At least one API key is required")

        # Ensure usage directory exists
        self._usage_dir.mkdir(parents=True, exist_ok=True)

        # Load existing usage or initialize fresh
        self._load_usage()

    @classmethod
    def from_env(cls, rpd_limit: int = DEFAULT_RPD_LIMIT) -> "KeyManager":
        """
        Create KeyManager from environment variables.

        Looks for GOOGLE_API_KEY_1 through GOOGLE_API_KEY_4.
        """
        keys = []
        for i in range(1, 5):
            key = os.environ.get(f"GOOGLE_API_KEY_{i}")
            if key:
                keys.append(key)

        if not keys:
            raise ValueError(
                "No API keys found. Set GOOGLE_API_KEY_1 through GOOGLE_API_KEY_4 "
                "environment variables."
            )

        return cls(keys=keys, rpd_limit=rpd_limit)

    def get_best_key(self) -> str:
        """
        Return the key with the most remaining RPD quota.

        Returns:
            API key string

        Raises:
            RuntimeError: If all keys are exhausted or rate-limited
        """
        self._check_date_rollover()

        best_key = None
        best_remaining = -1

        for key in self.keys:
            usage = self.usage.get(key, KeyUsage())

            # Skip rate-limited keys
            if usage.rate_limited:
                continue

            remaining = self.rpd_limit - usage.requests
            if remaining > best_remaining:
                best_remaining = remaining
                best_key = key

        if best_key is None:
            raise RuntimeError(
                "All API keys exhausted or rate-limited. "
                "Wait for quota reset or add more keys."
            )

        return best_key

    def record_usage(
        self,
        key: str,
        tokens_in: int = 0,
        tokens_out: int = 0
    ) -> None:
        """
        Record usage for a key after an API call.

        Args:
            key: The API key used
            tokens_in: Input tokens consumed
            tokens_out: Output tokens generated
        """
        self._check_date_rollover()

        if key not in self.usage:
            self.usage[key] = KeyUsage()

        usage = self.usage[key]
        usage.requests += 1
        usage.tokens_in += tokens_in
        usage.tokens_out += tokens_out
        usage.last_used = datetime.now().isoformat()

        self._save_usage()

    def mark_rate_limited(self, key: str) -> None:
        """
        Mark a key as rate-limited (received 429 error).

        Args:
            key: The API key that hit rate limit
        """
        if key not in self.usage:
            self.usage[key] = KeyUsage()

        self.usage[key].rate_limited = True
        self._save_usage()

    def get_next_key(self, current_key: str) -> str | None:
        """
        Get the next available key after the current one is rate-limited.

        Args:
            current_key: The key that just hit rate limit

        Returns:
            Next available key, or None if all exhausted
        """
        self._check_date_rollover()

        # Mark current as rate-limited if not already
        self.mark_rate_limited(current_key)

        try:
            return self.get_best_key()
        except RuntimeError:
            return None

    def get_usage_stats(self) -> dict[str, Any]:
        """
        Get current usage statistics for all keys.

        Returns:
            Dictionary with usage stats per key
        """
        self._check_date_rollover()

        stats = {
            "date": self._current_date.isoformat(),
            "rpd_limit": self.rpd_limit,
            "keys": {}
        }

        for i, key in enumerate(self.keys, 1):
            usage = self.usage.get(key, KeyUsage())
            # Mask the key for security
            masked_key = f"KEY_{i}"
            stats["keys"][masked_key] = {
                "requests": usage.requests,
                "remaining": self.rpd_limit - usage.requests,
                "tokens_in": usage.tokens_in,
                "tokens_out": usage.tokens_out,
                "rate_limited": usage.rate_limited,
            }

        return stats

    def reset_rate_limits(self) -> None:
        """Reset rate-limited status for all keys (manual override)."""
        for key in self.keys:
            if key in self.usage:
                self.usage[key].rate_limited = False
        self._save_usage()

    def _check_date_rollover(self) -> None:
        """Check if date has changed and reset usage if needed."""
        today = date.today()
        if today != self._current_date:
            self._current_date = today
            # Reset all usage for new day
            self.usage = {key: KeyUsage() for key in self.keys}
            self._save_usage()

    def _get_usage_file(self) -> Path:
        """Get path to today's usage file."""
        return self._usage_dir / f"{self._current_date.isoformat()}.json"

    def _load_usage(self) -> None:
        """Load usage from disk for today."""
        usage_file = self._get_usage_file()

        if usage_file.exists():
            try:
                with open(usage_file, "r") as f:
                    data = json.load(f)

                # Reconstruct usage objects
                self.usage = {}
                for key in self.keys:
                    if key in data:
                        self.usage[key] = KeyUsage(**data[key])
                    else:
                        self.usage[key] = KeyUsage()
            except (json.JSONDecodeError, TypeError):
                # Corrupted file, start fresh
                self.usage = {key: KeyUsage() for key in self.keys}
        else:
            # No file for today, start fresh
            self.usage = {key: KeyUsage() for key in self.keys}

    def _save_usage(self) -> None:
        """Persist usage to disk."""
        usage_file = self._get_usage_file()

        # Convert to serializable format
        data = {}
        for key, usage in self.usage.items():
            data[key] = {
                "requests": usage.requests,
                "tokens_in": usage.tokens_in,
                "tokens_out": usage.tokens_out,
                "rate_limited": usage.rate_limited,
                "last_used": usage.last_used,
            }

        with open(usage_file, "w") as f:
            json.dump(data, f, indent=2)
