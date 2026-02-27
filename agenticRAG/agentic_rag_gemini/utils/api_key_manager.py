"""API Key Manager for Gemini API with automatic rotation on quota errors.

This module provides a singleton class to manage multiple Gemini API keys
with automatic rotation when quota limits are exceeded.
"""

import os
import threading
from typing import List, Set, Optional
from utils.logger import get_logger

logger = get_logger(__name__)

# Singleton instance
_instance: Optional['APIKeyManager'] = None
_lock = threading.Lock()


def get_api_key_manager() -> 'APIKeyManager':
    """Get the singleton APIKeyManager instance.
    
    Returns:
        APIKeyManager: The singleton instance
    """
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = APIKeyManager()
    return _instance


def reset_api_key_manager():
    """Reset the singleton instance (useful for testing)."""
    global _instance
    with _lock:
        _instance = None


class APIKeyManager:
    """Manages multiple Gemini API keys with circular rotation on quota errors.
    
    This class loads API keys from environment variables and provides
    automatic round-robin rotation when a key hits its quota limit.
    Keys are NEVER permanently blacklisted; instead the system gives up
    only after 2 full rotation cycles of consecutive failures.
    
    Environment Variables:
        GEMINI_API_KEYS: Comma-separated list of API keys (preferred)
        GEMINI_API_KEY: Single API key (fallback for backward compatibility)
    
    Usage:
        manager = get_api_key_manager()
        key = manager.get_current_key()
        
        # On quota error:
        if manager.rotate_to_next_key():
            # Try again with new key
            key = manager.get_current_key()
        else:
            # 2 full cycles exhausted
            handle_error()
    """

    # How many consecutive failures across all keys before giving up.
    # Default = 2 full cycles (2 * N keys).
    _MAX_CYCLES = 2
    
    def __init__(self):
        """Initialize the API key manager."""
        self._keys: List[str] = []
        self._current_index: int = 0
        self._consecutive_failures: int = 0  # resets on any success
        self._last_error: str = ""
        self._lock = threading.Lock()
        
        self._load_keys()
        
        if not self._keys:
            logger.warning("No Gemini API keys found in environment variables!")
        else:
            logger.info(f"APIKeyManager initialized with {len(self._keys)} API key(s)")
    
    def _load_keys(self):
        """Load API keys from environment variables."""
        # Try loading multiple keys first (comma-separated)
        keys_str = os.getenv("GEMINI_API_KEYS", "")
        
        if keys_str:
            # Parse comma-separated keys, strip whitespace
            self._keys = [k.strip() for k in keys_str.split(",") if k.strip()]
            logger.info(f"Loaded {len(self._keys)} API keys from GEMINI_API_KEYS")
        
        # Fallback to single key for backward compatibility
        if not self._keys:
            single_key = os.getenv("GEMINI_API_KEY", "")
            if single_key:
                self._keys = [single_key.strip()]
                logger.info("Loaded 1 API key from GEMINI_API_KEY (legacy)")
    
    def get_current_key(self) -> str:
        """Get the current active API key.
        
        Returns:
            str: The current API key, or empty string if no keys available
        """
        with self._lock:
            if not self._keys:
                return ""
            return self._keys[self._current_index]
    
    def get_current_key_index(self) -> int:
        """Get the current key index.
        
        Returns:
            int: Current key index
        """
        return self._current_index
    
    def get_total_keys(self) -> int:
        """Get total number of API keys.
        
        Returns:
            int: Total number of keys
        """
        return len(self._keys)
    
    def rotate_to_next_key(self) -> bool:
        """Rotate to the next API key (circular / round-robin).
        
        Increments the consecutive-failure counter. Returns False only after
        2 full cycles (``_MAX_CYCLES * total_keys``) of consecutive failures.
        Calling ``reset_failed_keys()`` resets the counter.
        
        Returns:
            bool: True if rotated successfully, False if cycles exhausted
        """
        with self._lock:
            if not self._keys:
                return False

            self._consecutive_failures += 1
            max_attempts = self._MAX_CYCLES * len(self._keys)

            if self._consecutive_failures >= max_attempts:
                logger.error(
                    f"All API keys exhausted after {self._MAX_CYCLES} full cycle(s) "
                    f"({self._consecutive_failures} consecutive failures)."
                )
                self._last_error = (
                    f"All API keys have exceeded their quota after {self._MAX_CYCLES} "
                    "full rotation cycle(s). Please try again later or add more keys."
                )
                return False

            # Circular rotation — wrap from last key back to first
            self._current_index = (self._current_index + 1) % len(self._keys)
            cycle_num = (self._consecutive_failures - 1) // len(self._keys) + 1
            logger.warning(
                f"API key quota error #{self._consecutive_failures}. "
                f"Rotated to key {self._current_index + 1}/{len(self._keys)} "
                f"(cycle {cycle_num}/{self._MAX_CYCLES})"
            )
            return True
    
    def mark_key_failed(self, index: int = None):
        """Increment the failure counter (same as one rotation step).
        
        Args:
            index: Ignored (kept for backward compatibility).
        """
        self.rotate_to_next_key()
    
    def reset_failed_keys(self):
        """Reset the failure counter and restart from the first key.
        
        Call this periodically (e.g., daily) to retry previously failed keys
        as their quotas may have been refreshed.
        """
        with self._lock:
            self._consecutive_failures = 0
            self._current_index = 0
            self._last_error = ""
            logger.info("API key failure counter reset — all keys available again")
    
    def has_available_keys(self) -> bool:
        """Check if we have not yet exhausted all rotation cycles.
        
        Returns:
            bool: True if more retries are allowed
        """
        with self._lock:
            if not self._keys:
                return False
            return self._consecutive_failures < self._MAX_CYCLES * len(self._keys)
    
    def get_last_error(self) -> str:
        """Get the last error message.
        
        Returns:
            str: Last error message
        """
        return self._last_error
    
    def set_last_error(self, error: str):
        """Set the last error message.
        
        Args:
            error: Error message
        """
        self._last_error = error
    
    def reset_success(self):
        """Reset the consecutive failure counter after a successful API call.
        
        Call this whenever a key succeeds so the cycle counter starts fresh.
        """
        with self._lock:
            if self._consecutive_failures > 0:
                logger.debug(f"API key {self._current_index + 1} succeeded — resetting failure counter.")
                self._consecutive_failures = 0

    def get_status(self) -> dict:
        """Get the current status of the key manager.
        
        Returns:
            dict: Status information including total keys, current key index,
                  consecutive failures, and availability
        """
        with self._lock:
            total = len(self._keys)
            max_attempts = self._MAX_CYCLES * total
            return {
                "total_keys": total,
                "current_key_index": self._current_index + 1,  # 1-indexed for display
                "consecutive_failures": self._consecutive_failures,
                "max_attempts_before_give_up": max_attempts,
                "has_available_keys": self._consecutive_failures < max_attempts,
                "last_error": self._last_error
            }


if __name__ == "__main__":
    # Example usage
    import os
    
    # Set test keys
    os.environ["GEMINI_API_KEYS"] = "key1,key2,key3"
    
    # Reset singleton for testing
    reset_api_key_manager()
    
    manager = get_api_key_manager()
    print(f"Status: {manager.get_status()}")
    print(f"Current key: {manager.get_current_key()}")
    
    # Simulate quota errors — should allow 2 * 3 = 6 rotations before giving up
    print("\nSimulating quota errors (expect 6 rotations before exhaustion)...")
    for i in range(8):
        if manager.rotate_to_next_key():
            print(f"  Rotation {i+1}: now on key '{manager.get_current_key()}'")
        else:
            print(f"  Rotation {i+1}: All keys exhausted after {i} attempts!")
            break
    
    print(f"\nFinal status: {manager.get_status()}")
