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
    """Manages multiple Gemini API keys with automatic rotation on quota errors.
    
    This class loads API keys from environment variables and provides
    automatic rotation when a key hits its quota limit.
    
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
            # All keys exhausted
            handle_error()
    """
    
    def __init__(self):
        """Initialize the API key manager."""
        self._keys: List[str] = []
        self._current_index: int = 0
        self._failed_keys: Set[int] = set()
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
        """Rotate to the next available API key.
        
        Marks the current key as failed and switches to the next available key.
        
        Returns:
            bool: True if rotation succeeded, False if all keys exhausted
        """
        with self._lock:
            if not self._keys:
                return False
            
            # Mark current key as failed
            self._failed_keys.add(self._current_index)
            logger.warning(f"API key {self._current_index + 1}/{len(self._keys)} marked as failed (quota exceeded)")
            
            # Find next available key
            for i in range(len(self._keys)):
                next_index = (self._current_index + 1 + i) % len(self._keys)
                if next_index not in self._failed_keys:
                    self._current_index = next_index
                    logger.info(f"Rotated to API key {self._current_index + 1}/{len(self._keys)}")
                    return True
            
            # All keys exhausted
            logger.error("All API keys have exceeded their quota!")
            self._last_error = "All API keys have exceeded their quota. Please try again later or add more keys."
            return False
    
    def mark_key_failed(self, index: int = None):
        """Mark a specific key as failed.
        
        Args:
            index: Key index to mark as failed. If None, marks current key.
        """
        with self._lock:
            if index is None:
                index = self._current_index
            if 0 <= index < len(self._keys):
                self._failed_keys.add(index)
                logger.warning(f"API key {index + 1}/{len(self._keys)} manually marked as failed")
    
    def reset_failed_keys(self):
        """Reset all failed keys, making them available again.
        
        Call this periodically (e.g., daily) to retry previously failed keys
        as their quotas may have been refreshed.
        """
        with self._lock:
            self._failed_keys.clear()
            self._current_index = 0
            self._last_error = ""
            logger.info("All API keys reset and available")
    
    def has_available_keys(self) -> bool:
        """Check if any API keys are available (not failed).
        
        Returns:
            bool: True if at least one key is available
        """
        with self._lock:
            return len(self._failed_keys) < len(self._keys)
    
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
    
    def get_status(self) -> dict:
        """Get the current status of the key manager.
        
        Returns:
            dict: Status information including total keys, current key index,
                  failed keys count, and availability
        """
        with self._lock:
            return {
                "total_keys": len(self._keys),
                "current_key_index": self._current_index + 1,  # 1-indexed for display
                "failed_keys_count": len(self._failed_keys),
                "available_keys_count": len(self._keys) - len(self._failed_keys),
                "has_available_keys": len(self._failed_keys) < len(self._keys),
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
    
    # Simulate quota errors
    print("\nSimulating quota errors...")
    for i in range(4):
        if manager.rotate_to_next_key():
            print(f"Rotated to key: {manager.get_current_key()}")
        else:
            print("All keys exhausted!")
            break
    
    print(f"\nFinal status: {manager.get_status()}")
