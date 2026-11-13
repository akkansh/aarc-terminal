"""Thread-safe persistent disk cache with TTL support."""

import hashlib
import json
import os
import pickle
import threading
import time
from pathlib import Path
from typing import Any, Optional


class DiskCache:
    """A thread-safe persistent file-based key-value cache with optional TTL."""

    def __init__(self, cache_dir: str = ".cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _get_path(self, key: str) -> Path:
        """Hash key to avoid invalid filename characters."""
        key_hash = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{key_hash}.pkl"

    def get(self, key: str) -> Optional[Any]:
        """Retrieve a cached value if it exists and has not expired."""
        filepath = self._get_path(key)
        if not filepath.exists():
            return None

        with self._lock:
            try:
                with open(filepath, "rb") as f:
                    entry = pickle.load(f)

                expires_at = entry.get("expires_at")
                if expires_at is not None and time.time() > expires_at:
                    # Expired, clean up
                    try:
                        filepath.unlink(missing_ok=True)
                    except OSError:
                        pass
                    return None

                return entry.get("data")
            except Exception:
                return None

    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = 3600) -> None:
        """Store a value with an optional TTL in seconds."""
        filepath = self._get_path(key)
        expires_at = time.time() + ttl_seconds if ttl_seconds is not None else None

        entry = {
            "key": key,
            "created_at": time.time(),
            "expires_at": expires_at,
            "data": value,
        }

        with self._lock:
            temp_path = filepath.with_suffix(".tmp")
            try:
                with open(temp_path, "wb") as f:
                    pickle.dump(entry, f, protocol=pickle.HIGHEST_PROTOCOL)
                os.replace(temp_path, filepath)
            except Exception:
                if temp_path.exists():
                    try:
                        temp_path.unlink()
                    except OSError:
                        pass

    def delete(self, key: str) -> None:
        """Delete an entry from cache."""
        filepath = self._get_path(key)
        with self._lock:
            try:
                filepath.unlink(missing_ok=True)
            except OSError:
                pass

    def clear(self) -> None:
        """Clear all entries in cache directory."""
        with self._lock:
            for item in self.cache_dir.glob("*.pkl"):
                try:
                    item.unlink(missing_ok=True)
                except OSError:
                    pass


# Global default instance
_DEFAULT_CACHE: Optional[DiskCache] = None
_CACHE_LOCK = threading.Lock()


def get_disk_cache(cache_dir: str = ".cache") -> DiskCache:
    """Get or create singleton DiskCache."""
    global _DEFAULT_CACHE
    if _DEFAULT_CACHE is None:
        with _CACHE_LOCK:
            if _DEFAULT_CACHE is None:
                _DEFAULT_CACHE = DiskCache(cache_dir=cache_dir)
    return _DEFAULT_CACHE
