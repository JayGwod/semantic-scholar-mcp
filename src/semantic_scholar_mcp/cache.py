"""In-memory TTL cache for API responses."""

import hashlib
import json
import threading
import time
from dataclasses import dataclass
from typing import Any

from semantic_scholar_mcp.logging_config import get_logger

logger = get_logger("cache")


@dataclass
class CacheEntry:
    """A cached value with expiration time.

    Attributes:
        value: The cached response data.
        expires_at: Timestamp when this entry expires (using monotonic time).
    """

    value: dict[str, Any]
    expires_at: float

    @property
    def is_expired(self) -> bool:
        """Check if this cache entry has expired."""
        return time.monotonic() > self.expires_at


@dataclass
class CacheConfig:
    """Cache configuration.

    Attributes:
        enabled: Whether caching is enabled.
        default_ttl: Default time-to-live in seconds.
        paper_details_ttl: TTL for paper details in seconds.
        search_ttl: TTL for search results in seconds.
        max_entries: Maximum number of cached entries.
    """

    enabled: bool = True
    default_ttl: int = 300  # 5 minutes
    paper_details_ttl: int = 3600  # 1 hour for paper details
    search_ttl: int = 300  # 5 minutes for search results
    max_entries: int = 1000  # Max cached entries


class ResponseCache:
    """Thread-safe in-memory cache with TTL support.

    Features:
    - TTL-based expiration
    - LRU eviction when max entries reached
    - Thread-safe operations
    - Endpoint-specific TTLs
    """

    def __init__(self, config: CacheConfig | None = None) -> None:
        """Initialize the response cache.

        Args:
            config: Cache configuration. Uses defaults if not provided.
        """
        self._config = config or CacheConfig()
        self._cache: dict[str, CacheEntry] = {}
        self._access_order: list[str] = []  # For LRU eviction
        self._lock = threading.Lock()
        self._stats = {"hits": 0, "misses": 0}

    @staticmethod
    def _make_key(endpoint: str, params: dict[str, Any] | None = None) -> str:
        """Generate cache key from endpoint and params using SHA256 hash."""
        key_data = {"endpoint": endpoint, "params": params or {}}
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()[:16]

    def get(self, endpoint: str, params: dict[str, Any] | None = None) -> dict[str, Any] | None:
        """Get cached response if available and not expired.

        Args:
            endpoint: API endpoint.
            params: Request parameters.

        Returns:
            Cached response or None if not found/expired.
        """
        if not self._config.enabled:
            return None

        key = self._make_key(endpoint, params)

        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                self._stats["misses"] += 1
                return None

            if entry.is_expired:
                del self._cache[key]
                if key in self._access_order:
                    self._access_order.remove(key)
                self._stats["misses"] += 1
                return None

            # Update access order for LRU
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)

            self._stats["hits"] += 1
            logger.debug("Cache hit for %s", endpoint)
            return entry.value

    def set(
        self,
        endpoint: str,
        params: dict[str, Any] | None,
        value: dict[str, Any],
        ttl: int | None = None,
    ) -> None:
        """Cache a response.

        Args:
            endpoint: API endpoint.
            params: Request parameters.
            value: Response to cache.
            ttl: Time-to-live in seconds (uses endpoint-specific default if not specified).
        """
        if not self._config.enabled:
            return

        key = self._make_key(endpoint, params)

        # Determine TTL based on endpoint
        if ttl is None:
            if "/paper/" in endpoint and "/search" not in endpoint:
                ttl = self._config.paper_details_ttl
            else:
                ttl = self._config.search_ttl

        expires_at = time.monotonic() + ttl

        with self._lock:
            # Evict if at capacity (LRU)
            while len(self._cache) >= self._config.max_entries:
                if self._access_order:
                    oldest_key = self._access_order.pop(0)
                    self._cache.pop(oldest_key, None)

            self._cache[key] = CacheEntry(value=value, expires_at=expires_at)
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)

            logger.debug("Cached response for %s (ttl=%ds)", endpoint, ttl)

    def clear(self) -> None:
        """Clear all cached entries."""
        with self._lock:
            self._cache.clear()
            self._access_order.clear()
            self._stats = {"hits": 0, "misses": 0}

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with entries, hits, misses, and hit_rate.
        """
        with self._lock:
            total = self._stats["hits"] + self._stats["misses"]
            hit_rate = self._stats["hits"] / total if total > 0 else 0.0
            return {
                "entries": len(self._cache),
                "hits": self._stats["hits"],
                "misses": self._stats["misses"],
                "hit_rate": hit_rate,
            }
