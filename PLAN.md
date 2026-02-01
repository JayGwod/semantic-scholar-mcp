# Semantic Scholar MCP - Comprehensive Improvement Plan

## Summary
This plan addresses 8 weak spots across 5 phases. All changes include type checking (`ty`), formatting (`ruff`), and testing (unit + integration with real API).

**Commands used throughout:**
```bash
uv run pytest tests/ -v                    # Run unit tests
uv run pytest tests/ -v -m integration     # Run integration tests
uv run ruff check src/ tests/              # Lint
uv run ruff format src/ tests/             # Format
uv run ty check src/                       # Type check
```

---

## Phase 1: Foundational Infrastructure

### 1.1 Add Structured Logging Framework

**New file:** `src/semantic_scholar_mcp/logging_config.py`

```python
"""Centralized logging configuration for Semantic Scholar MCP."""

import logging
import sys
from typing import Literal

from semantic_scholar_mcp.config import settings

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


def setup_logging(
    level: LogLevel | None = None,
    format_style: Literal["simple", "detailed"] = "simple",
) -> logging.Logger:
    """Configure logging for the application.

    Args:
        level: Log level (defaults to settings.log_level or INFO)
        format_style: "simple" for basic, "detailed" for timestamps + module

    Returns:
        Configured root logger for semantic_scholar_mcp
    """
    log_level = level or getattr(settings, "log_level", "INFO")

    if format_style == "detailed":
        fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    else:
        fmt = "[%(levelname)s] %(name)s: %(message)s"

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(fmt))

    logger = logging.getLogger("semantic_scholar_mcp")
    logger.setLevel(getattr(logging, log_level))
    logger.addHandler(handler)
    logger.propagate = False

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger for a specific module.

    Args:
        name: Module name (use __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(f"semantic_scholar_mcp.{name}")
```

**Modify:** `src/semantic_scholar_mcp/config.py`

Add these settings (lines ~28-30):
```python
# Logging configuration
log_level: str = os.getenv("SS_LOG_LEVEL", "INFO")
log_format: str = os.getenv("SS_LOG_FORMAT", "simple")  # "simple" or "detailed"
```

**Add logging to modules:**

| File | Add logging for |
|------|-----------------|
| `paper_tracker.py` | `track()`, `track_many()`, `clear()` operations |
| `server.py` | Tool execution start/end with timing |
| `bibtex.py` | Export operations, entry generation |
| `client.py` | Already has logging, add DEBUG level for request/response bodies |

**Example for `paper_tracker.py` (add at top):**
```python
from semantic_scholar_mcp.logging_config import get_logger

logger = get_logger("paper_tracker")

# In track() method, add:
logger.debug("Tracking paper: %s (source: %s)", paper.paperId, source_tool)

# In clear() method, add:
logger.info("Cleared %d tracked papers", count)
```

---

### 1.2 Improve Error Handling in Client

**Modify:** `src/semantic_scholar_mcp/exceptions.py`

Add new exception classes (after line 49):
```python
class ServerError(SemanticScholarError):
    """Raised when the Semantic Scholar API returns a 5xx server error.

    This typically indicates a temporary issue with the API servers.
    These errors are often transient and may succeed on retry.
    """

    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


class AuthenticationError(SemanticScholarError):
    """Raised when API key authentication fails.

    This indicates the API key is invalid, expired, or lacks required permissions.
    """
    pass


class ConnectionError(SemanticScholarError):
    """Raised when unable to connect to the Semantic Scholar API.

    This may indicate network issues, DNS failures, or API unavailability.
    """
    pass
```

**Modify:** `src/semantic_scholar_mcp/client.py`

Update `_handle_response()` method (lines 82-133) to:

```python
async def _handle_response(self, response: httpx.Response, endpoint: str) -> dict[str, Any]:
    """Handle API response and raise appropriate exceptions.

    Args:
        response: The httpx response object
        endpoint: The API endpoint for error context

    Returns:
        Parsed JSON response data

    Raises:
        RateLimitError: When rate limit is exceeded (HTTP 429)
        NotFoundError: When resource is not found (HTTP 404)
        AuthenticationError: When API key is invalid (HTTP 401/403)
        ServerError: When server returns 5xx error
        SemanticScholarError: For other HTTP errors
    """
    logger.info(
        "API response: method=%s endpoint=%s status=%d",
        response.request.method,
        endpoint,
        response.status_code,
    )

    # Success
    if response.status_code < 400:
        return response.json()

    # Rate limit exceeded
    if response.status_code == 429:
        retry_after: float | None = None
        if "Retry-After" in response.headers:
            try:
                retry_after = float(response.headers["Retry-After"])
            except ValueError:
                pass
        raise RateLimitError(
            f"Rate limit exceeded for {endpoint}. "
            "Consider using an API key for higher limits. "
            "See: https://www.semanticscholar.org/product/api#api-key",
            retry_after=retry_after,
        )

    # Authentication errors
    if response.status_code in (401, 403):
        raise AuthenticationError(
            f"Authentication failed for {endpoint}. "
            "Please verify your API key is valid and has the required permissions."
        )

    # Not found
    if response.status_code == 404:
        raise NotFoundError(
            f"Resource not found: {endpoint}. "
            "For DOIs, use format 'DOI:10.xxxx/xxxxx'. "
            "For ArXiv IDs, use format 'ARXIV:xxxx.xxxxx'."
        )

    # Server errors (5xx) - these are retriable
    if 500 <= response.status_code < 600:
        raise ServerError(
            f"Semantic Scholar API server error ({response.status_code}) for {endpoint}. "
            "This is usually temporary. Please try again.",
            status_code=response.status_code,
        )

    # Other client errors (4xx)
    raise SemanticScholarError(
        f"API error ({response.status_code}) for {endpoint}: {response.text}"
    )
```

**Wrap HTTP calls with connection error handling** in `get()` and `post()` methods:

```python
try:
    response = await self._client.get(url, params=params)
except httpx.ConnectError as e:
    raise ConnectionError(f"Failed to connect to Semantic Scholar API: {e}") from e
except httpx.TimeoutException as e:
    raise ConnectionError(f"Request timed out: {e}") from e
```

---

## Phase 2: Resilience Improvements

### 2.1 Add Proactive Rate Limiting (Token Bucket)

**Modify:** `src/semantic_scholar_mcp/rate_limiter.py`

Add `TokenBucket` class (after `RetryConfig`):

```python
import time
import asyncio
from dataclasses import dataclass, field


@dataclass
class TokenBucket:
    """Token bucket rate limiter for proactive rate limiting.

    This prevents hitting API rate limits by controlling request frequency
    before sending requests, rather than reacting to 429 errors.

    Args:
        rate: Tokens added per second
        capacity: Maximum tokens (burst size)
    """
    rate: float  # tokens per second
    capacity: float  # max burst
    _tokens: float = field(init=False)
    _last_update: float = field(init=False)
    _lock: asyncio.Lock = field(init=False, default_factory=asyncio.Lock)

    def __post_init__(self) -> None:
        self._tokens = self.capacity
        self._last_update = time.monotonic()

    async def acquire(self, tokens: float = 1.0) -> float:
        """Acquire tokens, waiting if necessary.

        Args:
            tokens: Number of tokens to acquire (default 1)

        Returns:
            Time waited in seconds (0 if no wait needed)
        """
        async with self._lock:
            now = time.monotonic()

            # Add tokens based on elapsed time
            elapsed = now - self._last_update
            self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)
            self._last_update = now

            # If enough tokens, consume immediately
            if self._tokens >= tokens:
                self._tokens -= tokens
                return 0.0

            # Calculate wait time for needed tokens
            needed = tokens - self._tokens
            wait_time = needed / self.rate

            # Wait and then consume
            await asyncio.sleep(wait_time)
            self._tokens = 0  # Consumed all available + waited for rest
            self._last_update = time.monotonic()

            return wait_time


# Pre-configured rate limiters for Semantic Scholar API
def create_rate_limiter(has_api_key: bool) -> TokenBucket:
    """Create appropriate rate limiter based on authentication status.

    Args:
        has_api_key: Whether an API key is configured

    Returns:
        TokenBucket configured for the appropriate rate limit
    """
    if has_api_key:
        # With API key: 1 request per second (dedicated)
        return TokenBucket(rate=1.0, capacity=1.0)
    else:
        # Without API key: ~16.67 requests per second (5000 per 5 min, shared)
        # Use conservative estimate accounting for shared pool
        return TokenBucket(rate=10.0, capacity=20.0)
```

**Modify:** `src/semantic_scholar_mcp/client.py`

Integrate token bucket (add to `__init__` and use in `get()`/`post()`):

```python
from semantic_scholar_mcp.rate_limiter import TokenBucket, create_rate_limiter

class SemanticScholarClient:
    def __init__(self) -> None:
        # ... existing code ...
        self._rate_limiter: TokenBucket = create_rate_limiter(settings.has_api_key)

    async def get(self, endpoint: str, ...) -> dict[str, Any]:
        # Add before making request:
        wait_time = await self._rate_limiter.acquire()
        if wait_time > 0:
            logger.debug("Rate limiter: waited %.2fs before request", wait_time)
        # ... rest of existing code ...
```

---

### 2.2 Add Circuit Breaker Pattern

**New file:** `src/semantic_scholar_mcp/circuit_breaker.py`

```python
"""Circuit breaker pattern for API resilience."""

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine, TypeVar

from semantic_scholar_mcp.logging_config import get_logger

logger = get_logger("circuit_breaker")

T = TypeVar("T")


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing fast, not making requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""
    failure_threshold: int = 5      # Failures before opening
    recovery_timeout: float = 30.0  # Seconds before testing recovery
    half_open_max_calls: int = 1    # Test calls in half-open state


@dataclass
class CircuitBreaker:
    """Circuit breaker to prevent hammering a failing service.

    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Service considered down, requests fail immediately
    - HALF_OPEN: Testing if service recovered
    """
    config: CircuitBreakerConfig = field(default_factory=CircuitBreakerConfig)
    _state: CircuitState = field(init=False, default=CircuitState.CLOSED)
    _failure_count: int = field(init=False, default=0)
    _last_failure_time: float = field(init=False, default=0.0)
    _half_open_calls: int = field(init=False, default=0)
    _lock: asyncio.Lock = field(init=False, default_factory=asyncio.Lock)

    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        return self._state

    async def call(
        self,
        func: Callable[..., Coroutine[Any, Any, T]],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """Execute function with circuit breaker protection.

        Args:
            func: Async function to execute
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func

        Returns:
            Result of func

        Raises:
            CircuitOpenError: If circuit is open
            Exception: Any exception from func (also recorded as failure)
        """
        async with self._lock:
            await self._check_state_transition()

            if self._state == CircuitState.OPEN:
                raise CircuitOpenError(
                    "Circuit breaker is open. Service appears to be down."
                )

        try:
            result = await func(*args, **kwargs)
            await self._record_success()
            return result
        except Exception as e:
            await self._record_failure()
            raise

    async def _check_state_transition(self) -> None:
        """Check if state should transition based on time."""
        if self._state == CircuitState.OPEN:
            elapsed = time.monotonic() - self._last_failure_time
            if elapsed >= self.config.recovery_timeout:
                logger.info("Circuit breaker transitioning to HALF_OPEN after %.1fs", elapsed)
                self._state = CircuitState.HALF_OPEN
                self._half_open_calls = 0

    async def _record_success(self) -> None:
        """Record a successful call."""
        async with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                logger.info("Circuit breaker: test call succeeded, closing circuit")
                self._state = CircuitState.CLOSED
            self._failure_count = 0

    async def _record_failure(self) -> None:
        """Record a failed call."""
        async with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.monotonic()

            if self._state == CircuitState.HALF_OPEN:
                logger.warning("Circuit breaker: test call failed, reopening circuit")
                self._state = CircuitState.OPEN
            elif self._failure_count >= self.config.failure_threshold:
                logger.warning(
                    "Circuit breaker: %d failures, opening circuit",
                    self._failure_count
                )
                self._state = CircuitState.OPEN

    def reset(self) -> None:
        """Reset circuit breaker to initial state (for testing)."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time = 0.0
        self._half_open_calls = 0


class CircuitOpenError(Exception):
    """Raised when circuit breaker is open."""
    pass
```

**Add to config.py:**
```python
# Circuit breaker configuration
circuit_failure_threshold: int = int(os.getenv("SS_CIRCUIT_FAILURE_THRESHOLD", "5"))
circuit_recovery_timeout: float = float(os.getenv("SS_CIRCUIT_RECOVERY_TIMEOUT", "30.0"))
```

---

## Phase 3: Caching Layer

### 3.1 Add In-Memory TTL Cache

**New file:** `src/semantic_scholar_mcp/cache.py`

```python
"""In-memory TTL cache for API responses."""

import hashlib
import json
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from semantic_scholar_mcp.logging_config import get_logger

logger = get_logger("cache")


@dataclass
class CacheEntry:
    """A cached value with expiration time."""
    value: dict[str, Any]
    expires_at: float

    @property
    def is_expired(self) -> bool:
        return time.monotonic() > self.expires_at


@dataclass
class CacheConfig:
    """Cache configuration."""
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
        self._config = config or CacheConfig()
        self._cache: dict[str, CacheEntry] = {}
        self._access_order: list[str] = []  # For LRU eviction
        self._lock = threading.Lock()
        self._stats = {"hits": 0, "misses": 0}

    @staticmethod
    def _make_key(endpoint: str, params: dict[str, Any] | None = None) -> str:
        """Generate cache key from endpoint and params."""
        key_data = {"endpoint": endpoint, "params": params or {}}
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()[:16]

    def get(self, endpoint: str, params: dict[str, Any] | None = None) -> dict[str, Any] | None:
        """Get cached response if available and not expired.

        Args:
            endpoint: API endpoint
            params: Request parameters

        Returns:
            Cached response or None if not found/expired
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
            endpoint: API endpoint
            params: Request parameters
            value: Response to cache
            ttl: Time-to-live in seconds (uses default if not specified)
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
            # Evict if at capacity
            while len(self._cache) >= self._config.max_entries:
                if self._access_order:
                    oldest_key = self._access_order.pop(0)
                    self._cache.pop(oldest_key, None)

            self._cache[key] = CacheEntry(value=value, expires_at=expires_at)
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)

            logger.debug("Cached response for %s (ttl=%ds)", endpoint, ttl)

    def invalidate(self, endpoint_pattern: str) -> int:
        """Invalidate cached entries matching pattern.

        Args:
            endpoint_pattern: Pattern to match (e.g., "/paper/")

        Returns:
            Number of entries invalidated
        """
        with self._lock:
            # This is a simple implementation - in production you might want
            # to store endpoint with the key for pattern matching
            count = len(self._cache)
            self._cache.clear()
            self._access_order.clear()
            return count

    def clear(self) -> None:
        """Clear all cached entries."""
        with self._lock:
            self._cache.clear()
            self._access_order.clear()
            self._stats = {"hits": 0, "misses": 0}

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total = self._stats["hits"] + self._stats["misses"]
            hit_rate = self._stats["hits"] / total if total > 0 else 0.0
            return {
                "entries": len(self._cache),
                "hits": self._stats["hits"],
                "misses": self._stats["misses"],
                "hit_rate": hit_rate,
            }


# Global cache instance
_cache: ResponseCache | None = None
_cache_lock = threading.Lock()


def get_cache() -> ResponseCache:
    """Get or create the global cache instance."""
    global _cache
    if _cache is None:
        with _cache_lock:
            if _cache is None:
                from semantic_scholar_mcp.config import settings
                config = CacheConfig(
                    enabled=getattr(settings, "cache_enabled", True),
                    default_ttl=getattr(settings, "cache_ttl", 300),
                )
                _cache = ResponseCache(config)
    return _cache
```

**Add to config.py:**
```python
# Cache configuration
cache_enabled: bool = os.getenv("SS_CACHE_ENABLED", "true").lower() == "true"
cache_ttl: int = int(os.getenv("SS_CACHE_TTL", "300"))
cache_paper_ttl: int = int(os.getenv("SS_CACHE_PAPER_TTL", "3600"))
```

**Integrate with client.py** (in `get()` method):
```python
from semantic_scholar_mcp.cache import get_cache

async def get(self, endpoint: str, params: dict[str, Any] | None = None, ...) -> dict[str, Any]:
    cache = get_cache()

    # Check cache first
    cached = cache.get(endpoint, params)
    if cached is not None:
        return cached

    # ... make API request ...

    # Cache successful response
    result = await self._handle_response(response, endpoint)
    cache.set(endpoint, params, result)
    return result
```

---

## Phase 4: Server Refactoring

### 4.1 New Directory Structure

```
src/semantic_scholar_mcp/
├── __init__.py
├── server.py              # FastMCP init, client management, main()
├── config.py              # (existing)
├── client.py              # (existing, modified)
├── models.py              # (existing)
├── exceptions.py          # (modified)
├── rate_limiter.py        # (modified)
├── circuit_breaker.py     # (new)
├── cache.py               # (new)
├── logging_config.py      # (new)
├── paper_tracker.py       # (existing)
├── bibtex.py              # (existing)
└── tools/
    ├── __init__.py        # Exports all tools
    ├── _common.py         # Shared constants, get_client, get_tracker
    ├── papers.py          # search_papers, get_paper_details, citations, references
    ├── authors.py         # search_authors, get_author_details, find_duplicates, consolidate
    ├── recommendations.py # get_recommendations, get_related_papers
    └── tracking.py        # list_tracked_papers, clear_tracked_papers, export_bibtex
```

### 4.2 Detailed File Contents

**New file:** `src/semantic_scholar_mcp/tools/__init__.py`

```python
"""MCP tools for Semantic Scholar API."""

from semantic_scholar_mcp.tools.papers import (
    search_papers,
    get_paper_details,
    get_paper_citations,
    get_paper_references,
)
from semantic_scholar_mcp.tools.authors import (
    search_authors,
    get_author_details,
    find_duplicate_authors,
    consolidate_authors,
)
from semantic_scholar_mcp.tools.recommendations import (
    get_recommendations,
    get_related_papers,
)
from semantic_scholar_mcp.tools.tracking import (
    list_tracked_papers,
    clear_tracked_papers,
    export_bibtex,
)

__all__ = [
    # Papers
    "search_papers",
    "get_paper_details",
    "get_paper_citations",
    "get_paper_references",
    # Authors
    "search_authors",
    "get_author_details",
    "find_duplicate_authors",
    "consolidate_authors",
    # Recommendations
    "get_recommendations",
    "get_related_papers",
    # Tracking
    "list_tracked_papers",
    "clear_tracked_papers",
    "export_bibtex",
]
```

**New file:** `src/semantic_scholar_mcp/tools/_common.py`

```python
"""Shared utilities for MCP tools."""

from semantic_scholar_mcp.client import SemanticScholarClient
from semantic_scholar_mcp.paper_tracker import get_tracker as _get_tracker, PaperTracker

# Default fields to request from the API for comprehensive paper data
DEFAULT_PAPER_FIELDS = (
    "paperId,title,abstract,year,citationCount,authors,venue,"
    "publicationTypes,openAccessPdf,fieldsOfStudy,journal,externalIds,"
    "publicationDate,publicationVenue"
)

# Default fields to request from the API for comprehensive author data
DEFAULT_AUTHOR_FIELDS = (
    "authorId,name,affiliations,paperCount,citationCount,hIndex,externalIds,aliases,homepage"
)

# Fields to request when TLDR is included
PAPER_FIELDS_WITH_TLDR = f"{DEFAULT_PAPER_FIELDS},tldr"


# Re-export for tools to use
def get_tracker() -> PaperTracker:
    """Get the paper tracker instance."""
    return _get_tracker()


# Client accessor - this will be set by server.py
_get_client_func: callable | None = None


def set_client_getter(func: callable) -> None:
    """Set the function to get the client (called by server.py)."""
    global _get_client_func
    _get_client_func = func


def get_client() -> SemanticScholarClient:
    """Get the shared client instance."""
    if _get_client_func is None:
        raise RuntimeError("Client getter not configured. Call set_client_getter first.")
    return _get_client_func()
```

**New file:** `src/semantic_scholar_mcp/tools/papers.py`

Contains: `search_papers`, `get_paper_details`, `get_paper_citations`, `get_paper_references`

- Extract lines 96-409 from original `server.py`
- Update imports to use `from semantic_scholar_mcp.tools._common import ...`
- Remove `@mcp.tool()` decorators (these will be applied in server.py)

**New file:** `src/semantic_scholar_mcp/tools/authors.py`

Contains: `search_authors`, `get_author_details`, `find_duplicate_authors`, `consolidate_authors`

- Extract lines 412-1024 from original `server.py`
- Update imports similarly

**New file:** `src/semantic_scholar_mcp/tools/recommendations.py`

Contains: `get_recommendations`, `get_related_papers`

- Extract lines 552-742 from original `server.py`
- Update imports similarly

**New file:** `src/semantic_scholar_mcp/tools/tracking.py`

Contains: `list_tracked_papers`, `clear_tracked_papers`, `export_bibtex`

- Extract lines 1027-1191 from original `server.py`
- Update imports similarly

### 4.3 Refactored server.py

**Modified:** `src/semantic_scholar_mcp/server.py` (~100 lines)

```python
"""FastMCP server for Semantic Scholar API.

This module initializes the MCP server and registers all tools.
"""

import atexit
import asyncio
import threading

from fastmcp import FastMCP

from semantic_scholar_mcp.client import SemanticScholarClient
from semantic_scholar_mcp.logging_config import setup_logging, get_logger
from semantic_scholar_mcp.tools._common import set_client_getter

# Import all tools for registration
from semantic_scholar_mcp.tools import (
    search_papers,
    get_paper_details,
    get_paper_citations,
    get_paper_references,
    search_authors,
    get_author_details,
    find_duplicate_authors,
    consolidate_authors,
    get_recommendations,
    get_related_papers,
    list_tracked_papers,
    clear_tracked_papers,
    export_bibtex,
)

# Initialize logging
setup_logging()
logger = get_logger("server")

# Initialize the MCP server
mcp = FastMCP(
    name="semantic-scholar",
    instructions="Search and analyze academic papers through Semantic Scholar API",
)

# Shared client instance with thread-safe initialization
_client: SemanticScholarClient | None = None
_client_lock = threading.Lock()


def get_client() -> SemanticScholarClient:
    """Get or create the shared client instance (thread-safe)."""
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                _client = SemanticScholarClient()
                logger.info("Created new SemanticScholarClient")
    return _client


# Configure tools to use our client getter
set_client_getter(get_client)


def _cleanup_client() -> None:
    """Clean up the shared client instance on exit."""
    global _client
    if _client is not None:
        try:
            loop = asyncio.new_event_loop()
            loop.run_until_complete(_client.close())
            loop.close()
            logger.info("Client closed successfully")
        except Exception:
            pass
        _client = None


atexit.register(_cleanup_client)


# Register all tools with the MCP server
mcp.tool()(search_papers)
mcp.tool()(get_paper_details)
mcp.tool()(get_paper_citations)
mcp.tool()(get_paper_references)
mcp.tool()(search_authors)
mcp.tool()(get_author_details)
mcp.tool()(find_duplicate_authors)
mcp.tool()(consolidate_authors)
mcp.tool()(get_recommendations)
mcp.tool()(get_related_papers)
mcp.tool()(list_tracked_papers)
mcp.tool()(clear_tracked_papers)
mcp.tool()(export_bibtex)

logger.info("Registered %d MCP tools", 13)


def main() -> None:
    """Run the MCP server."""
    logger.info("Starting Semantic Scholar MCP server")
    mcp.run()


if __name__ == "__main__":
    main()
```

---

## Phase 5: Testing

### 5.1 Fix Test Isolation

**Modify:** `tests/conftest.py`

Add global fixtures (after line 94):

```python
from semantic_scholar_mcp.paper_tracker import PaperTracker
from semantic_scholar_mcp.cache import get_cache


@pytest.fixture(autouse=True)
def reset_tracker() -> Generator[None]:
    """Reset the paper tracker singleton before each test."""
    PaperTracker.reset_instance()
    yield
    PaperTracker.reset_instance()


@pytest.fixture(autouse=True)
def reset_cache() -> Generator[None]:
    """Reset the cache before each test."""
    cache = get_cache()
    cache.clear()
    yield
    cache.clear()
```

### 5.2 Add Integration Tests

**New file:** `tests/test_integration.py`

```python
"""Integration tests using real Semantic Scholar API.

These tests hit the actual API and verify end-to-end functionality.
Run with: uv run pytest tests/test_integration.py -v -m integration
"""

import asyncio
import pytest

from semantic_scholar_mcp.client import SemanticScholarClient
from semantic_scholar_mcp.paper_tracker import get_tracker, PaperTracker
from semantic_scholar_mcp.tools.papers import search_papers, get_paper_details
from semantic_scholar_mcp.tools._common import set_client_getter


# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="module")
async def real_client():
    """Create a real client for integration tests."""
    client = SemanticScholarClient()
    set_client_getter(lambda: client)
    yield client
    await client.close()


@pytest.fixture(autouse=True)
def reset_tracker_integration():
    """Reset tracker between integration tests."""
    PaperTracker.reset_instance()
    yield
    PaperTracker.reset_instance()


class TestSearchIntegration:
    """Integration tests for paper search."""

    @pytest.mark.asyncio
    async def test_search_real_papers(self, real_client: SemanticScholarClient) -> None:
        """Test searching for a known paper returns results."""
        result = await search_papers("attention is all you need", limit=5)

        assert isinstance(result, list)
        assert len(result) > 0
        assert any("attention" in p.title.lower() for p in result)

    @pytest.mark.asyncio
    async def test_search_with_year_filter(self, real_client: SemanticScholarClient) -> None:
        """Test search with year filter."""
        result = await search_papers(
            "transformer neural network",
            year="2020-2024",
            limit=5
        )

        assert isinstance(result, list)
        if result:  # May be empty for very specific queries
            assert all(2020 <= (p.year or 0) <= 2024 for p in result)


class TestPaperDetailsIntegration:
    """Integration tests for paper details."""

    @pytest.mark.asyncio
    async def test_get_known_paper(self, real_client: SemanticScholarClient) -> None:
        """Test fetching a known paper by ID."""
        # "Attention Is All You Need" paper ID
        paper_id = "204e3073870fae3d05bcbc2f6a8e263d9b72e776"
        result = await get_paper_details(paper_id)

        assert not isinstance(result, str)  # Not an error message
        assert "attention" in result.title.lower()

    @pytest.mark.asyncio
    async def test_get_paper_by_doi(self, real_client: SemanticScholarClient) -> None:
        """Test fetching paper by DOI."""
        result = await get_paper_details("DOI:10.48550/arXiv.1706.03762")

        # May return paper or "not found" message
        if not isinstance(result, str):
            assert result.title is not None


class TestWorkflowIntegration:
    """Integration tests for complete workflows."""

    @pytest.mark.asyncio
    async def test_search_track_workflow(self, real_client: SemanticScholarClient) -> None:
        """Test searching papers and tracking them."""
        # Search for papers
        result = await search_papers("BERT language model", limit=3)

        assert isinstance(result, list)
        assert len(result) > 0

        # Check papers are tracked
        tracker = get_tracker()
        tracked = tracker.get_all_papers()

        assert len(tracked) == len(result)
        assert all(p.paperId in [t.paperId for t in tracked] for p in result)


class TestRateLimitIntegration:
    """Integration tests for rate limit handling."""

    @pytest.mark.asyncio
    async def test_multiple_requests_succeed(self, real_client: SemanticScholarClient) -> None:
        """Test that multiple sequential requests succeed with rate limiting."""
        # Make several requests in sequence
        for i in range(3):
            result = await search_papers(f"machine learning {i}", limit=2)
            assert isinstance(result, list) or "No papers found" in str(result)
            await asyncio.sleep(0.5)  # Small delay between requests
```

**Add pytest marker in `pyproject.toml`:**

```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
markers = [
    "integration: marks tests as integration tests (deselect with '-m \"not integration\"')",
]
```

### 5.3 Add Concurrency Tests

**New file:** `tests/test_concurrency.py`

```python
"""Concurrency and thread-safety tests."""

import asyncio
import threading
import pytest

from semantic_scholar_mcp.paper_tracker import get_tracker, PaperTracker
from semantic_scholar_mcp.cache import ResponseCache, CacheConfig
from semantic_scholar_mcp.models import Paper


@pytest.fixture(autouse=True)
def reset_tracker():
    """Reset tracker between tests."""
    PaperTracker.reset_instance()
    yield
    PaperTracker.reset_instance()


class TestPaperTrackerConcurrency:
    """Thread-safety tests for paper tracker."""

    def test_concurrent_tracking(self) -> None:
        """Test tracking papers from multiple threads."""
        tracker = get_tracker()
        papers_per_thread = 100
        num_threads = 10

        def track_papers(thread_id: int) -> None:
            for i in range(papers_per_thread):
                paper = Paper(
                    paperId=f"paper-{thread_id}-{i}",
                    title=f"Paper {thread_id}-{i}",
                )
                tracker.track(paper, f"thread-{thread_id}")

        threads = [
            threading.Thread(target=track_papers, args=(i,))
            for i in range(num_threads)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All papers should be tracked
        assert tracker.count() == papers_per_thread * num_threads

    def test_concurrent_read_write(self) -> None:
        """Test concurrent reads and writes."""
        tracker = get_tracker()

        # Pre-populate with some papers
        for i in range(50):
            paper = Paper(paperId=f"initial-{i}", title=f"Initial {i}")
            tracker.track(paper, "setup")

        errors: list[Exception] = []

        def writer() -> None:
            try:
                for i in range(100):
                    paper = Paper(paperId=f"new-{i}", title=f"New {i}")
                    tracker.track(paper, "writer")
            except Exception as e:
                errors.append(e)

        def reader() -> None:
            try:
                for _ in range(100):
                    _ = tracker.get_all_papers()
                    _ = tracker.count()
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=writer),
            threading.Thread(target=reader),
            threading.Thread(target=reader),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


class TestCacheConcurrency:
    """Thread-safety tests for cache."""

    def test_concurrent_cache_operations(self) -> None:
        """Test concurrent cache reads and writes."""
        cache = ResponseCache(CacheConfig(max_entries=1000))
        errors: list[Exception] = []

        def writer(thread_id: int) -> None:
            try:
                for i in range(100):
                    cache.set(
                        f"/endpoint/{thread_id}",
                        {"i": i},
                        {"data": f"value-{thread_id}-{i}"},
                    )
            except Exception as e:
                errors.append(e)

        def reader(thread_id: int) -> None:
            try:
                for i in range(100):
                    _ = cache.get(f"/endpoint/{thread_id % 5}", {"i": i % 50})
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=writer, args=(i,))
            for i in range(5)
        ] + [
            threading.Thread(target=reader, args=(i,))
            for i in range(5)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
```

### 5.4 Update Existing Tests

**Modify:** `tests/test_server.py`

Update imports to use the new tool modules:

```python
from semantic_scholar_mcp.tools.papers import (
    search_papers,
    get_paper_details,
    get_paper_citations,
    get_paper_references,
)
from semantic_scholar_mcp.tools.authors import (
    search_authors,
    get_author_details,
)
from semantic_scholar_mcp.tools.recommendations import (
    get_recommendations,
    get_related_papers,
)
from semantic_scholar_mcp.tools._common import set_client_getter
```

Update test fixture to use set_client_getter:

```python
@pytest.fixture(autouse=True)
def mock_client_getter(mock_client: MagicMock) -> Generator[None]:
    """Set mock client for all tests."""
    set_client_getter(lambda: mock_client)
    yield
```

Update test methods - remove `.fn` and patch references:

```python
# Before:
with patch.object(server, "get_client", return_value=mock_client):
    result = await server.search_papers.fn("attention mechanism")

# After:
result = await search_papers("attention mechanism")
```

---

## Verification Plan

### Step 1: Format and Lint
```bash
uv run ruff format src/ tests/
uv run ruff check src/ tests/ --fix
```

### Step 2: Type Check
```bash
uv run ty check src/
```

### Step 3: Run Unit Tests
```bash
uv run pytest tests/ -v -m "not integration"
```

### Step 4: Run Integration Tests
```bash
# Requires network access to Semantic Scholar API
uv run pytest tests/test_integration.py -v -m integration
```

### Step 5: Run All Tests
```bash
uv run pytest tests/ -v
```

### Step 6: Manual Verification
```bash
# Start the server
uv run semantic-scholar-mcp

# In another terminal, test via Claude Code MCP:
# 1. search_papers("machine learning", limit=3)
# 2. Check logs show structured output
# 3. Repeat search - check cache hit in logs
# 4. get_paper_details("invalid-id") - check error handling
```

---

## Implementation Order

1. **Phase 1.1** - Logging (needed to verify other changes)
2. **Phase 1.2** - Error handling improvements
3. **Phase 2.1** - Token bucket rate limiting
4. **Phase 2.2** - Circuit breaker
5. **Phase 3.1** - Caching layer
6. **Phase 4** - Server refactoring (biggest change)
7. **Phase 5.1** - Fix test isolation
8. **Phase 5.2-5.4** - Add integration and concurrency tests, update existing tests

---

## Files Summary

### New Files (11)
| File | Lines (est.) |
|------|--------------|
| `src/semantic_scholar_mcp/logging_config.py` | ~50 |
| `src/semantic_scholar_mcp/circuit_breaker.py` | ~120 |
| `src/semantic_scholar_mcp/cache.py` | ~180 |
| `src/semantic_scholar_mcp/tools/__init__.py` | ~40 |
| `src/semantic_scholar_mcp/tools/_common.py` | ~50 |
| `src/semantic_scholar_mcp/tools/papers.py` | ~200 |
| `src/semantic_scholar_mcp/tools/authors.py` | ~350 |
| `src/semantic_scholar_mcp/tools/recommendations.py` | ~150 |
| `src/semantic_scholar_mcp/tools/tracking.py` | ~150 |
| `tests/test_integration.py` | ~150 |
| `tests/test_concurrency.py` | ~120 |

### Modified Files (9)
| File | Changes |
|------|---------|
| `src/semantic_scholar_mcp/config.py` | Add 8 new env var settings |
| `src/semantic_scholar_mcp/exceptions.py` | Add 3 new exception classes |
| `src/semantic_scholar_mcp/client.py` | Integrate cache, circuit breaker, improved errors |
| `src/semantic_scholar_mcp/rate_limiter.py` | Add TokenBucket class |
| `src/semantic_scholar_mcp/paper_tracker.py` | Add logging |
| `src/semantic_scholar_mcp/server.py` | Refactor to ~100 lines, import from tools/ |
| `tests/conftest.py` | Add global reset fixtures |
| `tests/test_server.py` | Update imports and patches |
| `pyproject.toml` | Add integration test marker |
