"""Configuration settings for Semantic Scholar MCP server."""

import os


def _parse_int_with_bounds(env_var: str, default: int, min_val: int, max_val: int) -> int:
    """Parse an integer environment variable with bounds validation.

    Args:
        env_var: Name of the environment variable.
        default: Default value if not set or invalid.
        min_val: Minimum allowed value (inclusive).
        max_val: Maximum allowed value (inclusive).

    Returns:
        The parsed and clamped integer value.
    """
    raw = os.environ.get(env_var)
    if raw is None:
        return default
    try:
        value = int(raw)
        if value < min_val:
            return min_val
        if value > max_val:
            return max_val
        return value
    except ValueError:
        return default


class Settings:
    """Configuration settings loaded from environment variables.

    Attributes:
        api_key: Optional Semantic Scholar API key for higher rate limits.
        graph_api_base_url: Base URL for the Graph API.
        recommendations_api_base_url: Base URL for the Recommendations API.
        retry_max_attempts: Maximum number of retry attempts for rate limit errors.
        retry_base_delay: Base delay in seconds for exponential backoff.
        retry_max_delay: Maximum delay in seconds between retries.
        enable_auto_retry: Whether to automatically retry on rate limit errors.
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_format: Logging format style (simple or detailed).
        circuit_failure_threshold: Number of failures before circuit breaker opens.
        circuit_recovery_timeout: Seconds before testing if service recovered.
        cache_enabled: Whether to enable response caching.
        cache_ttl: Default cache TTL in seconds.
        cache_paper_ttl: Cache TTL for paper details in seconds.
        default_search_limit: Default limit for paper search results.
        default_papers_limit: Default limit for author papers results.
        default_citations_limit: Default limit for citations/references results.
    """

    def __init__(self) -> None:
        # Strip whitespace from API key; treat whitespace-only as missing
        raw_api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
        self.api_key: str | None = raw_api_key.strip() if raw_api_key else None
        if self.api_key == "":
            self.api_key = None
        self.graph_api_base_url: str = "https://api.semanticscholar.org/graph/v1"
        self.recommendations_api_base_url: str = (
            "https://api.semanticscholar.org/recommendations/v1"
        )
        self.disable_ssl_verify: bool = os.environ.get("DISABLE_SSL_VERIFY", "").lower() in (
            "true",
            "1",
            "yes",
        )

        # Retry configuration
        self.retry_max_attempts: int = int(os.environ.get("SS_RETRY_MAX_ATTEMPTS", "5"))
        self.retry_base_delay: float = float(os.environ.get("SS_RETRY_BASE_DELAY", "1.0"))
        self.retry_max_delay: float = float(os.environ.get("SS_RETRY_MAX_DELAY", "60.0"))
        self.enable_auto_retry: bool = os.environ.get("SS_ENABLE_AUTO_RETRY", "true").lower() in (
            "true",
            "1",
            "yes",
        )

        # Logging configuration
        self.log_level: str = os.environ.get("SS_LOG_LEVEL", "INFO")
        self.log_format: str = os.environ.get("SS_LOG_FORMAT", "simple")

        # Circuit breaker configuration
        self.circuit_failure_threshold: int = int(
            os.environ.get("SS_CIRCUIT_FAILURE_THRESHOLD", "5")
        )
        self.circuit_recovery_timeout: float = float(
            os.environ.get("SS_CIRCUIT_RECOVERY_TIMEOUT", "30.0")
        )

        # Cache configuration
        self.cache_enabled: bool = os.environ.get("SS_CACHE_ENABLED", "true").lower() in (
            "true",
            "1",
            "yes",
        )
        self.cache_ttl: int = int(os.environ.get("SS_CACHE_TTL", "300"))
        self.cache_paper_ttl: int = int(os.environ.get("SS_CACHE_PAPER_TTL", "3600"))

        # Default limits configuration
        self.default_search_limit: int = _parse_int_with_bounds(
            "SS_DEFAULT_SEARCH_LIMIT", default=10, min_val=1, max_val=100
        )
        self.default_papers_limit: int = _parse_int_with_bounds(
            "SS_DEFAULT_PAPERS_LIMIT", default=10, min_val=1, max_val=1000
        )
        self.default_citations_limit: int = _parse_int_with_bounds(
            "SS_DEFAULT_CITATIONS_LIMIT", default=50, min_val=1, max_val=1000
        )

    @property
    def has_api_key(self) -> bool:
        """Check if an API key is configured."""
        return self.api_key is not None and len(self.api_key) > 0


settings = Settings()
