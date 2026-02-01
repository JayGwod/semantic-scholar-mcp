"""Shared utilities for MCP tools.

This module provides shared constants, client accessor, and tracker accessor
for all tool modules to use consistently.
"""

from collections.abc import Callable

from semantic_scholar_mcp.client import SemanticScholarClient
from semantic_scholar_mcp.paper_tracker import PaperTracker
from semantic_scholar_mcp.paper_tracker import get_tracker as _get_tracker

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


def get_tracker() -> PaperTracker:
    """Get the paper tracker instance.

    Returns:
        The singleton PaperTracker instance.
    """
    return _get_tracker()


# Client accessor - this will be set by server.py
_get_client_func: Callable[[], SemanticScholarClient] | None = None


def set_client_getter(func: Callable[[], SemanticScholarClient]) -> None:
    """Set the function to get the client (called by server.py).

    Args:
        func: A callable that returns a SemanticScholarClient instance.
    """
    global _get_client_func
    _get_client_func = func


def get_client() -> SemanticScholarClient:
    """Get the shared client instance.

    Returns:
        The shared SemanticScholarClient instance.

    Raises:
        RuntimeError: If the client getter has not been configured.
    """
    if _get_client_func is None:
        raise RuntimeError("Client getter not configured. Call set_client_getter first.")
    return _get_client_func()
