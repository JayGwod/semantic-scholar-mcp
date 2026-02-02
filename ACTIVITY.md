# Activity

Append activity log at bottom. Follow the template:

```markdown
### Current atetime in YYYY-MM-DD HH:MM format for Central European Timezone (use tools)

**Tasks completed:** Tasks that are already completed; take ID and description from @SPEC.md.

**Current tasks:** Task that is implemented in the current instance; take description from @SPEC.md.

**Blockers:** Issues encountered.
```

## Session Log

### 2026-02-02 14:22 (CET)

**Task completed:** US-1: Rename ConnectionError to avoid builtin shadowing

**Changes made:**
- `src/semantic_scholar_mcp/exceptions.py`: Renamed `ConnectionError` class to `APIConnectionError`
- `src/semantic_scholar_mcp/client.py`: Updated import and all 8 usages of `ConnectionError` to `APIConnectionError`
- `tests/test_client.py`: Updated import and all usages of `ConnectionError` to `APIConnectionError`

**Verification:**
- `uv run ruff check src/ tests/`: All checks passed!
- `uv run ruff format src/ tests/`: 1 file reformatted
- `uv run pytest -v`: 193 passed, 6 failed (integration tests failing due to SSL certificate issues - unrelated to this change)
- `uv run ty check src/`: 4 pre-existing diagnostics (unrelated to this change)

**Blockers:** None

### 2026-02-02 14:25 (CET)

**Task completed:** US-2: Add logging to silent exception handlers

**Changes made:**
- `src/semantic_scholar_mcp/server.py`: Added `logger.debug("Error during client cleanup: %s", e)` to the exception handler in `_cleanup_client()` function (line 77-78)

**Verification:**
- `uv run ruff check src/ tests/`: All checks passed!
- `uv run ruff format src/ tests/`: 30 files left unchanged
- `uv run pytest -v`: 193 passed, 6 failed (integration tests failing due to SSL certificate issues - unrelated to this change)
- `uv run ty check src/`: 4 pre-existing diagnostics (unrelated to this change)

**Blockers:** None

### 2026-02-02 14:29 (CET)

**Task completed:** US-10: Add comprehensive ruff linting rules

**Changes made:**
- `pyproject.toml`: Added ruff rules B (bugbear), C4 (comprehensions), SIM (simplify); added `ignore = ["E501"]` for line length; added isort configuration with `known-first-party = ["semantic_scholar_mcp"]`
- `src/semantic_scholar_mcp/client.py`: Added `contextlib` import and replaced try/except/pass with `contextlib.suppress(ValueError)` (SIM105 fix)
- `src/semantic_scholar_mcp/tools/tracking.py`: Converted if/else to ternary operator (SIM108 fix)
- `tests/test_client.py`: Renamed unused loop variable `i` to `_i` (B007 fix)
- `tests/test_rate_limiter.py`: Combined nested `with` statements into single statement (SIM117 fix)

**Verification:**
- `uv run ruff check src/ tests/`: All checks passed!
- `uv run ruff format src/ tests/`: 30 files left unchanged
- `uv run pytest -v`: 193 passed, 6 failed (integration tests failing due to SSL certificate issues - unrelated to this change)
- `uv run ty check src/`: 4 pre-existing diagnostics (unrelated to this change)

**Blockers:** None

### 2026-02-02 14:31 (CET)

**Task completed:** US-11: Add test coverage reporting

**Changes made:**
- `pyproject.toml`: Added `pytest-cov>=4.0` to dev dependencies; added `addopts = "--cov=semantic_scholar_mcp --cov-report=term-missing"` to pytest config

**Verification:**
- `uv run ruff check src/ tests/`: All checks passed!
- `uv run ruff format src/ tests/`: 30 files left unchanged
- `uv run pytest -v`: 193 passed, 6 failed (integration tests failing due to SSL certificate issues - unrelated to this change); coverage report generated showing 87% total coverage
- `uv run ty check src/`: 4 pre-existing diagnostics (unrelated to this change)

**Blockers:** None

### 2026-02-02 14:34 (CET)

**Task completed:** US-3: Extract DRY helper for nested paper fields

**Changes made:**
- `src/semantic_scholar_mcp/tools/_common.py`: Added `build_nested_paper_fields(prefix: str) -> str` function with docstring
- `src/semantic_scholar_mcp/tools/papers.py`: Updated import to include `build_nested_paper_fields`; replaced manual field transformation in `get_paper_citations()` (line 213) and `get_paper_references()` (line 290) with calls to the new helper

**Verification:**
- `uv run ruff check src/ tests/`: All checks passed!
- `uv run ruff format src/ tests/`: 30 files left unchanged
- `uv run pytest -v`: 193 passed, 6 failed (integration tests failing due to SSL certificate issues - unrelated to this change)
- `uv run ty check src/`: 4 pre-existing diagnostics (unrelated to this change)

**Blockers:** None

### 2026-02-02 14:38 (CET)

**Task completed:** US-4: Extract DRY helper for sorting by citations

**Changes made:**
- `src/semantic_scholar_mcp/tools/_common.py`: Added `HasCitationCount` Protocol class and `sort_by_citations[T: HasCitationCount]()` generic helper function with docstring using PEP 695 type parameter syntax
- `src/semantic_scholar_mcp/tools/authors.py`: Updated import to include `sort_by_citations`; replaced 4 manual sorting patterns with calls to the helper:
  - `find_duplicate_authors` - ORCID groups sorting (line 278)
  - `find_duplicate_authors` - DBLP groups sorting (line 300)
  - `consolidate_authors` - authors sorting (line 404)
  - `get_author_top_papers` - papers sorting (line 562)

**Verification:**
- `uv run ruff check src/ tests/`: All checks passed!
- `uv run ruff format src/ tests/`: 1 file reformatted, 29 files left unchanged
- `uv run pytest -v`: 193 passed, 6 failed (integration tests failing due to SSL certificate issues - unrelated to this change)
- `uv run ty check src/`: 4 pre-existing diagnostics (unrelated to this change)

**Blockers:** None

### 2026-02-02 14:43 (CET)

**Task completed:** US-5: Add tests for server initialization and lifecycle

**Changes made:**
- `tests/test_server_init.py`: Created new test file with 15 tests covering:
  - TestServerCreation: Tests for MCP server instance creation (3 tests)
  - TestToolRegistration: Tests for tool registration verification (5 tests)
  - TestClientSingleton: Tests for client singleton behavior (3 tests)
  - TestCleanupHandler: Tests for cleanup handler registration and behavior (4 tests)

**Verification:**
- `uv run ruff check src/ tests/`: All checks passed!
- `uv run ruff format src/ tests/`: 31 files left unchanged
- `uv run pytest -v`: 208 passed, 6 failed (integration tests failing due to SSL certificate issues - unrelated to this change)
- `uv run ty check src/`: 4 pre-existing diagnostics (unrelated to this change)

**Blockers:** None
