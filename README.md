# Semantic Scholar MCP Server

A Model Context Protocol (MCP) server that enables AI assistants to search, retrieve, and analyze academic papers through the Semantic Scholar API.

> **Disclaimer:** This project is not officially affiliated with, endorsed by, or sponsored by Semantic Scholar or the Allen Institute for AI.

## Quick Start

### Claude Code

```bash
claude mcp add semantic-scholar -s user -- uvx --from git+https://github.com/akapet00/semantic-scholar-mcp semantic-scholar-mcp
```

### Claude Desktop

Add to your config file (`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS, `%APPDATA%\Claude\claude_desktop_config.json` on Windows):

```json
{
  "mcpServers": {
    "semantic-scholar": {
      "command": "uvx",
      "args": [
        "--from",
        "git+https://github.com/akapet00/semantic-scholar-mcp",
        "semantic-scholar-mcp"
      ]
    }
  }
}
```

That's it! No API key required for basic usage.

---

## Features

- **Search papers** by keyword, topic, or semantic similarity
- **Explore citations** - find papers that cite a given paper
- **Explore references** - find papers that a given paper cites
- **Search authors** by name and retrieve their profiles
- **Author deduplication** - find and consolidate duplicate author records
- **ML-based recommendations** - discover similar papers using Semantic Scholar's recommendation system
- **Multi-paper recommendations** - find papers related to multiple examples
- **Session tracking** - track papers retrieved during a session
- **BibTeX export** - export tracked papers to BibTeX format

### Reliability & Performance

- **Intelligent caching** - TTL-based response caching with LRU eviction
- **Rate limiting** - token bucket algorithm with automatic backoff
- **Circuit breaker** - automatic failure detection and recovery
- **Automatic retries** - exponential backoff with jitter for transient errors

## Installation

### Run Directly (no clone required)

```bash
uvx --from git+https://github.com/akapet00/semantic-scholar-mcp semantic-scholar-mcp
```

### Local Installation

```bash
git clone https://github.com/akapet00/semantic-scholar-mcp.git
cd semantic-scholar-mcp
uv sync
```

## Client Configuration

### Claude Code

Add the MCP server globally (available in all projects):

```bash
claude mcp add semantic-scholar -s user -- uvx --from git+https://github.com/akapet00/semantic-scholar-mcp semantic-scholar-mcp
```

With an API key:

```bash
claude mcp add semantic-scholar -s user -e SEMANTIC_SCHOLAR_API_KEY=your_key -- uvx --from git+https://github.com/akapet00/semantic-scholar-mcp semantic-scholar-mcp
```

To verify:

```bash
claude mcp list
```

To remove:

```bash
claude mcp remove semantic-scholar -s user
```

### Claude Desktop

Add the following to your Claude Desktop configuration file:

**macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`

**Windows:** `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "semantic-scholar": {
      "command": "uvx",
      "args": [
        "--from",
        "git+https://github.com/akapet00/semantic-scholar-mcp",
        "semantic-scholar-mcp"
      ],
      "env": {
        "SEMANTIC_SCHOLAR_API_KEY": ""
      }
    }
  }
}
```

#### Using a Local Clone

If you prefer to run from a local clone:

```json
{
  "mcpServers": {
    "semantic-scholar": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/semantic-scholar-mcp", "semantic-scholar-mcp"],
      "env": {
        "SEMANTIC_SCHOLAR_API_KEY": ""
      }
    }
  }
}
```

## Updating the Server

When installed via `git+https://...`, the package is cached locally. Changes pushed to GitHub are **not automatically pulled**.

### Get the Latest Version

**Claude Code:**

```bash
claude mcp remove semantic-scholar -s user
claude mcp add semantic-scholar -s user -- uvx --from git+https://github.com/akapet00/semantic-scholar-mcp semantic-scholar-mcp
```

**Direct run with refresh:**

```bash
uvx --refresh --from git+https://github.com/akapet00/semantic-scholar-mcp semantic-scholar-mcp
```

### Pin to a Specific Version

You can pin to a branch, tag, or commit for reproducible installations:

```bash
# Specific branch
uvx --from git+https://github.com/akapet00/semantic-scholar-mcp@main semantic-scholar-mcp

# Specific tag
uvx --from git+https://github.com/akapet00/semantic-scholar-mcp@v0.1.0 semantic-scholar-mcp

# Specific commit
uvx --from git+https://github.com/akapet00/semantic-scholar-mcp@abc1234 semantic-scholar-mcp
```

## API Key (Optional)

The Semantic Scholar API works without an API key, but you can get one for higher rate limits:

1. Visit https://www.semanticscholar.org/product/api
2. Request an API key
3. Set the `SEMANTIC_SCHOLAR_API_KEY` environment variable

You can also create a `.env` file based on `.env.example`:

```bash
cp .env.example .env
# Edit .env and add your API key
```

### Rate Limits

| Access Level | Rate Limit |
|--------------|------------|
| Unauthenticated | 5,000 requests per 5 minutes (shared pool) |
| With API key | 1 request per second (dedicated) |

For most research use cases, the unauthenticated rate limit is sufficient.

## Corporate Networks / SSL Issues

> **Note:** If you're behind a corporate proxy or firewall that intercepts HTTPS traffic, you may encounter SSL certificate errors like `CERTIFICATE_VERIFY_FAILED: self-signed certificate in certificate chain`.

To bypass SSL verification, set the `DISABLE_SSL_VERIFY` environment variable.

**Claude Code:**

```bash
claude mcp add semantic-scholar -s user -e DISABLE_SSL_VERIFY=true -- uvx --from git+https://github.com/akapet00/semantic-scholar-mcp semantic-scholar-mcp
```

**Claude Desktop:**

```json
{
  "mcpServers": {
    "semantic-scholar": {
      "command": "uvx",
      "args": [
        "--from",
        "git+https://github.com/akapet00/semantic-scholar-mcp",
        "semantic-scholar-mcp"
      ],
      "env": {
        "DISABLE_SSL_VERIFY": "true"
      }
    }
  }
}
```

**Warning:** Only use this option in trusted corporate networks. Disabling SSL verification removes protection against man-in-the-middle attacks.

## Available Tools

### Paper Tools

#### search_papers

Search for academic papers by keyword or phrase.

```python
# Search for papers on a topic
search_papers("transformer attention mechanism")

# Search with filters
search_papers(
    query="CRISPR gene editing",
    year="2020-2024",
    min_citation_count=50,
    fields_of_study=["Biology", "Medicine"],
    limit=20
)
```

#### get_paper_details

Get detailed information about a specific paper.

```python
# Using Semantic Scholar ID
get_paper_details("649def34f8be52c8b66281af98ae884c09aef38b")

# Using DOI
get_paper_details("DOI:10.18653/v1/N18-3011")

# Using ArXiv ID
get_paper_details("ARXIV:1706.03762")
```

#### get_paper_citations

Get papers that cite a given paper.

```python
# Get citations for a paper
get_paper_citations("ARXIV:1706.03762")

# With year filter and limit
get_paper_citations("ARXIV:1706.03762", year="2020-2024", limit=50)
```

#### get_paper_references

Get papers that a given paper references.

```python
# Get references for a paper
get_paper_references("ARXIV:1706.03762")

# With custom limit
get_paper_references("ARXIV:1706.03762", limit=200)
```

### Author Tools

#### search_authors

Search for authors by name.

```python
# Search for an author
search_authors("Geoffrey Hinton")

# Search with limit
search_authors("Smith", limit=20)
```

#### get_author_details

Get detailed information about an author.

```python
# Get author details with papers
get_author_details("1741101")

# Get author details without papers
get_author_details("1741101", include_papers=False)

# Get more papers
get_author_details("1741101", papers_limit=50)
```

#### find_duplicate_authors

Identify potentially duplicate author records by matching external IDs.

```python
# Search for potential duplicates
find_duplicate_authors(["Geoffrey Hinton", "G. Hinton", "Geoffrey E. Hinton"])

# Match by specific identifier
find_duplicate_authors(["Yoshua Bengio"], match_by_orcid=True, match_by_dblp=False)
```

#### consolidate_authors

Preview or merge duplicate author records.

```python
# Preview merged record
consolidate_authors(["1741101", "1741102"])

# Confirm merge
consolidate_authors(["1741101", "1741102"], confirm_merge=True)
```

### Recommendation Tools

#### get_recommendations

Find similar papers using ML-based recommendations.

```python
# Get recommendations for a paper
get_recommendations("ARXIV:1706.03762")

# Get recommendations from all CS papers
get_recommendations("ARXIV:1706.03762", from_pool="all-cs", limit=20)
```

#### get_related_papers

Find papers related to multiple example papers.

```python
# Single positive example
get_related_papers(["ARXIV:1706.03762"])

# Multiple positive examples
get_related_papers(
    positive_paper_ids=["ARXIV:1706.03762", "ARXIV:1810.04805"],
    limit=20
)

# With negative examples (papers to avoid similarity to)
get_related_papers(
    positive_paper_ids=["DOI:10.18653/v1/N18-3011"],
    negative_paper_ids=["DOI:10.1145/3292500.3330701"],
    limit=15
)
```

### Session & Export Tools

#### list_tracked_papers

View papers retrieved during the current session.

```python
# List all tracked papers
list_tracked_papers()

# Filter by source tool
list_tracked_papers(source_tool="search_papers")
```

#### clear_tracked_papers

Reset paper tracking for the current session.

```python
clear_tracked_papers()
```

#### export_bibtex

Export papers to BibTeX format.

```python
# Export all tracked papers
export_bibtex()

# Export specific papers
export_bibtex(paper_ids=["abc123", "def456"])

# Export with options
export_bibtex(
    include_abstract=True,
    cite_key_format="author_year_title",
    file_path="references.bib"
)
```

## Configuration Reference

All settings can be configured via environment variables.

### API Settings

| Variable | Description | Default |
|----------|-------------|---------|
| `SEMANTIC_SCHOLAR_API_KEY` | API key for higher rate limits | (none) |
| `DISABLE_SSL_VERIFY` | Bypass SSL verification for corporate proxies | `false` |

### Default Limits

| Variable | Description | Default | Max |
|----------|-------------|---------|-----|
| `SS_DEFAULT_SEARCH_LIMIT` | Default limit for paper search results | `10` | `100` |
| `SS_DEFAULT_PAPERS_LIMIT` | Default limit for author papers results | `10` | `1000` |
| `SS_DEFAULT_CITATIONS_LIMIT` | Default limit for citations/references | `50` | `1000` |

### Cache Settings

| Variable | Description | Default |
|----------|-------------|---------|
| `SS_CACHE_ENABLED` | Enable response caching | `true` |
| `SS_CACHE_TTL` | Default cache TTL in seconds | `300` |
| `SS_CACHE_PAPER_TTL` | Paper details cache TTL in seconds | `3600` |

### Retry Settings

| Variable | Description | Default |
|----------|-------------|---------|
| `SS_ENABLE_AUTO_RETRY` | Enable automatic retries | `true` |
| `SS_RETRY_MAX_ATTEMPTS` | Maximum retry attempts | `5` |
| `SS_RETRY_BASE_DELAY` | Initial retry delay in seconds | `1.0` |
| `SS_RETRY_MAX_DELAY` | Maximum delay between retries | `60.0` |

### Circuit Breaker Settings

| Variable | Description | Default |
|----------|-------------|---------|
| `SS_CIRCUIT_FAILURE_THRESHOLD` | Failures before circuit opens | `5` |
| `SS_CIRCUIT_RECOVERY_TIMEOUT` | Seconds before recovery test | `30` |

### Logging Settings

| Variable | Description | Default |
|----------|-------------|---------|
| `SS_LOG_LEVEL` | Log level (DEBUG, INFO, WARNING, ERROR) | `INFO` |
| `SS_LOG_FORMAT` | Log format (`simple` or `detailed`) | `simple` |

## Development

### Running the server locally

```bash
# Development mode with inspector
uv run fastmcp dev src/semantic_scholar_mcp/server.py

# Production mode
uv run semantic-scholar-mcp
```

### Running tests

```bash
uv run pytest tests/ -v
```

### Linting and formatting

```bash
# Check linting
uv run ruff check src/ tests/

# Format code
uv run ruff format src/ tests/

# Type checking
uv run ty check src/
```

## Project Structure

```
src/semantic_scholar_mcp/
├── server.py              # FastMCP server entry point
├── client.py              # HTTP client with caching, rate limiting, circuit breaker
├── config.py              # Settings from environment variables
├── cache.py               # TTL-based response caching
├── rate_limiter.py        # Token bucket rate limiter with exponential backoff
├── circuit_breaker.py     # Circuit breaker pattern implementation
├── paper_tracker.py       # Session-based paper tracking
├── bibtex.py              # BibTeX export system
├── models.py              # Pydantic response models
├── exceptions.py          # Custom exception types
├── logging_config.py      # Logging configuration
└── tools/                 # MCP tool implementations
    ├── papers.py          # Paper search, details, citations, references
    ├── authors.py         # Author search, details, deduplication
    ├── recommendations.py # Paper recommendations
    └── tracking.py        # Session tracking and BibTeX export
```

## License

MIT License
