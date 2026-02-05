# AudioMuse-AI Testing & Comparison Suite

A comprehensive tool for testing all features, database quality, API results, and performance of AudioMuse-AI — comparing two live instances side-by-side (e.g., **main branch** vs **feature branch**).

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
  - [CLI Arguments](#cli-arguments)
  - [YAML Config File](#yaml-config-file)
  - [Environment Variables](#environment-variables)
- [Test Categories](#test-categories)
  - [API Comparison](#1-api-comparison-30-endpoints)
  - [Database Comparison](#2-database-comparison-17-tables)
  - [Docker & Infrastructure](#3-docker--infrastructure)
  - [Performance Benchmarks](#4-performance-benchmarks)
  - [Existing Test Suite](#5-existing-test-suite-27-tests)
- [Deployment Scenarios](#deployment-scenarios)
  - [Same Machine (Different Ports)](#scenario-1-same-machine-different-ports)
  - [Two Remote Machines](#scenario-2-two-remote-machines-via-ssh)
  - [API-Only Comparison](#scenario-3-api-only-no-db-or-docker)
- [Reports](#reports)
- [Selective Testing](#selective-testing)
- [Test Discovery](#test-discovery)
- [Interpreting Results](#interpreting-results)
- [Troubleshooting](#troubleshooting)

---

## Overview

The testing suite connects to **two AudioMuse-AI instances** simultaneously via:

| Connection     | What it tests                                                   |
|----------------|-----------------------------------------------------------------|
| **API (HTTP)** | All 30+ REST endpoints — response codes, shapes, values, errors |
| **PostgreSQL** | Schema integrity, data quality, embedding health, distributions |
| **Docker**     | Container health, resource usage, log error analysis            |
| **Performance**| Latency benchmarks (p50/p95/p99), concurrent load, DB queries   |

It also discovers and runs the **27 existing tests** (unit, integration, E2E) already in the codebase.

The final output is a **self-contained HTML report** (dark theme, filterable, with visual performance charts) plus a **JSON report** for programmatic consumption.

---

## Architecture

```
testing_suite/
├── run_comparison.py            # CLI entry point
├── __main__.py                  # python -m testing_suite
├── config.py                    # Configuration (CLI / YAML / env vars)
├── utils.py                     # HTTP helpers, DB connectors, Docker log fetchers
├── orchestrator.py              # Coordinates all modules, generates reports
├── comparators/
│   ├── api_comparator.py        # 30+ API endpoint tests
│   ├── db_comparator.py         # Schema, data quality, embeddings, integrity
│   ├── docker_comparator.py     # Container health, logs, resource usage
│   └── performance_comparator.py # Latency, throughput, DB query benchmarks
├── test_runner/
│   └── existing_tests.py        # Discovers & runs 27 existing test files
├── reports/
│   ├── html_report.py           # Self-contained HTML report generator
│   └── output/                  # Generated reports (gitignored)
├── comparison_config.example.yaml
└── requirements.txt
```

---

## Prerequisites

1. **Python 3.10+** on the machine running the suite
2. **Both AudioMuse-AI instances running** (Flask app + Worker + PostgreSQL + Redis)
3. **Network access** from the test runner to both instances (API ports, DB ports)

Install dependencies:

```bash
pip install -r testing_suite/requirements.txt
```

The suite requires: `requests`, `psycopg2-binary`, `pyyaml`, `pytest`, `pytest-json-report`, `pytest-timeout`.

> **Note:** Docker comparison features require the `docker` CLI accessible from the test runner (either locally or via SSH to the remote hosts).

---

## Quick Start

### Minimal (API-only comparison)

```bash
python -m testing_suite \
  --url-a http://192.168.1.100:8000 \
  --url-b http://192.168.1.101:8000
```

This tests all API endpoints and runs existing unit tests. Database and Docker tests will be skipped if hosts aren't specified.

### Full comparison

```bash
python -m testing_suite \
  --url-a http://192.168.1.100:8000 \
  --url-b http://192.168.1.101:8000 \
  --pg-host-a 192.168.1.100 \
  --pg-host-b 192.168.1.101 \
  --flask-container-a audiomuse-main-flask \
  --flask-container-b audiomuse-feature-flask \
  --name-a main --branch-a main \
  --name-b feature --branch-b fix/my-feature
```

### From config file

```bash
cp testing_suite/comparison_config.example.yaml my_config.yaml
# Edit my_config.yaml with your instance details
python -m testing_suite --config my_config.yaml
```

---

## Configuration

There are three ways to configure the suite (in order of precedence):

### CLI Arguments

All settings can be passed as command-line flags. Each instance has a matching set of flags suffixed with `-a` or `-b`:

```
Instance Connection:
  --url-a / --url-b                API base URL (e.g., http://host:8000)
  --name-a / --name-b              Display name (default: main / feature)
  --branch-a / --branch-b          Git branch name for reporting

PostgreSQL:
  --pg-host-a / --pg-host-b        Database host
  --pg-port-a / --pg-port-b        Database port (default: 5432)
  --pg-user-a / --pg-user-b        Database user (default: audiomuse)
  --pg-pass-a / --pg-pass-b        Database password (default: audiomusepassword)
  --pg-db-a / --pg-db-b            Database name (default: audiomusedb)

Redis:
  --redis-a / --redis-b            Redis URL (default: redis://localhost:6379/0)

Docker:
  --flask-container-a / -b         Flask app container name
  --worker-container-a / -b        RQ worker container name
  --ssh-host-a / -b                SSH host for remote Docker access
  --ssh-user-a / -b                SSH username
  --ssh-key-a / -b                 SSH private key path

Test Control:
  --only CATEGORIES                Only run listed categories (comma-separated)
  --skip CATEGORIES                Skip listed categories
  --warmup N                       Warmup requests before benchmarking (default: 3)
  --bench-requests N               Benchmark iterations per endpoint (default: 10)
  --concurrent N                   Concurrent users for load test (default: 5)

Output:
  --output-dir PATH                Report output directory
  --format {html,json,both}        Report format (default: both)
  -v / --verbose                   Debug-level logging
```

### YAML Config File

Copy and edit the example:

```bash
cp testing_suite/comparison_config.example.yaml comparison_config.yaml
```

The YAML file supports all the same settings. See `comparison_config.example.yaml` for the full annotated template with all available options including quality thresholds, test track references, and performance parameters.

### Environment Variables

Every setting can be set via environment variables with the `INSTANCE_A_` / `INSTANCE_B_` prefix:

```bash
export INSTANCE_A_API_URL=http://192.168.1.100:8000
export INSTANCE_A_PG_HOST=192.168.1.100
export INSTANCE_B_API_URL=http://192.168.1.101:8000
export INSTANCE_B_PG_HOST=192.168.1.101
export COMPARISON_VERBOSE=true
export COMPARISON_OUTPUT_DIR=./reports

python -m testing_suite
```

---

## Test Categories

### 1. API Comparison (30+ endpoints)

Tests every AudioMuse-AI API endpoint on both instances and compares:

| What's tested | Details |
|---------------|---------|
| **Status codes** | Both instances return the same HTTP status |
| **Response shape** | Same JSON keys, same structure |
| **List lengths** | Playlists, search results, etc. have comparable sizes |
| **Required fields** | Track objects have `item_id`, `title`, etc. |
| **Error handling** | Both handle invalid inputs the same way |
| **Functional tests** | Track search, similarity, alchemy, path finding with real data |

**Endpoints covered:**
- `/api/config`, `/api/playlists`, `/api/active_tasks`, `/api/last_task`
- `/api/search_tracks`, `/api/similar_tracks`, `/api/max_distance`
- `/api/map`, `/api/map_cache_status`
- `/api/clap/stats`, `/api/clap/warmup/status`, `/api/clap/top_queries`
- `/api/alchemy`, `/api/find_path`, `/api/sonic_fingerprint/generate`
- `/api/artist_projections`, `/api/search_artists`
- `/api/setup/status`, `/api/setup/providers`, `/api/setup/settings`, `/api/setup/server-info`
- `/api/setup/providers/types`, `/api/providers/enabled`
- `/api/cron`, `/api/waveform`, `/api/collection/last_task`
- `/external/search`, `/chat/api/config_defaults`
- Error cases: nonexistent tasks, missing parameters

### 2. Database Comparison (17 tables)

Connects directly to both PostgreSQL instances and validates:

| Category | Tests |
|----------|-------|
| **Schema** | All 17 expected tables exist with correct columns |
| **Row counts** | Compared with configurable tolerance (default 5%) |
| **Data quality** | NULL rates in critical score columns (item_id, title, author, tempo, key, scale, mood_vector) |
| **Duplicates** | No duplicate item_ids in score table |
| **Mood vector format** | Validates mood_vector string format |
| **Embedding integrity** | Coverage (% of scores with embeddings), NULL checks, dimension consistency |
| **Referential integrity** | No orphaned rows in embedding→score, provider_track→provider |
| **Score distributions** | Statistical comparison of tempo, energy (min/max/avg/stddev) |
| **Key distribution** | Musical key value comparison between instances |
| **Playlist quality** | Distinct count, avg tracks per playlist, NULL item_ids |
| **Index data** | Voyager HNSW, Artist GMM, Map projection, Artist projection presence |
| **Task health** | Failed task count, stuck tasks (>2hr), success rate comparison |
| **Provider config** | Same provider types and settings |
| **App settings** | Same configuration keys |

### 3. Docker & Infrastructure

Analyzes container health and logs via the Docker CLI (local or SSH):

| Category | Tests |
|----------|-------|
| **Container status** | Running/stopped for Flask, Worker, PostgreSQL, Redis |
| **Restart counts** | Flags high restart counts (>5 = FAIL) |
| **Health checks** | Docker health check status comparison |
| **Memory usage** | MB comparison with % difference threshold |
| **CPU usage** | Percentage comparison |
| **Error patterns** | 11 patterns: tracebacks, OOM, connection errors, timeouts, permission, disk, crashes, worker deaths, DB errors, Redis errors |
| **Warning patterns** | 5 patterns: deprecation, warnings, retries, slow ops, memory pressure |
| **Python tracebacks** | Exact count comparison (>10 = FAIL) |
| **Redis connectivity** | Ping test from inside Flask container |
| **PostgreSQL connectivity** | SELECT 1 test from inside Flask container |

### 4. Performance Benchmarks

Measures and compares response times with statistical rigor:

| Category | Details |
|----------|---------|
| **Endpoint latency** | 16 endpoints benchmarked with warmup phase, measuring p50/p95/p99/mean/max/stddev |
| **Concurrent load** | Configurable concurrent users hitting key endpoints simultaneously, measuring throughput (req/s) |
| **DB query performance** | 8 critical queries benchmarked: counts, joins, aggregations, group-bys |

**Thresholds:**
- **PASS**: Instance B within 20% of A (or faster)
- **WARN**: Instance B 20-100% slower
- **FAIL**: Instance B >2x slower

### 5. Existing Test Suite (27 tests)

Discovers and runs all tests already in the codebase:

| Category | Files | Tests |
|----------|-------|-------|
| **Unit tests** | 17 files in `tests/unit/` | test_ai, test_analysis, test_app_analysis, test_artist_gmm_manager, test_clap_text_search, test_clustering, test_clustering_helper, test_clustering_postprocessing, test_commons, test_mediaserver, test_memory_cleanup, test_memory_utils, test_path_manager, test_song_alchemy, test_sonic_fingerprint_manager, test_string_sanitization, test_voyager_manager |
| **Integration tests** | 2 files in `test/` | test_analysis_integration, test_clap_analysis_integration |
| **E2E API tests** | 8 tests in `test/test.py` | analysis smoke, instant playlist, sonic fingerprint, song alchemy, map visualization, similarity, song path, clustering smoke |

Unit tests run once (they mock dependencies). E2E tests run against both instances with the `BASE_URL` pointed at each.

---

## Deployment Scenarios

### Scenario 1: Same Machine, Different Ports

Two Docker Compose stacks running on ports 8000 and 8001:

```bash
python -m testing_suite \
  --url-a http://localhost:8000 --url-b http://localhost:8001 \
  --pg-host-a localhost --pg-port-a 5432 \
  --pg-host-b localhost --pg-port-b 5433 \
  --flask-container-a audiomuse-main-flask \
  --flask-container-b audiomuse-feature-flask \
  --worker-container-a audiomuse-main-worker \
  --worker-container-b audiomuse-feature-worker
```

### Scenario 2: Two Remote Machines (via SSH)

Instance A on server1, Instance B on server2:

```bash
python -m testing_suite \
  --url-a http://server1:8000 --url-b http://server2:8000 \
  --pg-host-a server1 --pg-host-b server2 \
  --ssh-host-a server1 --ssh-user-a deploy --ssh-key-a ~/.ssh/id_rsa \
  --ssh-host-b server2 --ssh-user-b deploy --ssh-key-b ~/.ssh/id_rsa
```

The suite will SSH into each server to run `docker inspect`, `docker logs`, and `docker stats`.

### Scenario 3: API-Only (No DB or Docker)

If you only have HTTP access to both instances:

```bash
python -m testing_suite \
  --url-a http://main.example.com --url-b http://feature.example.com \
  --only api,performance
```

---

## Reports

Every run produces two reports in the output directory (`testing_suite/reports/output/` by default):

### HTML Report

A self-contained, dark-themed HTML file with:
- Overall pass/fail status badge
- Summary cards (total, passed, failed, errors, warnings)
- Per-category expandable sections
- Filterable result tables (filter by Pass/Fail/Warn/Error/Skip)
- Side-by-side Instance A vs Instance B values
- Visual performance bar charts comparing latency

Open in any browser: `testing_suite/reports/output/comparison_latest.html`

### JSON Report

Machine-readable format with full test details:

```json
{
  "timestamp": "2025-01-15T10:30:00.000000",
  "instance_a": {"name": "main", "branch": "main"},
  "instance_b": {"name": "feature", "branch": "feature"},
  "overall_status": "PASS",
  "summary": {"total": 150, "passed": 142, "failed": 3, "errors": 0},
  "categories": {
    "api": {"total": 50, "passed": 48, "failed": 2, ...},
    "database": {"total": 40, "passed": 38, ...},
    ...
  }
}
```

---

## Selective Testing

### Run only specific categories

```bash
# API and database only
python -m testing_suite --url-a ... --url-b ... --only api,db

# Performance only
python -m testing_suite --url-a ... --url-b ... --only performance

# Existing unit tests only
python -m testing_suite --url-a ... --url-b ... --only unit
```

**Category names:** `api`, `db` (or `database`), `docker`, `performance` (or `perf`), `existing_tests`, `unit`, `integration`

### Skip specific categories

```bash
# Skip Docker and existing tests (faster)
python -m testing_suite --url-a ... --url-b ... --skip docker,existing_tests

# Skip performance benchmarks
python -m testing_suite --url-a ... --url-b ... --skip perf
```

### Tune performance test parameters

```bash
# Light benchmarking (fast)
python -m testing_suite --url-a ... --url-b ... --warmup 1 --bench-requests 3 --concurrent 2

# Heavy benchmarking (thorough)
python -m testing_suite --url-a ... --url-b ... --warmup 10 --bench-requests 50 --concurrent 20
```

---

## Test Discovery

List all available tests without running anything:

```bash
python -m testing_suite --discover
```

Output:

```
=== AudioMuse-AI Test Discovery ===

Unit Tests (17 files):
  [OK] tests/unit/test_ai.py
  [OK] tests/unit/test_analysis.py
  ...

Integration Tests (2 files):
  [OK] test/test_analysis_integration.py
  [OK] test/test_clap_analysis_integration.py

E2E API Tests (8 tests):
  [OK] test_analysis_smoke_flow (test/test.py)
  ...

Total: 27 test files/entries discovered.
```

---

## Interpreting Results

### Status Codes

| Status | Meaning |
|--------|---------|
| **PASS** | Both instances match or values are within acceptable thresholds |
| **FAIL** | Significant difference detected, or a quality check failed |
| **WARN** | Minor difference or non-critical issue detected |
| **SKIP** | Test could not run (missing table, unreachable endpoint, etc.) |
| **ERROR** | Test itself errored (connection failure, timeout, exception) |

### Exit Codes

The CLI returns:
- `0` — All tests passed (or only warnings/skips)
- `1` — One or more tests failed or errored

This makes it suitable for CI/CD pipelines:

```bash
python -m testing_suite --config config.yaml || echo "Comparison found regressions!"
```

### Performance Comparison Logic

- **B/A ratio ≤ 1.2** → PASS (B is within 20% of A)
- **B/A ratio ≤ 2.0** → WARN (B is up to 2x slower)
- **B/A ratio > 2.0** → FAIL (B is more than 2x slower)
- If B is faster than A, that's always a PASS

---

## Troubleshooting

### "Cannot connect to either database instance"

- Verify PostgreSQL is accessible from the test runner machine
- Check `--pg-host-a/b`, `--pg-port-a/b`, `--pg-user-a/b`, `--pg-pass-a/b`
- Ensure `pg_hba.conf` allows connections from the test runner IP
- Try: `psql -h <host> -p <port> -U <user> -d <dbname> -c "SELECT 1"`

### "Neither instance is reachable"

- Verify the API URLs are correct and the Flask servers are running
- Check firewall rules allow traffic on port 8000
- Try: `curl http://<host>:8000/api/config`

### "Cannot inspect containers (Docker not available)"

- Docker CLI must be installed on the test runner (or accessible via SSH)
- Container names must match what's running (`docker ps --format '{{.Names}}'`)
- For remote Docker access, SSH must be configured: `--ssh-host-a/b`, `--ssh-user-a/b`

### "pytest-json-report not found"

```bash
pip install pytest-json-report
```

Or install all dependencies:

```bash
pip install -r testing_suite/requirements.txt
```

### Customizing test tracks

The API functional tests (search, similarity, alchemy, path) use reference tracks. Set them to tracks that exist in your library:

```bash
python -m testing_suite --config my_config.yaml
```

In your YAML config:

```yaml
test_track_artist_1: "Artist In Your Library"
test_track_title_1: "Song Title"
test_track_artist_2: "Another Artist"
test_track_title_2: "Another Song"
```
