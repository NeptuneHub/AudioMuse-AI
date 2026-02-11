# AudioMuse-AI v0.9.0 - Comprehensive Test Checklist

## Branch: `multi-provider-v2` vs `main`

**Scope**: 96 changed files, +22,091/-1,489 lines, 47 commits across 7 feature areas.

---

## Table of Contents

1. [How to Use the Test Suite](#1-how-to-use-the-test-suite)
2. [Automated vs Manual Testing Summary](#2-automated-vs-manual-testing-summary)
3. [Multi-Provider Architecture](#3-multi-provider-architecture)
4. [GUI Setup Wizard](#4-gui-setup-wizard)
5. [Environment / Config Setup](#5-environment--config-setup)
6. [API Endpoints](#6-api-endpoints)
7. [App Interactions (UI/UX)](#7-app-interactions-uiux)
8. [Instant Playlist & AI Changes](#8-instant-playlist--ai-changes)
9. [MCP Tools](#9-mcp-tools)
10. [Provider-Specific Testing](#10-provider-specific-testing)
11. [Database & Schema Changes](#11-database--schema-changes)
12. [Dark Mode](#12-dark-mode)
13. [Analysis Pipeline](#13-analysis-pipeline)
14. [Playlist Ordering](#14-playlist-ordering)
15. [Deployment & Docker](#15-deployment--docker)
16. [Regression Tests](#16-regression-tests)
17. [Security](#17-security)

---

## 1. How to Use the Test Suite

### Test Directory Structure

```
AudioMuse-AI/
├── tests/
│   ├── conftest.py              # Shared fixtures (importlib bypass, DB mocks, config restore)
│   └── unit/                    # Unit tests (no external services needed)
│       ├── test_analysis.py     # Audio analysis (50+ tests)
│       ├── test_ai.py           # AI provider routing (30+ tests)
│       ├── test_ai_mcp_client.py    # NEW - AI MCP client (60+ tests)
│       ├── test_clustering.py   # Clustering helpers (60+ tests)
│       ├── test_clustering_helper.py
│       ├── test_clustering_postprocessing.py
│       ├── test_mediaserver.py  # Jellyfin provider (15+ tests)
│       ├── test_voyager_manager.py  # Similarity search (20+ tests)
│       ├── test_commons.py      # Score vectors (10+ tests)
│       ├── test_app_analysis.py
│       ├── test_clap_text_search.py
│       ├── test_artist_gmm_manager.py
│       ├── test_memory_cleanup.py
│       ├── test_memory_utils.py
│       ├── test_path_manager.py
│       ├── test_song_alchemy.py
│       ├── test_sonic_fingerprint_manager.py
│       ├── test_string_sanitization.py
│       ├── test_mcp_server.py         # NEW - MCP tools
│       ├── test_playlist_ordering.py  # NEW - Playlist ordering
│       ├── test_app_setup.py          # NEW - Setup wizard & providers
│       ├── test_app_chat.py           # NEW - Instant playlist pipeline
│       └── test_mediaserver_localfiles.py  # NEW - LocalFiles provider
├── test/                    # Integration tests (require running services)
│   ├── test.py              # End-to-end smoke tests
│   ├── test_analysis_integration.py
│   ├── test_clap_analysis_integration.py
│   ├── test_gpu_status.py
│   ├── verify_onnx_embeddings.py
│   └── provider_testing_stack/    # Multi-provider Docker test stack
│       ├── docker-compose-test-audiomuse.yaml
│       ├── docker-compose-test-providers.yaml
│       └── TEST_GUIDE.md
├── testing_suite/           # Comparison & benchmarking
│   ├── __main__.py                # CLI entry point
│   ├── config.py                  # Suite configuration
│   ├── orchestrator.py            # Test orchestration
│   ├── utils.py                   # Shared utilities
│   ├── test_instant_playlist.py   # Instant playlist scenarios
│   ├── test_ai_naming.py          # AI naming quality
│   ├── comparators/               # Cross-instance comparison
│   │   ├── api_comparator.py
│   │   ├── db_comparator.py
│   │   ├── docker_comparator.py
│   │   └── performance_comparator.py
│   ├── test_runner/               # Existing test runner
│   │   └── existing_tests.py
│   ├── run_comparison.py          # Entry point
│   └── reports/html_report.py
└── pytest.ini               # Test configuration
```

### Running Tests

```bash
# ============================================================
# UNIT TESTS (no external services, 2-5 minutes)
# ============================================================

# Run ALL unit tests
pytest tests/unit/ -v

# Run specific test file
pytest tests/unit/test_mcp_server.py -v

# Run specific test class
pytest tests/unit/test_mcp_server.py::TestSearchDatabase -v

# Run specific test method
pytest tests/unit/test_mcp_server.py::TestSearchDatabase::test_genre_regex_prevents_substring_match -v

# Skip slow tests
pytest tests/unit/ -v -m "not slow"

# Run only new tests (for this branch)
pytest tests/unit/test_mcp_server.py tests/unit/test_playlist_ordering.py tests/unit/test_app_setup.py tests/unit/test_app_chat.py tests/unit/test_mediaserver_localfiles.py tests/unit/test_ai_mcp_client.py -v

# ============================================================
# INTEGRATION TESTS (require running services, 20+ minutes)
# ============================================================

# Requires: Flask server, PostgreSQL, Redis, ONNX models
pytest test/ -v -s --timeout=1200

# ============================================================
# COMPARISON SUITE (require Docker + AI services)
# ============================================================

# Run full comparison between two instances
python testing_suite/run_comparison.py

# Run instant playlist benchmarks
python testing_suite/test_instant_playlist.py --runs 5

# Run AI naming benchmarks
python testing_suite/test_ai_naming.py --runs 5

# ── Benchmark Configuration ──────────────────────────────
# Both benchmarks are self-contained — they do NOT use the
# main app's AI keys (Gemini, OpenAI, Mistral). Instead they
# route cloud models through OpenRouter.
#
# Config files (gitignored — copy from .example.yaml first):
#   cp testing_suite/instant_playlist_test_config.example.yaml \
#      testing_suite/instant_playlist_test_config.yaml
#   cp testing_suite/ai_naming_test_config.example.yaml \
#      testing_suite/ai_naming_test_config.yaml
#
# Provider setup (in each YAML under "defaults"):
#   Ollama (local) – no API key needed, just run Ollama
#   OpenRouter     – set defaults.openrouter.api_key
#
# To disable a model, set enabled: false in the YAML.
# If no local config exists the scripts fall back to the
# .example.yaml automatically.

# ============================================================
# MULTI-PROVIDER TEST STACK (Docker-based)
# ============================================================

# Start test providers (Jellyfin, Navidrome)
cd test/provider_testing_stack
docker compose -f docker-compose-test-providers.yaml up -d

# Start AudioMuse test instance
docker compose -f docker-compose-test-audiomuse.yaml up -d

# See TEST_GUIDE.md for detailed instructions
```

### Test Markers

```bash
pytest -m unit -v          # Unit tests only
pytest -m integration -v   # Integration tests only
pytest -m "not slow" -v    # Skip slow tests
```

### URL Prefix Note

Some unit tests register blueprints **without** `url_prefix` for simplicity (e.g., `test_app_chat.py` tests `/api/config_defaults`). In production, these routes have prefixes:
- `chat_bp` → `/chat/...` (e.g., `/chat/api/config_defaults`)
- `external_bp` → `/external/...` (e.g., `/external/get_score`)

The endpoint paths in this checklist reflect **production** URLs.

### Test Dependencies

```bash
# Unit tests
pip install pytest>=7.0.0

# Integration tests
pip install -r test/requirements.txt

# Comparison suite
pip install -r testing_suite/requirements.txt
```

---

## 2. Automated vs Manual Testing Summary

### Can Be Automated (Unit Tests)

| Area | Tests | Status |
|------|-------|--------|
| MCP tool logic (genre regex, brainstorm matching, relevance scoring) | 40+ | **NEW** |
| AI MCP client (system prompt, tool defs, provider dispatch, energy conversion) | 60+ | **NEW** |
| Playlist ordering (greedy NN, Circle of Fifths, energy arc) | 25+ | **NEW** |
| Setup wizard (provider CRUD, settings, validation) | 30+ | **NEW** |
| Instant playlist pipeline (iteration loop, diversity, sampling) | 35+ | **NEW** |
| LocalFiles provider (hashing, metadata, M3U) | 25+ | **NEW** |
| Energy normalization (0-1 to raw conversion) | 10+ | **NEW** |
| Config validation (defaults, env parsing) | 10+ | **NEW** |
| Existing core tests (analysis, clustering, voyager, AI) | 200+ | Existing |

### Can Be Automated (Integration Tests)

| Area | Tests | Status |
|------|-------|--------|
| API endpoint responses (status codes, JSON shape) | 50+ | Partially exists |
| Provider connection testing | 5+ | Via test stack |
| Cross-provider ID remapping | 5+ | Via test stack |
| Database schema migration | 5+ | Via test stack |

### Requires Manual Testing

| Area | Why Manual | Steps |
|------|-----------|-------|
| Setup Wizard UI flow | Multi-step interactive wizard | See [Section 4](#4-gui-setup-wizard) |
| Dark mode visual correctness | Visual inspection of 18 templates | See [Section 12](#12-dark-mode) |
| Sidebar navigation | Interactive menu behavior | See [Section 7](#7-app-interactions-uiux) |
| Chart.js dark mode colors | Canvas-rendered, no DOM assertion | See [Section 12](#12-dark-mode) |
| Provider-specific playlist creation | Requires real media servers | See [Section 10](#10-provider-specific-testing) |
| AI quality assessment | Subjective playlist quality | See [Section 8](#8-instant-playlist--ai-changes) |
| Docker deployment | Full stack spin-up | See [Section 15](#15-deployment--docker) |
| Instant playlist UX | Streaming response, progress display | See [Section 8](#8-instant-playlist--ai-changes) |

---

## 3. Multi-Provider Architecture

### 3.1 Fresh Install (No Existing Data)

| # | Test Case | Type | Steps | Expected |
|---|-----------|------|-------|----------|
| 3.1.1 | First-run redirect | Auto | Start app with empty DB | Redirects to `/setup` |
| 3.1.2 | Provider table creation | Auto | Check DB after `init_db()` | `provider` table exists with correct schema |
| 3.1.3 | Settings table creation | Auto | Check DB after `init_db()` | `app_settings` table exists |
| 3.1.4 | Add Jellyfin provider | Manual | Setup wizard: select Jellyfin, enter URL/token/user | Provider saved, connection test passes |
| 3.1.5 | Add Navidrome provider | Manual | Setup wizard: select Navidrome, enter URL/user/pass | Provider saved, connection test passes |
| 3.1.6 | Add Lyrion provider | Manual | Setup wizard: select Lyrion, enter URL | Provider saved, connection test passes |
| 3.1.7 | Add Emby provider | Manual | Setup wizard: select Emby, enter URL/token/user | Provider saved, connection test passes |
| 3.1.8 | Add LocalFiles provider | Manual | Setup wizard: select LocalFiles, enter music dir | Provider saved, directory scan succeeds |
| 3.1.9 | Multiple providers | Manual | Add 2+ providers of different types | All listed, all enabled |
| 3.1.10 | Provider priority ordering | Auto | Add providers with different priorities | Returned in priority order |
| 3.1.11 | Duplicate provider rejection | Auto | Add same type+name twice | Returns error, no duplicate |
| 3.1.12 | music_path_prefix detection | Manual | Add provider, click auto-detect | Correct prefix detected from sample tracks |

### 3.2 Migration (Existing Single-Provider Data)

| # | Test Case | Type | Steps | Expected |
|---|-----------|------|-------|----------|
| 3.2.1 | Existing env vars preserved | Auto | Start with existing `.env` (JELLYFIN_*) | Config values still work |
| 3.2.2 | Setup wizard shows on upgrade | Manual | Upgrade from main, start app | Redirects to `/setup` once |
| 3.2.3 | Existing score data intact | Auto | Check `score` table after migration | All existing rows preserved |
| 3.2.4 | New columns added | Auto | Check `score` table schema | `album_artist`, `year`, `rating`, `file_path` columns exist |
| 3.2.5 | New columns nullable | Auto | Check existing rows | New columns are NULL for old data |
| 3.2.6 | Re-analysis populates new fields | Manual | Run analysis on existing library | New fields populated |
| 3.2.7 | Cross-provider file_path linking | Auto | Analyze same track via 2 providers | `find_existing_analysis_by_file_path()` finds match |
| 3.2.8 | Analysis reuse via file_path | Auto | Mock existing analysis, add new provider | `copy_analysis_to_new_item()` copies instead of re-analyzing |

### 3.3 Provider CRUD API

| # | Test Case | Type | Endpoint | Expected |
|---|-----------|------|----------|----------|
| 3.3.1 | List providers (empty) | Auto | `GET /api/setup/providers` | `[]` |
| 3.3.2 | Add provider | Auto | `POST /api/setup/providers` | 201, provider returned |
| 3.3.3 | Get provider by ID | Auto | `GET /api/setup/providers/<id>` | Provider details returned |
| 3.3.4 | Update provider | Auto | `PUT /api/setup/providers/<id>` | Updated fields reflected |
| 3.3.5 | Delete provider | Auto | `DELETE /api/setup/providers/<id>` | 200, provider removed |
| 3.3.6 | Test connection (by ID) | Auto | `POST /api/setup/providers/<id>/test` | `{"success": true}` |
| 3.3.7 | Test connection (inline) | Auto | `POST /api/setup/providers/test` with config | `{"success": true}` or `{"success": false, "error": "..."}` |
| 3.3.8 | Get libraries | Manual | `POST /api/setup/providers/libraries` | Library list returned |
| 3.3.9 | Rescan paths | Manual | `POST /api/setup/providers/<id>/rescan-paths` | Track list with file paths |
| 3.3.10 | Get enabled providers | Auto | `GET /api/providers/enabled` | Only enabled providers |
| 3.3.11 | Invalid provider type | Auto | `POST /api/setup/providers` with bad type | 400 error |
| 3.3.12 | Missing required fields | Auto | `POST /api/setup/providers` incomplete | 400 error |
| 3.3.13 | Get provider types | Auto | `GET /api/setup/providers/types` | List of supported provider types |
| 3.3.14 | Multi-provider config | Auto | `POST /api/setup/multi-provider` | Multi-provider setup applied |
| 3.3.15 | Set primary provider | Auto | `PUT /api/setup/primary-provider` | Primary provider updated |
| 3.3.16 | Server info | Auto | `GET /api/setup/server-info` | Server configuration returned |
| 3.3.17 | Browse directories | Manual | `GET /api/setup/browse-directories` | Directory listing returned |
| 3.3.18 | Complete setup | Auto | `POST /api/setup/complete` | Setup marked as complete |

### 3.4 Multi-Provider Playlist Creation

| # | Test Case | Type | Steps | Expected |
|---|-----------|------|-------|----------|
| 3.4.1 | Single provider playlist | Auto | `create_playlist_from_ids(ids, provider_ids=1)` | Playlist on provider 1 only |
| 3.4.2 | All providers playlist | Auto | `create_playlist_from_ids(ids, provider_ids='all')` | Playlist on all enabled providers |
| 3.4.3 | Specific providers list | Auto | `create_playlist_from_ids(ids, provider_ids=[1,3])` | Playlist on providers 1 and 3 |
| 3.4.4 | Cross-provider ID remapping | Auto | Create playlist with IDs from provider A on provider B | file_path hash lookup maps IDs correctly |
| 3.4.5 | Unmapped track handling | Auto | Create playlist with track missing on target provider | Track skipped, warning logged |
| 3.4.6 | Provider selector UI | Manual | Open instant playlist, select target providers | Dropdown shows enabled providers |

---

## 4. GUI Setup Wizard

> **All Manual** - Interactive multi-step wizard UI

| # | Test Case | Steps | Expected |
|---|-----------|-------|----------|
| 4.1 | Wizard loads on first run | Navigate to app URL | Setup wizard renders with welcome step |
| 4.2 | Step 1: Welcome | Read welcome text, click Next | Advances to provider selection |
| 4.3 | Step 2: Provider selection | Select provider type from dropdown | Configuration form appears for selected type |
| 4.4 | Step 2: Jellyfin config form | Select Jellyfin | Shows URL, User ID, Token fields |
| 4.5 | Step 2: Navidrome config form | Select Navidrome | Shows URL, Username, Password fields |
| 4.6 | Step 2: Lyrion config form | Select Lyrion | Shows URL field |
| 4.7 | Step 2: Emby config form | Select Emby | Shows URL, User ID, Token fields |
| 4.8 | Step 2: LocalFiles config form | Select LocalFiles | Shows Music Directory, Formats, Scan Subdirs fields |
| 4.9 | Step 3: Connection test success | Enter valid credentials, click Test | Green checkmark, "Connection successful" |
| 4.10 | Step 3: Connection test failure | Enter invalid credentials, click Test | Red X, error message displayed |
| 4.11 | Step 3: Library discovery | After successful test | Music libraries listed for selection |
| 4.12 | Step 3: Path prefix auto-detect | Click auto-detect button | Prefix field populated |
| 4.13 | Step 4: Add another provider | Click "Add Another Provider" | Returns to provider selection |
| 4.14 | Step 5: Complete setup | Click Complete | Redirects to main app, setup marked complete |
| 4.15 | Wizard skipped after setup | Return to app after completion | No redirect to `/setup` |
| 4.16 | Settings page access | Navigate to `/settings` | Settings page loads with current config |
| 4.17 | Settings: update AI provider | Change AI provider dropdown | Saves, applies to next instant playlist |
| 4.18 | Settings: update clustering | Change clustering algorithm | Saves, applies to next clustering run |
| 4.19 | Settings: disable provider | Toggle provider off | Provider excluded from playlist creation |
| 4.20 | Settings: re-enable provider | Toggle provider back on | Provider included in playlist creation |
| 4.21 | Form validation | Submit empty required fields | Client-side validation error shown |
| 4.22 | XSS prevention | Enter `<script>` in provider name | Escaped in display, not executed |

---

## 5. Environment / Config Setup

### 5.1 Config Variable Validation (Automated)

| # | Test Case | Type | Variable | Expected |
|---|-----------|------|----------|----------|
| 5.1.1 | Default MEDIASERVER_TYPE | Auto | `MEDIASERVER_TYPE` not set | Defaults to `localfiles` |
| 5.1.2 | Valid MEDIASERVER_TYPE | Auto | `MEDIASERVER_TYPE=jellyfin` | Accepted |
| 5.1.3 | LocalFiles defaults | Auto | No LOCALFILES_* vars | Music dir `/music`, all formats, scan subdirs |
| 5.1.4 | LOCALFILES_FORMATS parsing | Auto | `LOCALFILES_FORMATS=.mp3,.flac` | Only mp3/flac accepted |
| 5.1.5 | MAX_SONGS_PER_ARTIST_PLAYLIST | Auto | Not set | Defaults to 5 |
| 5.1.6 | PLAYLIST_ENERGY_ARC | Auto | Not set | Defaults to False |
| 5.1.7 | ENERGY_MIN/MAX range | Auto | Default values | 0.01 / 0.15 |
| 5.1.8 | AI_REQUEST_TIMEOUT_SECONDS | Auto | Not set | Defaults to 300 |
| 5.1.9 | MPD vars removed | Auto | `MPD_HOST` in config | Should NOT exist |

### 5.2 Environment File

| # | Test Case | Type | Steps | Expected |
|---|-----------|------|-------|----------|
| 5.2.1 | .env.example complete | Auto | Compare .env.example to config.py | All variables documented |
| 5.2.2 | .env.example defaults work | Manual | Copy .env.example as .env, start app | App starts (with setup wizard) |
| 5.2.3 | No .env file | Manual | Start without .env | App starts with sensible defaults |

### 5.3 Settings Persistence

| # | Test Case | Type | Steps | Expected |
|---|-----------|------|-------|----------|
| 5.3.1 | Settings saved to DB | Auto | `PUT /api/setup/settings` | Saved in `app_settings` table |
| 5.3.2 | Settings loaded on startup | Auto | Restart app | `apply_settings_to_config()` overrides config |
| 5.3.3 | Settings override env vars | Auto | Set env var AND DB setting | DB setting takes precedence |

---

## 6. API Endpoints

### 6.1 Endpoints Without Provider (No media server configured)

| # | Test Case | Type | Endpoint | Expected |
|---|-----------|------|----------|----------|
| 6.1.1 | Homepage renders | Auto | `GET /` | 200, HTML page (or redirect to setup) |
| 6.1.2 | Config endpoint | Auto | `GET /api/config` | 200, JSON with app config |
| 6.1.3 | Status endpoint (bad ID) | Auto | `GET /api/status/nonexistent` | 404 |
| 6.1.4 | Active tasks (none) | Auto | `GET /api/active_tasks` | 200, empty/null |
| 6.1.5 | Cancel nonexistent task | Auto | `POST /api/cancel/nonexistent` | 404 |
| 6.1.6 | Playlists (empty) | Auto | `GET /api/playlists` | 200, `[]` |
| 6.1.7 | Setup status | Auto | `GET /api/setup/status` | 200, setup complete/incomplete |
| 6.1.8 | Providers list (empty) | Auto | `GET /api/setup/providers` | 200, `[]` |
| 6.1.9 | Settings GET | Auto | `GET /api/setup/settings` | 200, current settings |
| 6.1.10 | Config defaults | Auto | `GET /chat/api/config_defaults` | 200, AI provider defaults |
| 6.1.11 | Similarity page | Auto | `GET /similarity` | 200, HTML |
| 6.1.12 | Alchemy page | Auto | `GET /alchemy` | 200, HTML |
| 6.1.13 | Map page | Auto | `GET /map` | 200, HTML |
| 6.1.14 | Path page | Auto | `GET /path` | 200, HTML |
| 6.1.15 | Cron page | Auto | `GET /cron` | 200, HTML |
| 6.1.16 | Cleaning page | Auto | `GET /cleaning` | 200, HTML |

### 6.2 Endpoints With Provider (Media server configured, tracks analyzed)

| # | Test Case | Type | Endpoint | Expected |
|---|-----------|------|----------|----------|
| 6.2.1 | Search tracks | Auto | `GET /api/search_tracks?title=test` | 200, array of matches |
| 6.2.2 | Similar tracks by ID | Auto | `GET /api/similar_tracks?item_id=X` | 200, similar tracks array |
| 6.2.3 | Similar tracks by title+artist | Auto | `GET /api/similar_tracks?title=X&artist=Y` | 200, similar tracks array |
| 6.2.4 | Get score | Auto | `GET /external/get_score?id=X` | 200, track metadata incl. new fields |
| 6.2.5 | Get embedding | Auto | `GET /external/get_embedding?id=X` | 200, embedding vector |
| 6.2.6 | Artist search | Auto | `GET /api/search_artists?query=X` | 200, artist matches |
| 6.2.7 | Similar artists | Auto | `GET /api/similar_artists?artist=X` | 200, similar artists |
| 6.2.8 | Alchemy search | Auto | `POST /api/alchemy` with items | 200, results array |
| 6.2.9 | Path finding | Auto | `GET /api/find_path?start_song_id=X&end_song_id=Y` | 200, path array |
| 6.2.10 | Map data | Auto | `GET /api/map?percent=100` | 200, items with projections |
| 6.2.11 | Map cache status | Auto | `GET /api/map_cache_status` | 200, cache info |
| 6.2.12 | CLAP search (if enabled) | Auto | `POST /api/clap/search` with query | 200, results |
| 6.2.13 | MuLan search (if enabled) | Auto | `POST /api/mulan/search` with query | 200, results |
| 6.2.14 | Chat playlist | Manual | `POST /chat/api/chatPlaylist` | 200, streaming playlist |
| 6.2.15 | Start analysis | Auto | `POST /api/analysis/start` | 202, task_id |
| 6.2.16 | Start clustering | Auto | `POST /api/clustering/start` | 202, task_id |
| 6.2.17 | Cron CRUD | Auto | `POST/GET/DELETE /api/cron` | Cron entry management |
| 6.2.18 | Sonic fingerprint | Manual | `POST /api/sonic_fingerprint/generate` | 200, fingerprint results |

### 6.3 Error Handling

| # | Test Case | Type | Endpoint | Expected |
|---|-----------|------|----------|----------|
| 6.3.1 | Missing required params | Auto | `GET /api/similar_tracks` (no params) | 400 |
| 6.3.2 | Invalid item_id | Auto | `GET /api/similar_tracks?item_id=INVALID` | 404 |
| 6.3.3 | Task conflict | Auto | Start clustering twice | 409 on second call |
| 6.3.4 | Invalid JSON body | Auto | `POST /api/alchemy` with bad JSON | 400 |
| 6.3.5 | Voyager index not ready | Auto | Query before analysis | 503 |

---

## 7. App Interactions (UI/UX)

> **All Manual** - Browser-based interaction testing

| # | Test Case | Steps | Expected |
|---|-----------|-------|----------|
| 7.1 | Sidebar navigation | Click each nav item | Correct page loads, active class applied |
| 7.2 | Sidebar collapse/expand | Click hamburger menu | Sidebar toggles, state persists via localStorage |
| 7.3 | Sidebar state persistence | Toggle sidebar, navigate to another page | Sidebar state preserved |
| 7.4 | Settings link in sidebar | Click Settings | Settings page loads |
| 7.5 | Setup Wizard link in sidebar | Click Setup Wizard | Setup wizard loads |
| 7.6 | Provider selector dropdown | Open instant playlist page | Provider dropdown shows enabled providers |
| 7.7 | Provider selector in clustering | Open clustering page | Provider selector available |
| 7.8 | Real-time task progress | Start analysis or clustering | Progress bar updates via polling |
| 7.9 | Task cancellation | Click cancel during a task | Task cancelled, status updated |
| 7.10 | Error notifications | Trigger an error (e.g., bad connection) | Error toast/message displayed |
| 7.11 | Chart.js visualizations | Open map, clustering results | Charts render correctly |
| 7.12 | Chart.js dark mode | Toggle dark mode on chart pages | Grid colors update to dark theme |
| 7.13 | Responsive layout | Resize browser window | Layout adapts, no overflow |
| 7.14 | escapeHtml XSS prevention | Track with `<script>` in title | Rendered as text, not executed |

---

## 8. Instant Playlist & AI Changes

### 8.1 Agentic Loop (Automated)

| # | Test Case | Type | Details |
|---|-----------|------|---------|
| 8.1.1 | Iteration 0: initial request | Auto | User input passed as-is, system prompt includes library context |
| 8.1.2 | Iteration 0: fallback on AI error | Auto | On exception, falls back to `search_database(top 2 genres)` |
| 8.1.3 | Iteration 1+: rich feedback | Auto | Feedback includes progress %, top artists, diversity, tools used |
| 8.1.4 | Stop at 100 songs | Auto | Loop exits when `len(all_songs) >= 100` |
| 8.1.5 | Stop on no new songs | Auto | Loop exits when `iteration_songs_added == 0` |
| 8.1.6 | Stop at max iterations | Auto | Loop exits after iteration 4 (5 total) |
| 8.1.7 | Deduplication by item_id | Auto | Same song from different tools not counted twice |
| 8.1.8 | Tool call tracking | Auto | Each song tracks which tool call produced it |

### 8.2 Pre-Execution Validation (Automated)

| # | Test Case | Type | Details |
|---|-----------|------|---------|
| 8.2.1 | Reject song_similarity without title | Auto | Empty `song_title` → rejected, logged |
| 8.2.2 | Reject song_similarity without artist | Auto | Empty `song_artist` → rejected, logged |
| 8.2.3 | Reject search_database no filters | Auto | No genres/moods/tempo/energy/key/scale/year/rating → rejected |
| 8.2.4 | Accept search_database with one filter | Auto | Only `genres` set → accepted |

### 8.3 Proportional Sampling (Automated)

| # | Test Case | Type | Details |
|---|-----------|------|---------|
| 8.3.1 | No sampling under 100 songs | Auto | 80 songs collected → all 80 returned |
| 8.3.2 | Proportional reduction over 100 | Auto | 200 songs (100 from tool A, 100 from tool B) → 50 each |
| 8.3.3 | Minimum 1 per tool | Auto | 150 songs (140 from A, 10 from B) → B gets at least 1 |
| 8.3.4 | Randomized after sampling | Auto | Output is shuffled |

### 8.4 Artist Diversity (Automated)

| # | Test Case | Type | Details |
|---|-----------|------|---------|
| 8.4.1 | Cap at MAX_SONGS_PER_ARTIST | Auto | Artist with 10 songs → only 5 kept (default) |
| 8.4.2 | Backfill from overflow | Auto | Excess songs removed, backfill from least-represented artists |
| 8.4.3 | Backfill prioritizes least represented | Auto | Artist with 1 song backfilled before artist with 3 |
| 8.4.4 | No backfill needed | Auto | All artists under limit → no changes |
| 8.4.5 | Unknown artist handling | Auto | Songs with no artist field → treated as "Unknown" |

### 8.5 System Prompt (Automated)

| # | Test Case | Type | Details |
|---|-----------|------|---------|
| 8.5.1 | _build_system_prompt includes library context | Auto | Total songs, artists, genres, moods in prompt |
| 8.5.2 | Energy scale documented as 0-1 | Auto | Prompt mentions "0.0 (calm) to 1.0 (intense)" |
| 8.5.3 | Dynamic genre list from DB | Auto | Top 5 genres from `get_library_context()` in prompt |
| 8.5.4 | CLAP tool included when enabled | Auto | `text_search` in tool list when `CLAP_ENABLED=true` |
| 8.5.5 | CLAP tool excluded when disabled | Auto | `text_search` NOT in tool list when `CLAP_ENABLED=false` |

### 8.6 AI MCP Client (Automated - `test_ai_mcp_client.py`)

| # | Test Case | Type | Details |
|---|-----------|------|---------|
| 8.6.1 | Prompt includes all tool names | Auto | All 5-6 tool names in prompt text |
| 8.6.2 | CLAP decision tree has 6 steps | Auto | With CLAP enabled, 6 steps in decision tree |
| 8.6.3 | No-CLAP decision tree has 5 steps | Auto | With CLAP disabled, 5 steps |
| 8.6.4 | Library context injected | Auto | Total songs, artists, genres in prompt |
| 8.6.5 | No library section when None | Auto | Graceful handling of missing context |
| 8.6.6 | Dynamic genres from context | Auto | Top genres from DB in prompt |
| 8.6.7 | Dynamic moods from context | Auto | Top moods from DB in prompt |
| 8.6.8 | Fallback genres when no context | Auto | Default genre list used |
| 8.6.9 | Year range shown | Auto | Min/max year in prompt |
| 8.6.10 | Rating info shown | Auto | Rating coverage % in prompt |
| 8.6.11 | 6 tools with CLAP | Auto | `get_mcp_tools()` returns 6 |
| 8.6.12 | 5 tools without CLAP | Auto | `get_mcp_tools()` returns 5 |
| 8.6.13 | Tools have required keys | Auto | name, description, parameters present |
| 8.6.14 | search_database filter properties | Auto | Genre, mood, tempo, energy, key, scale, year, rating |
| 8.6.15 | Energy 0 → ENERGY_MIN | Auto | `execute_mcp_tool` converts correctly |
| 8.6.16 | Energy 1 → ENERGY_MAX | Auto | `execute_mcp_tool` converts correctly |
| 8.6.17 | Energy 0.5 → midpoint | Auto | Linear interpolation |
| 8.6.18 | No energy args → None | Auto | Missing args pass through as None |
| 8.6.19 | Unknown tool → error | Auto | Returns error dict |
| 8.6.20 | song_alchemy normalizes strings | Auto | Plain strings → dicts |
| 8.6.21 | Provider dispatch routing | Auto | Gemini/OpenAI/Mistral/Ollama dispatch correctly |
| 8.6.22 | Unknown provider → error | Auto | Returns error dict |
| 8.6.23 | Ollama JSON parsing | Auto | Valid JSON tool_calls extracted |
| 8.6.24 | Ollama markdown stripping | Auto | Code blocks stripped before parsing |
| 8.6.25 | Ollama schema detection rejected | Auto | Schema-like response → error |
| 8.6.26 | Ollama JSON decode error | Auto | Malformed JSON → error |
| 8.6.27 | Ollama read timeout | Auto | httpx.ReadTimeout → error |
| 8.6.28 | Gemini missing API key | Auto | Empty key → error |
| 8.6.29 | Gemini schema type conversion | Auto | JSON schema → Gemini types |
| 8.6.30 | OpenAI tool call extraction | Auto | Standard OpenAI format parsed |
| 8.6.31 | OpenAI read timeout | Auto | httpx.ReadTimeout → error |
| 8.6.32 | Mistral missing API key | Auto | Empty/placeholder key → error |
| 8.6.33 | Mistral tool call extraction | Auto | Mistral SDK format parsed |

### 8.7 AI Provider Integration (Manual + Automated)

| # | Test Case | Type | Provider | Details |
|---|-----------|------|----------|---------|
| 8.7.1 | Gemini tool calling | Manual | Gemini | Sends request, receives tool calls, executes them |
| 8.7.2 | Gemini protobuf handling | Auto | Gemini | `_convert_protobuf_to_dict()` converts correctly |
| 8.7.3 | Gemini fc.args vs fc.arguments | Auto | Gemini | Both formats parsed correctly |
| 8.7.4 | OpenAI tool calling | Manual | OpenAI | Standard tool calling works |
| 8.7.5 | OpenAI timeout handling | Auto | OpenAI | `httpx.ReadTimeout` caught and handled |
| 8.7.6 | Mistral tool calling | Manual | Mistral | SDK-based tool calling works |
| 8.7.7 | Ollama JSON extraction | Auto | Ollama | Markdown code blocks stripped, JSON parsed |
| 8.7.8 | Ollama edge case: text before JSON | Auto | Ollama | `Here is my response: {"tool_calls":...}` → parsed |
| 8.7.9 | Ollama edge case: schema in response | Auto | Ollama | `{"tool_calls": {"_description": ...}}` → rejected |
| 8.7.10 | Ollama edge case: invalid JSON | Auto | Ollama | Malformed JSON → empty tool_calls |
| 8.7.11 | AI_MODEL_PROVIDER=NONE | Auto | None | Playlist naming skipped |
| 8.7.12 | Full pipeline per provider | Manual | All | End-to-end playlist generation with each provider |

### 8.8 Library Context (Automated)

| # | Test Case | Type | Details |
|---|-----------|------|---------|
| 8.8.1 | get_library_context returns stats | Auto | Total songs, unique artists, top genres, etc. |
| 8.8.2 | Caching works | Auto | Second call returns cached result |
| 8.8.3 | Empty library handling | Auto | No songs in DB → graceful defaults |
| 8.8.4 | Year range calculation | Auto | Min/max year from DB |
| 8.8.5 | Rating coverage percentage | Auto | % of tracks with rating > 0 |

---

## 9. MCP Tools

### 9.1 song_similarity (Automated)

| # | Test Case | Details |
|---|-----------|---------|
| 9.1.1 | Exact title+artist match | Case-insensitive DB lookup succeeds |
| 9.1.2 | Fuzzy normalized match | "dont stop believin" matches "Don't Stop Believin'" |
| 9.1.3 | No match found | Returns empty list, no crash |
| 9.1.4 | Seed song excluded from results | Seed not in similar list |
| 9.1.5 | Result count respects `get_songs` | Returns exactly N songs |

### 9.2 text_search (Automated)

| # | Test Case | Details |
|---|-----------|---------|
| 9.2.1 | Basic query | "upbeat pop" returns results |
| 9.2.2 | Tempo filter: slow | Only 40-100 BPM tracks |
| 9.2.3 | Tempo filter: fast | Only 140-200 BPM tracks |
| 9.2.4 | Energy filter: low (raw) | Only 0.01-0.05 energy tracks |
| 9.2.5 | Energy filter: high (raw) | Only 0.10-0.15 energy tracks |
| 9.2.6 | Disabled when CLAP_ENABLED=false | Returns error/empty |

### 9.3 artist_similarity (Automated)

| # | Test Case | Details |
|---|-----------|---------|
| 9.3.1 | Known artist | Returns similar artists + songs |
| 9.3.2 | Unknown artist | Returns error message |
| 9.3.3 | Artist name normalization | "ac dc" matches "AC/DC" |
| 9.3.4 | Component matches breakdown | Shows count from original vs similar |

### 9.4 song_alchemy (Automated)

| # | Test Case | Details |
|---|-----------|---------|
| 9.4.1 | Two add items | Centroid calculated, neighbors found |
| 9.4.2 | Add + subtract items | Subtract centroid applied |
| 9.4.3 | Single add item rejected | Requires >= 2 items |
| 9.4.4 | Plain string normalization | AI sends strings → converted to dicts |

### 9.5 ai_brainstorm (Automated)

| # | Test Case | Details |
|---|-----------|---------|
| 9.5.1 | Stage 1: exact match | "Bohemian Rhapsody" by "Queen" → match |
| 9.5.2 | Stage 2: fuzzy match | "Don't Stop" matches "Dont Stop" |
| 9.5.3 | Both title AND artist required | Title match with wrong artist → no match |
| 9.5.4 | Normalization: apostrophes | "it's" → "its" |
| 9.5.5 | Normalization: dashes | "up-beat" → "upbeat" |
| 9.5.6 | Normalization: spaces | "The Beatles" → "thebeatles" |
| 9.5.7 | Deduplication | Same normalized title+artist → first kept |
| 9.5.8 | Child AI failure | Exception → empty results |
| 9.5.9 | SQL injection prevention | Parameterized queries with LIKE escaping |

### 9.6 search_database (Automated)

| # | Test Case | Details |
|---|-----------|---------|
| 9.6.1 | Genre regex: exact match | `rock` matches `rock:0.82,...` |
| 9.6.2 | Genre regex: no substring | `rock` does NOT match `indie rock:0.31,...` |
| 9.6.3 | Genre regex: after comma | `rock` matches `pop:0.45,rock:0.82` |
| 9.6.4 | Relevance scoring | Higher confidence genres ranked first |
| 9.6.5 | Multiple genres (OR) | `["rock", "pop"]` → tracks with either |
| 9.6.6 | Mood filtering | `["danceable"]` → tracks with that mood |
| 9.6.7 | Tempo range | `tempo_min=120, tempo_max=140` → BPM range |
| 9.6.8 | Energy normalization | AI sends 0.5 → converted to 0.08 raw |
| 9.6.9 | Key filter | `key="C"` → only C key tracks |
| 9.6.10 | Scale filter | `scale="major"` → only major scale |
| 9.6.11 | Year range | `year_min=1980, year_max=1989` → 80s songs |
| 9.6.12 | Rating minimum | `min_rating=4` → 4+ star tracks |
| 9.6.13 | Combined filters (AND) | Genre + tempo + energy → intersection |
| 9.6.14 | Empty result handling | Filters match nothing → empty list |
| 9.6.15 | No filters rejected | No params → rejected at validation |

---

## 10. Provider-Specific Testing

### 10.1 Common Provider Interface (Per Provider)

> **Manual** - Requires running media server instances

For EACH provider (Jellyfin, Navidrome, Lyrion, Emby, LocalFiles):

| # | Test Case | Steps | Expected |
|---|-----------|-------|----------|
| 10.1.1 | Connection test | Call `test_provider_connection()` | Returns True |
| 10.1.2 | Get music libraries | Call `get_music_libraries()` | Library list returned |
| 10.1.3 | Get all songs | Call `get_all_songs()` | Song list with all fields |
| 10.1.4 | Song has album_artist | Check returned song dict | `OriginalAlbumArtist` field present |
| 10.1.5 | Song has year | Check returned song dict | `Year` field present (int or None) |
| 10.1.6 | Song has rating | Check returned song dict | `Rating` field (0-5 scale) |
| 10.1.7 | Song has file_path | Check returned song dict | `FilePath` field present |
| 10.1.8 | Get recent albums | Call `get_recent_albums(10)` | 10 albums returned |
| 10.1.9 | Get tracks from album | Call `get_tracks_from_album(album_id)` | Track list returned |
| 10.1.10 | Download track | Call `download_track(temp_dir, item)` | File downloaded, path returned |
| 10.1.11 | Get all playlists | Call `get_all_playlists()` | Playlist list returned |
| 10.1.12 | Create playlist | Call `create_playlist("Test", [ids])` | Playlist created on server |
| 10.1.13 | Delete playlist | Call `delete_playlist(id)` | Playlist removed from server |
| 10.1.14 | Create instant playlist | Call `create_instant_playlist(...)` | Playlist created with metadata |

### 10.2 LocalFiles Provider (Automated)

| # | Test Case | Type | Details |
|---|-----------|------|---------|
| 10.2.1 | SHA-256 item ID generation | Auto | Same path → same hash, different path → different hash |
| 10.2.2 | Path normalization | Auto | Windows backslashes → forward slashes |
| 10.2.3 | Supported format filtering | Auto | `.mp3`, `.flac` accepted; `.txt` rejected |
| 10.2.4 | Subdirectory scanning | Auto | `LOCALFILES_SCAN_SUBDIRS=true` → recursive scan |
| 10.2.5 | No subdirectory scanning | Auto | `LOCALFILES_SCAN_SUBDIRS=false` → top-level only |
| 10.2.6 | ID3 tag extraction | Auto | MP3 with ID3 tags → title, artist, album extracted |
| 10.2.7 | Vorbis tag extraction | Auto | FLAC/OGG with Vorbis → metadata extracted |
| 10.2.8 | MP4 tag extraction | Auto | M4A with MP4 tags → metadata extracted |
| 10.2.9 | Rating from POPM tag | Auto | POPM rating → 0-5 scale |
| 10.2.10 | Rating from TXXX:RATING | Auto | TXXX rating → 0-5 scale |
| 10.2.11 | Year parsing: simple | Auto | "2023" → 2023 |
| 10.2.12 | Year parsing: date string | Auto | "2023-05-01" → 2023 |
| 10.2.13 | Year parsing: ID3 TDRC | Auto | TDRC frame → year extracted |
| 10.2.14 | M3U playlist creation | Auto | Create playlist → .m3u file written |
| 10.2.15 | M3U playlist listing | Auto | List playlists → .m3u files found |
| 10.2.16 | M3U playlist deletion | Auto | Delete → .m3u file removed |
| 10.2.17 | Empty directory handling | Auto | No music files → empty list, no crash |
| 10.2.18 | Non-existent directory | Auto | Bad path → error returned |

---

## 11. Database & Schema Changes

### 11.1 Schema Migration (Automated)

| # | Test Case | Type | Details |
|---|-----------|------|---------|
| 11.1.1 | `provider` table created | Auto | `init_db()` creates table |
| 11.1.2 | `app_settings` table created | Auto | `init_db()` creates table |
| 11.1.3 | `album_artist` column added | Auto | `ALTER TABLE score ADD COLUMN IF NOT EXISTS` |
| 11.1.4 | `year` column added | Auto | Integer column |
| 11.1.5 | `rating` column added | Auto | Integer column (0-5) |
| 11.1.6 | `file_path` column added | Auto | Text column |
| 11.1.7 | Idempotent migration | Auto | Running `init_db()` twice → no error |
| 11.1.8 | Existing data preserved | Auto | Old rows not affected by new columns |

### 11.2 Data Integrity (Automated)

| # | Test Case | Type | Details |
|---|-----------|------|---------|
| 11.2.1 | Score insert with new fields | Auto | All 4 new fields stored correctly |
| 11.2.2 | Score select includes new fields | Auto | `get_score_data_by_ids()` returns new fields |
| 11.2.3 | NULL new fields for old data | Auto | Pre-migration rows have NULL for new columns |
| 11.2.4 | Provider JSONB config | Auto | Config stored and retrieved as JSON |
| 11.2.5 | Provider unique constraint | Auto | Same type+name → integrity error |
| 11.2.6 | Settings upsert | Auto | Same key → updated, not duplicated |

---

## 12. Dark Mode

> **Mostly Manual** - Visual verification required

### 12.1 Toggle & Persistence

| # | Test Case | Type | Steps | Expected |
|---|-----------|------|-------|----------|
| 12.1.1 | Toggle button visible | Manual | Open any page | Sun/moon button in sidebar |
| 12.1.2 | Toggle to dark mode | Manual | Click toggle button | Body gets `dark-mode` class, colors change |
| 12.1.3 | Toggle back to light | Manual | Click toggle again | `dark-mode` class removed |
| 12.1.4 | localStorage persistence | Auto | Toggle dark, reload page | Still dark mode |
| 12.1.5 | System preference detection | Manual | Set OS to dark mode, clear localStorage | App starts in dark mode |
| 12.1.6 | No FOUC on load | Manual | Hard refresh (Ctrl+Shift+R) | No flash of wrong theme |
| 12.1.7 | FOUC prevention script | Auto | Check `layout.html` for inline script | Script in `<head>` before CSS |

### 12.2 Visual Correctness (All Manual)

For EACH of these 21 pages/templates, verify in BOTH light and dark mode:

| # | Page | Template | Check Items |
|---|------|----------|-------------|
| 12.2.1 | Homepage | `index.html` | Background, navigation cards, layout |
| 12.2.2 | Chat (instant playlist) | `chat.html` | Background, text, input fields, buttons, results cards |
| 12.2.3 | Similarity search | `similarity.html` | Background, search inputs, results table, cards |
| 12.2.4 | Alchemy | `alchemy.html` | Background, add/subtract buttons, results, projections |
| 12.2.5 | Artist similarity | `artist_similarity.html` | Background, search, results list |
| 12.2.6 | Path finding | `path.html` | Background, start/end selectors, path visualization |
| 12.2.7 | Map | `map.html` | Background, map container, tooltips |
| 12.2.8 | CLAP search | `clap_search.html` | Background, search input, results |
| 12.2.9 | MuLan search | `mulan_search.html` | Background, search input, results |
| 12.2.10 | Sonic fingerprint | `sonic_fingerprint.html` | Background, results, credentials form |
| 12.2.11 | Waveform | `waveform.html` | Background, waveform canvas, controls |
| 12.2.12 | Analysis | `script.html` | Background, progress display, status |
| 12.2.13 | Cleaning | `cleaning.html` | Background, status display |
| 12.2.14 | Cron | `cron.html` | Background, cron table, add form |
| 12.2.15 | Collection sync | `collection.html` | Background, sync status |
| 12.2.16 | Collection script | `collection_script.html` | Background, script display |
| 12.2.17 | Settings | `settings.html` | Background, form fields, dropdowns |
| 12.2.18 | Setup wizard | `setup.html` | Background, wizard steps, buttons |
| 12.2.19 | Sidebar navigation | `sidebar_navi.html` | Background, links, active state, toggle button |
| 12.2.20 | Chart.js charts | _(cross-cutting)_ | Grid lines, labels, data points use CSS variable colors |
| 12.2.21 | Error/loading states | _(cross-cutting)_ | Spinner, error messages visible in both themes |

### 12.3 CSS Variables (Automated)

| # | Test Case | Type | Details |
|---|-----------|------|---------|
| 12.3.1 | :root variables defined | Auto | All 30+ CSS custom properties in `:root` |
| 12.3.2 | body.dark-mode overrides | Auto | All variables overridden in dark mode |
| 12.3.3 | No hardcoded colors remaining | Auto | Grep for `#[0-9a-fA-F]{3,8}` outside `:root` |
| 12.3.4 | Transition smoothing | Auto | `transition: background 0.3s, color 0.3s` present |

---

## 13. Analysis Pipeline

### 13.1 Analysis with New Fields (Automated + Manual)

| # | Test Case | Type | Details |
|---|-----------|------|---------|
| 13.1.1 | Analysis stores album_artist | Auto | `save_track_analysis_and_embedding()` stores field |
| 13.1.2 | Analysis stores year | Auto | Year parsed and stored |
| 13.1.3 | Analysis stores rating | Auto | Rating normalized and stored |
| 13.1.4 | Analysis stores file_path | Auto | File path stored |
| 13.1.5 | Analysis stores provider_id | Auto | Provider ID stored |
| 13.1.6 | Cross-provider reuse | Auto | Same file_path → analysis copied, not re-run |
| 13.1.7 | Full analysis pipeline | Manual | Start analysis, wait for completion | All tracks analyzed with new fields |
| 13.1.8 | Queue drain before index rebuild | Auto | Analysis waits for all album jobs before Voyager rebuild |

### 13.2 Voyager Index (Automated)

| # | Test Case | Type | Details |
|---|-----------|------|---------|
| 13.2.1 | Index rebuild after analysis | Auto | New embeddings indexed |
| 13.2.2 | search_tracks returns album_artist | Auto | Query result includes new field |
| 13.2.3 | create_playlist_from_ids with provider_ids | Auto | Delegates to multi-provider creation |

---

## 14. Playlist Ordering

### 14.1 Greedy Nearest-Neighbor (Automated)

| # | Test Case | Type | Details |
|---|-----------|------|---------|
| 14.1.1 | Single song | Auto | Returns single song unchanged |
| 14.1.2 | Two songs | Auto | Both songs in output |
| 14.1.3 | Empty list | Auto | Returns empty list |
| 14.1.4 | Start from 25th percentile energy | Auto | First song is low-energy |
| 14.1.5 | Consecutive songs are similar | Auto | Adjacent songs have small composite distance |

### 14.2 Composite Distance (Automated)

| # | Test Case | Type | Details |
|---|-----------|------|---------|
| 14.2.1 | Tempo weight = 35% | Auto | BPM difference contributes 35% |
| 14.2.2 | Energy weight = 35% | Auto | Energy difference contributes 35% |
| 14.2.3 | Key weight = 30% | Auto | Key distance contributes 30% |
| 14.2.4 | Tempo normalization (80 BPM span) | Auto | 80 BPM diff = 1.0, 0 BPM diff = 0.0 |
| 14.2.5 | Energy normalization (0.14 span) | Auto | 0.14 diff = 1.0, 0 diff = 0.0 |
| 14.2.6 | Same scale bonus (20%) | Auto | Same scale → key distance * 0.8 |

### 14.3 Circle of Fifths Key Distance (Automated)

| # | Test Case | Type | Details |
|---|-----------|------|---------|
| 14.3.1 | Same key = 0 | Auto | C → C = 0.0 |
| 14.3.2 | Adjacent keys | Auto | C → G = 1/6, C → F = 1/6 |
| 14.3.3 | Opposite keys | Auto | C → F# = 6/6 = 1.0 |
| 14.3.4 | Enharmonic equivalents | Auto | F# == Gb |

### 14.4 Energy Arc (Automated)

| # | Test Case | Type | Details |
|---|-----------|------|---------|
| 14.4.1 | Arc disabled by default | Auto | `PLAYLIST_ENERGY_ARC=False` → no shaping |
| 14.4.2 | Arc enabled | Auto | Gentle start → peak → cooldown |
| 14.4.3 | Arc requires >= 10 songs | Auto | < 10 songs → no arc applied |

---

## 15. Deployment & Docker

> **All Manual** - Requires Docker environment

| # | Test Case | Steps | Expected |
|---|-----------|-------|----------|
| 15.1 | Unified CPU compose | `docker compose -f deployment/docker-compose-unified.yaml up` | App + worker start, health check passes |
| 15.2 | Unified NVIDIA compose | `docker compose -f deployment/docker-compose-unified-nvidia.yaml up` | GPU detected, CUDA available |
| 15.3 | Unified NVIDIA test compose | `docker compose -f deployment/docker-compose-unified-nvidia-test.yaml up` | Test variant starts |
| 15.4 | Split deployment: server | `docker compose -f deployment/docker-compose-server.yaml up` | API responds on port |
| 15.5 | Split deployment: worker CPU | `docker compose -f deployment/docker-compose-worker-cpu.yaml up` | Worker connects and processes jobs |
| 15.6 | Split deployment: worker NVIDIA | `docker compose -f deployment/docker-compose-worker-nvidia.yaml up` | GPU worker connects |
| 15.7 | DMR compose | `docker compose -f deployment/docker-compose-dmr.yaml up` | DMR variant starts |
| 15.8 | Unraid templates | Import XML templates | Containers configured correctly |
| 15.9 | .env.example works | Copy to .env, start | App starts with setup wizard |
| 15.10 | No `version:` in compose | Check all compose files | No deprecated `version: '3.8'` line |
| 15.11 | Test provider stack | Start test compose files | Jellyfin + Navidrome + AudioMuse running |

---

## 16. Regression Tests

### 16.1 Features That Must Still Work (from main)

| # | Test Case | Type | Details |
|---|-----------|------|---------|
| 16.1.1 | CLAP text search | Manual | Search by text description → relevant results |
| 16.1.2 | MuLan text search | Manual | Search by text → relevant results |
| 16.1.3 | Voyager similarity search | Auto | Song similarity returns correct neighbors |
| 16.1.4 | Song alchemy | Manual | Add/subtract songs → blended results |
| 16.1.5 | Song path finding | Manual | Start/end song → smooth path |
| 16.1.6 | Music map visualization | Manual | 2D projection renders |
| 16.1.7 | Artist similarity | Manual | Similar artists found |
| 16.1.8 | Sonic fingerprint | Manual | Listening profile generated |
| 16.1.9 | Waveform visualization | Manual | Audio peaks extracted and rendered |
| 16.1.10 | Clustering/playlist generation | Manual | Evolutionary clustering produces playlists |
| 16.1.11 | Cron scheduling | Manual | Scheduled task executes on time |
| 16.1.12 | Database cleaning | Manual | Orphaned albums removed |
| 16.1.13 | Audio analysis (ONNX) | Manual | Tracks analyzed, embeddings stored |
| 16.1.14 | Memory management | Auto | Memory utils function correctly |

### 16.2 Breaking Changes to Verify

| # | Test Case | Type | Details |
|---|-----------|------|---------|
| 16.2.1 | MPD removal | Auto | No references to `mediaserver_mpd` or `python-mpd2` |
| 16.2.2 | energy_normalized → energy fix | Auto | `chat_manager.py` uses `energy` column |
| 16.2.3 | Default MEDIASERVER_TYPE | Auto | .env.example defaults to `localfiles` |
| 16.2.4 | Setup redirect on upgrade | Manual | Existing installation sees setup wizard once |
| 16.2.5 | Auto-created tables | Auto | `init_db()` creates new tables without errors |
| 16.2.6 | Backward-compatible API | Auto | `create_playlist_from_ids()` works without `provider_ids` |

---

## 17. Security

### 17.1 XSS Prevention (Automated)

| # | Test Case | Type | Details |
|---|-----------|------|---------|
| 17.1.1 | escapeHtml in templates | Auto | All `innerHTML` assignments use `escapeHtml()` |
| 17.1.2 | utils.js loaded | Auto | `escapeHtml()` function available in all pages |
| 17.1.3 | Provider name escaping | Auto | `<script>` in provider name → escaped |
| 17.1.4 | Track title escaping | Auto | `<script>` in track title → escaped |

### 17.2 SQL Injection Prevention (Automated)

| # | Test Case | Type | Details |
|---|-----------|------|---------|
| 17.2.1 | Parameterized queries in MCP tools | Auto | All SQL uses `%s` placeholders |
| 17.2.2 | LIKE pattern escaping in brainstorm | Auto | `%` and `_` escaped in fuzzy search |
| 17.2.3 | Provider config input validation | Auto | Malicious JSONB → rejected or escaped |

### 17.3 Authentication & Authorization

| # | Test Case | Type | Details |
|---|-----------|------|---------|
| 17.3.1 | Provider credentials not logged | Manual | Check logs for token/password values |
| 17.3.2 | API keys not in responses | Auto | GET endpoints don't return API keys |
| 17.3.3 | Provider config API sanitized | Auto | `GET /api/setup/providers` redacts sensitive fields |

---

## Test Execution Priority

### P0 - Must Pass Before Merge

- [ ] All existing unit tests pass (`pytest tests/unit/ -v`)
- [ ] New MCP tool tests pass (Section 9)
- [ ] New AI MCP client tests pass (Section 8.6)
- [ ] New playlist ordering tests pass (Section 14)
- [ ] New instant playlist pipeline tests pass (Section 8)
- [ ] New setup wizard tests pass (Section 3)
- [ ] Database migration is idempotent (Section 11)
- [ ] Breaking changes verified (Section 16.2)
- [ ] XSS prevention verified (Section 17.1)

### P1 - Should Pass Before Release

- [ ] All API endpoints respond correctly (Section 6)
- [ ] Dark mode toggle and persistence (Section 12.1)
- [ ] LocalFiles provider tests (Section 10.2)
- [ ] Energy normalization tests (Section 9.6.8)
- [ ] Artist diversity enforcement (Section 8.4)
- [ ] Provider CRUD API (Section 3.3)
- [ ] Setup wizard UI flow (Section 4)

### P2 - Should Pass Before GA

- [ ] Full regression testing of all features (Section 16.1)
- [ ] Dark mode visual correctness on all pages (Section 12.2)
- [ ] All provider-specific tests (Section 10.1)
- [ ] Docker deployment tests (Section 15)
- [ ] AI provider integration tests (Section 8.7)
- [ ] Comparison suite benchmarks (testing_suite/)
- [ ] Security audit (Section 17)

---

## Appendix: Test Count Summary

| Category | Automated | Manual | Total |
|----------|-----------|--------|-------|
| Multi-Provider Architecture | 22 | 10 | 32 |
| GUI Setup Wizard | 0 | 22 | 22 |
| Environment / Config | 12 | 3 | 15 |
| API Endpoints | 26 | 4 | 30 |
| App Interactions | 0 | 14 | 14 |
| Instant Playlist & AI (pipeline) | 30 | 6 | 36 |
| AI MCP Client (test_ai_mcp_client.py) | 63 | 0 | 63 |
| MCP Tools | 40 | 0 | 40 |
| Provider-Specific | 18 | 14 per provider | 88 |
| Database & Schema | 12 | 0 | 12 |
| Dark Mode | 5 | 23 | 28 |
| Analysis Pipeline | 8 | 1 | 9 |
| Playlist Ordering | 16 | 0 | 16 |
| Deployment | 0 | 11 | 11 |
| Regression | 6 | 13 | 19 |
| Security | 7 | 1 | 8 |
| **TOTAL** | **265** | **~122** | **~387** |

**Existing automated tests:** ~200+
**New automated tests:** ~265
**Manual test cases:** ~122
