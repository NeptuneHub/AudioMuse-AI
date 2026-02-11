# AudioMuse-AI v0.9.0 - Test Results

**Date:** 2026-02-11
**Branch:** `multi-provider-setup-gui`
**Tester:** Claude + User
**Environment:** Docker (NVIDIA GPU test stack) + Windows 11 local
**Docker Compose:** `deployment/docker-compose-unified-nvidia-test.yaml`
**Local Python:** 3.14.2 (Windows 11 Pro)
**Container Python:** 3.12 (Ubuntu 24.04 + CUDA 12.8.1)

---

## Pre-flight: Collection Status

### Local Environment (Python 3.14.2, Windows)
- **622 tests collected** across all test files
- **9 files blocked** (import `librosa` via `tasks/__init__.py` — not installed locally)
  - `test_analysis.py`, `test_artist_gmm_manager.py`, `test_clustering_helper.py`,
    `test_clustering_postprocessing.py`, `test_commons.py`, `test_memory_utils.py`,
    `test_path_manager.py`, `test_song_alchemy.py`, `test_sonic_fingerprint_manager.py`
- **213 runtime failures** — also `librosa` import (test_clustering, test_mediaserver, test_voyager_manager, etc.)
- **407 passed, 2 skipped, 0 real failures** on tests that don't need `librosa`
- Installed `audioop-lts` to fix Python 3.14 `pydub` compatibility

### Docker Environment (Python 3.12.3, Ubuntu 24.04 + CUDA 12.8.1)
- **833 tests collected**, 0 collection errors
- **832 passed, 1 failed, 4 warnings**
- Build time: ~15 min (GPU packages + model downloads)
- Containers: redis (healthy), postgres (healthy), flask-app (up), worker (up)

### Bug Found & Fixed During Testing
| Test | Error | Fix |
|------|-------|-----|
| `test_mediaserver_localfiles.py::TestPathNormalization::test_posix_conversion` | Backslashes not converted on Linux | `normalize_file_path()`: replace `\` before creating `PurePosixPath` |

### After Fix: 833 passed, 0 failed, 4 warnings (63.80s)

### Warnings (non-blocking)
- `analysis.py:387` — RuntimeWarning: invalid value in log10 (known, edge case in mel spectrogram)
- `sklearn.linear_model` — FutureWarning: `penalty`/`n_jobs` deprecated in 1.8 (upgrade notice)

---

## Section 3: Multi-Provider Architecture

### 3.1 Fresh Install (No Existing Data)
_(pending)_

### 3.2 Migration (Existing Single-Provider Data)
_(pending)_

### 3.3 Provider CRUD API
_(pending)_

### 3.4 Multi-Provider Playlist Creation
_(pending)_

---

## Section 4: GUI Setup Wizard
_(pending - manual)_

---

## Section 5: Environment / Config Setup

### 5.1 Config Variable Validation
_(pending)_

### 5.2 Environment File
_(pending)_

### 5.3 Settings Persistence
_(pending)_

---

## Section 6: API Endpoints

### 6.1 Without Provider
_(pending)_

### 6.2 With Provider
_(pending)_

### 6.3 Error Handling
_(pending)_

---

## Section 7: App Interactions (UI/UX)
_(pending - manual)_

---

## Section 8: Instant Playlist & AI Changes

### 8.1 Agentic Loop
_(pending)_

### 8.2 Pre-Execution Validation
_(pending)_

### 8.3 Proportional Sampling
_(pending)_

### 8.4 Artist Diversity
_(pending)_

### 8.5 System Prompt
_(pending)_

### 8.6 AI MCP Client
_(pending)_

### 8.7 AI Provider Integration
_(pending)_

### 8.8 Library Context
_(pending)_

---

## Section 9: MCP Tools

### 9.1 song_similarity
_(pending)_

### 9.2 text_search
_(pending)_

### 9.3 artist_similarity
_(pending)_

### 9.4 song_alchemy
_(pending)_

### 9.5 ai_brainstorm
_(pending)_

### 9.6 search_database
_(pending)_

---

## Section 10: Provider-Specific Testing

### 10.1 Common Provider Interface
_(pending - manual)_

### 10.2 LocalFiles Provider
_(pending)_

---

## Section 11: Database & Schema Changes

### 11.1 Schema Migration
_(pending)_

### 11.2 Data Integrity
_(pending)_

---

## Section 12: Dark Mode

### 12.1 Toggle & Persistence
_(pending)_

### 12.2 Visual Correctness
_(pending - manual)_

### 12.3 CSS Variables
_(pending)_

---

## Section 13: Analysis Pipeline

### 13.1 Analysis with New Fields
_(pending)_

### 13.2 Voyager Index
_(pending)_

---

## Section 14: Playlist Ordering

### 14.1 Greedy Nearest-Neighbor
_(pending)_

### 14.2 Composite Distance
_(pending)_

### 14.3 Circle of Fifths Key Distance
_(pending)_

### 14.4 Energy Arc
_(pending)_

---

## Section 15: Deployment & Docker
_(pending - manual)_

---

## Section 16: Regression Tests

### 16.1 Features That Must Still Work
_(pending)_

### 16.2 Breaking Changes to Verify
_(pending)_

---

## Section 17: Security

### 17.1 XSS Prevention
_(pending)_

### 17.2 SQL Injection Prevention
_(pending)_

### 17.3 Authentication & Authorization
_(pending)_

---

## Summary

| Section | Pass | Fail | Skip | Total |
|---------|------|------|------|-------|
| 3. Multi-Provider | | | | |
| 4. Setup Wizard | | | | |
| 5. Config | | | | |
| 6. API | | | | |
| 7. UI/UX | | | | |
| 8. Instant Playlist | | | | |
| 9. MCP Tools | | | | |
| 10. Providers | | | | |
| 11. Database | | | | |
| 12. Dark Mode | | | | |
| 13. Analysis | | | | |
| 14. Playlist Ordering | | | | |
| 15. Docker | | | | |
| 16. Regression | | | | |
| 17. Security | | | | |
| **TOTAL** | | | | |
