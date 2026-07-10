# Playlist Curator Isolated Test Deployment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resolve PR 417's three Bandit B608 findings with verified narrow suppressions, then deploy the current branch as a CPU-only test web instance using a clone of the stopped `audiomuse-test` database without touching the production worker.

**Architecture:** The security change adds runtime characterization tests for the allowlisted filter builder and documents exactly three Bandit suppressions at the trusted query-assembly boundary. Deployment builds one CPU image, clones the stopped test PostgreSQL volume read-only, creates a fresh Redis volume, and starts only PostgreSQL, Redis, and Flask in the dedicated `playlist-curator-test` Compose project.

**Tech Stack:** Python 3.11+, Flask, psycopg2, pytest, Bandit, Docker Desktop Linux containers, Docker Compose, PostgreSQL 15, Redis 7, PowerShell.

## Global Constraints

- Never stage, commit, overwrite, or discard `deployment/docker-compose-nvidia.yaml`.
- Never stop, restart, recreate, inspect secrets from, or attach to Compose project `audiomuse-ai-worker` except for redacted read-only verification.
- Never connect a test service to PostgreSQL or Redis at `192.168.1.172`.
- Never attach NVIDIA GPU device 0 to a test container.
- Never start a test RQ worker.
- Mount `audiomuse-test_test-postgres-data` read-only during cloning and never modify or remove it.
- Use a fresh Redis volume; do not copy `audiomuse-test_test-redis-data`.
- Publish only Flask port `18001`; do not publish PostgreSQL or Redis.
- Do not write media-server, database, authentication, or API secrets to the repository or a temporary environment file.
- Do not reply to or resolve GitHub review threads without explicit user authorization.

---

### Task 1: Prove and document the Bandit B608 false positives

**Files:**
- Create: `test/unit/test_playlist_curator_security.py`
- Modify: `app_playlist_curator.py:502-515`

**Interfaces:**
- Consumes: `_build_filter_query(filters: list[dict], match_mode: str) -> tuple[str, list]` from `app_playlist_curator.py`.
- Produces: three documented `# nosec B608` query templates whose values remain in psycopg2 parameters.

- [ ] **Step 1: Add the security characterization and suppression-count tests**

Create `test/unit/test_playlist_curator_security.py` with:

```python
# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Security regression tests for Playlist Curator SQL filter assembly.

Main Features:
* Keep hostile request values in psycopg2 parameters rather than SQL text.
* Reject non-allowlisted fields and operators from structural query fragments.
* Require narrowly scoped Bandit B608 suppressions at trusted assembly sites.
"""

from pathlib import Path

from app_playlist_curator import _build_filter_query


REPO_ROOT = Path(__file__).resolve().parents[2]
CURATOR_SOURCE = REPO_ROOT / "app_playlist_curator.py"


def test_hostile_filter_value_stays_in_bound_parameters():
    hostile = "x%' OR 1=1; DROP TABLE score; --"

    clause, params = _build_filter_query(
        [{"field": "artist", "operator": "contains", "value": hostile}],
        "all",
    )

    assert clause == "(author ILIKE %s)"
    assert hostile not in clause
    assert params == [f"%{hostile}%"]


def test_untrusted_structural_inputs_cannot_enter_filter_clause():
    hostile = "title); DROP TABLE score; --"

    clause, params = _build_filter_query(
        [{"field": hostile, "operator": hostile, "value": "ignored"}],
        hostile,
    )

    assert clause == "1=1"
    assert params == []
    assert hostile not in clause


def test_match_mode_is_reduced_to_fixed_and_or_tokens():
    hostile_mode = "all); DROP TABLE score; --"

    clause, params = _build_filter_query(
        [
            {"field": "artist", "operator": "is", "value": "A"},
            {"field": "album", "operator": "is", "value": "B"},
        ],
        hostile_mode,
    )

    assert clause == "(author = %s OR album = %s)"
    assert hostile_mode not in clause
    assert params == ["A", "B"]


def test_bandit_suppressions_are_limited_to_three_reviewed_queries():
    source = CURATOR_SOURCE.read_text(encoding="utf-8")

    assert source.count("# nosec B608") == 3
    assert "only allowlisted SQL identifiers and operators" in source
```

- [ ] **Step 2: Run the new tests and verify the suppression test is red**

Run:

```powershell
python -m pytest test/unit/test_playlist_curator_security.py -q
```

Expected: three characterization tests pass and `test_bandit_suppressions_are_limited_to_three_reviewed_queries` fails because the source currently has zero `# nosec B608` annotations.

- [ ] **Step 3: Add three scoped query variables and suppressions**

Replace the three flagged executions in `search_api` with:

```python
            # _build_filter_query emits only allowlisted SQL identifiers and operators;
            # every request value remains in the psycopg2 params tuple below.
            if search_only:
                offset = (page - 1) * per_page
                count_query = f"SELECT COUNT(*) AS total FROM score WHERE {where_clause}"  # nosec B608
                cur.execute(count_query, tuple(params))
                count_row = cur.fetchone()
                filter_total = int(count_row['total'] if count_row else 0)
                page_query = (
                    f"SELECT item_id FROM score WHERE {where_clause} "  # nosec B608
                    "ORDER BY item_id LIMIT %s OFFSET %s"
                )
                cur.execute(page_query, tuple(params + [per_page, offset]))
            else:
                filter_query = f"SELECT item_id FROM score WHERE {where_clause}"  # nosec B608
                cur.execute(filter_query, tuple(params))
```

- [ ] **Step 4: Run the focused tests and linters**

Run:

```powershell
python -m pytest test/unit/test_playlist_curator_security.py test/unit/test_playlist_curator_merge_compat.py -q
python -m ruff check app_playlist_curator.py test/unit/test_playlist_curator_security.py
if (-not (Test-Path 'C:\tmp\playlist-curator-bandit-venv\Scripts\python.exe')) {
  python -m venv C:\tmp\playlist-curator-bandit-venv
  & 'C:\tmp\playlist-curator-bandit-venv\Scripts\python.exe' -m pip install "bandit[sarif]"
}
& 'C:\tmp\playlist-curator-bandit-venv\Scripts\python.exe' -m bandit -r app_playlist_curator.py --severity-level medium --confidence-level medium
```

Expected: all pytest tests pass, Ruff reports `All checks passed!`, and Bandit reports no B608 findings for the three reviewed query templates.

- [ ] **Step 5: Run the broader policy and curator checks**

Run:

```powershell
python -m pytest test/unit/test_file_header_convention.py test/unit/test_no_em_dash_in_source.py test/unit/test_playlist_curator_templates.py test/unit/test_sql_injection_params.py -q
git diff --check
```

Expected: all tests pass and `git diff --check` emits no errors.

- [ ] **Step 6: Commit only the security fix**

Run:

```powershell
git add -- app_playlist_curator.py test/unit/test_playlist_curator_security.py
git diff --cached --check
git diff --cached --name-only
git commit -m "fix: document safe curator SQL assembly"
```

Expected staged names: only `app_playlist_curator.py` and `test/unit/test_playlist_curator_security.py`. The user's `deployment/docker-compose-nvidia.yaml` remains unstaged.

- [ ] **Step 7: Push and wait for PR 417 checks**

Run:

```powershell
git push origin feature/playlist-curator-port
gh pr checks 417 --watch --interval 10
```

Expected: Ruff, Unit Tests, Security Scan, and all required checks complete without failures. Re-read unresolved review threads afterward; report remaining state without replying or resolving.

---

### Task 2: Build the CPU image and prepare isolated test data

**Files:**
- Create temporarily: `C:\tmp\playlist-curator-test.compose.yaml`
- Read only: Docker container `audiomuse-ai-worker-instance`
- Read only: Docker volume `audiomuse-test_test-postgres-data`
- Create: Docker volume `playlist-curator-test_postgres-data`

**Interfaces:**
- Consumes: current Git HEAD after Task 1 and stopped `audiomuse-test` container configuration.
- Produces: image `audiomuse-ai-playlist-curator:test`, cloned PostgreSQL volume, and a secret-free Compose definition.

- [ ] **Step 1: Capture the redacted production-worker baseline**

Run a read-only `docker inspect` and retain this JSON output for final comparison:

```powershell
$container = (docker inspect 'audiomuse-ai-worker-instance' | ConvertFrom-Json)[0]
$envMap = @{}
foreach ($entry in $container.Config.Env) {
  $parts = $entry -split '=', 2
  $envMap[$parts[0]] = if ($parts.Count -gt 1) { $parts[1] } else { '' }
}
$redis = [Uri]$envMap['REDIS_URL']
[pscustomobject]@{
  Id = $container.Id
  StartedAt = $container.State.StartedAt
  Status = $container.State.Status
  Network = @($container.NetworkSettings.Networks.PSObject.Properties.Name)
  PostgresHost = $envMap['POSTGRES_HOST']
  RedisHost = $redis.Host
  GpuDeviceIds = @($container.HostConfig.DeviceRequests | ForEach-Object { $_.DeviceIDs })
} | ConvertTo-Json -Depth 5
```

Expected: status `running`, network `audiomuse-ai-worker_default`, PostgreSQL and Redis host `192.168.1.172`, and GPU device `0`.

- [ ] **Step 2: Run collision preflight checks**

Run:

```powershell
if (Get-NetTCPConnection -LocalPort 18001 -State Listen -ErrorAction SilentlyContinue) {
  throw 'Port 18001 is already in use.'
}
$plannedNames = @('playlist-curator-test-flask', 'playlist-curator-test-postgres', 'playlist-curator-test-redis')
$existingNames = @(docker ps -a --format '{{.Names}}')
$collisions = @($plannedNames | Where-Object { $_ -in $existingNames })
if ($collisions) { throw "Container name collision: $($collisions -join ', ')" }
docker volume inspect playlist-curator-test_postgres-data *> $null
if ($LASTEXITCODE -eq 0) { throw 'Destination PostgreSQL volume already exists.' }
```

Expected: no output and no exception.

- [ ] **Step 3: Build the current branch with the CPU base**

Run:

```powershell
docker build --build-arg BASE_IMAGE=ubuntu:24.04 -t audiomuse-ai-playlist-curator:test .
```

Expected: image build succeeds. Verify no GPU declaration is present:

```powershell
docker image inspect audiomuse-ai-playlist-curator:test --format '{{json .Config.Env}}'
```

- [ ] **Step 4: Clone the stopped test PostgreSQL volume read-only**

Run:

```powershell
docker volume create playlist-curator-test_postgres-data
docker run --rm --network none `
  -v audiomuse-test_test-postgres-data:/source:ro `
  -v playlist-curator-test_postgres-data:/destination `
  alpine:3.20 sh -c 'cp -a /source/. /destination/'
```

Expected: the helper exits 0, the source is mounted read-only, and no production resource is referenced.

- [ ] **Step 5: Create the secret-free temporary Compose file**

Create `C:\tmp\playlist-curator-test.compose.yaml` with:

```yaml
name: playlist-curator-test

services:
  redis:
    image: redis:7-alpine
    container_name: playlist-curator-test-redis
    restart: unless-stopped
    networks: [test]
    volumes:
      - redis-data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s
      timeout: 3s
      retries: 20

  postgres:
    image: postgres:15-alpine
    container_name: playlist-curator-test-postgres
    restart: unless-stopped
    networks: [test]
    environment:
      POSTGRES_USER: "${POSTGRES_USER}"
      POSTGRES_PASSWORD: "${POSTGRES_PASSWORD}"
      POSTGRES_DB: "${POSTGRES_DB}"
    volumes:
      - postgres-data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U $$POSTGRES_USER -d $$POSTGRES_DB"]
      interval: 5s
      timeout: 3s
      retries: 30

  audiomuse-ai-flask:
    image: audiomuse-ai-playlist-curator:test
    container_name: playlist-curator-test-flask
    restart: unless-stopped
    networks: [test]
    depends_on:
      redis:
        condition: service_healthy
      postgres:
        condition: service_healthy
    ports:
      - "18001:8000"
    environment:
      SERVICE_TYPE: flask
      TZ: "${TZ:-UTC}"
      POSTGRES_USER: "${POSTGRES_USER}"
      POSTGRES_PASSWORD: "${POSTGRES_PASSWORD}"
      POSTGRES_DB: "${POSTGRES_DB}"
      POSTGRES_HOST: postgres
      POSTGRES_PORT: "5432"
      REDIS_URL: redis://redis:6379/0
      MEDIASERVER_TYPE: "${MEDIASERVER_TYPE}"
      JELLYFIN_URL: "${JELLYFIN_URL}"
      JELLYFIN_USER_ID: "${JELLYFIN_USER_ID}"
      JELLYFIN_TOKEN: "${JELLYFIN_TOKEN}"
      AUTH_ENABLED: "${AUTH_ENABLED:-true}"
      AUDIOMUSE_USER: "${AUDIOMUSE_USER}"
      AUDIOMUSE_PASSWORD: "${AUDIOMUSE_PASSWORD}"
      API_TOKEN: "${API_TOKEN}"
      JWT_SECRET: "${JWT_SECRET}"
      CLAP_ENABLED: "false"
      NVIDIA_VISIBLE_DEVICES: none
      USE_GPU_CLUSTERING: "false"
      TEMP_DIR: /app/temp_audio
    volumes:
      - temp-audio-flask:/app/temp_audio
      - plugins-flask:/app/plugin/installed
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health', timeout=5)"]
      interval: 10s
      timeout: 6s
      retries: 30

networks:
  test:
    name: playlist-curator-test_default

volumes:
  postgres-data:
    external: true
    name: playlist-curator-test_postgres-data
  redis-data:
    name: playlist-curator-test_redis-data
  temp-audio-flask:
    name: playlist-curator-test_temp-audio-flask
  plugins-flask:
    name: playlist-curator-test_plugins-flask
```

Expected: the file contains references only, not interpolated secret values.

---

### Task 3: Start and verify the isolated test instance

**Files:**
- Read temporarily: `C:\tmp\playlist-curator-test.compose.yaml`
- Read only: stopped container `audiomuse-test-flask-app` for source configuration.

**Interfaces:**
- Consumes: image, volume, and Compose file from Task 2.
- Produces: healthy Flask endpoint at `http://localhost:18001` with no production-worker overlap.

- [ ] **Step 1: Load old test configuration into process memory and start the stack**

Run as one PowerShell command so secret values exist only in process memory:

```powershell
$source = (docker inspect 'audiomuse-test-flask-app' | ConvertFrom-Json)[0]
$sourceEnv = @{}
foreach ($entry in $source.Config.Env) {
  $parts = $entry -split '=', 2
  $sourceEnv[$parts[0]] = if ($parts.Count -gt 1) { $parts[1] } else { '' }
}
$keys = @(
  'POSTGRES_USER', 'POSTGRES_PASSWORD', 'POSTGRES_DB', 'TZ',
  'MEDIASERVER_TYPE', 'JELLYFIN_URL', 'JELLYFIN_USER_ID', 'JELLYFIN_TOKEN',
  'AUTH_ENABLED', 'AUDIOMUSE_USER', 'AUDIOMUSE_PASSWORD', 'API_TOKEN', 'JWT_SECRET'
)
foreach ($key in $keys) {
  if ($sourceEnv.ContainsKey($key)) { Set-Item -Path "Env:$key" -Value $sourceEnv[$key] }
}
docker compose -p playlist-curator-test -f C:\tmp\playlist-curator-test.compose.yaml up -d postgres redis audiomuse-ai-flask
```

Expected: exactly three containers are created under project `playlist-curator-test`; no worker service exists.

- [ ] **Step 2: Wait for health and inspect service status**

Run:

```powershell
docker compose -p playlist-curator-test -f C:\tmp\playlist-curator-test.compose.yaml ps
$health = Invoke-RestMethod -Uri 'http://localhost:18001/api/health' -TimeoutSec 10
if ($health.status -ne 'ok') { throw "Unexpected health response: $($health | ConvertTo-Json -Compress)" }
```

Expected: PostgreSQL, Redis, and Flask are running and healthy; health response is `{"status":"ok"}`.

- [ ] **Step 3: Verify cloned data and Playlist Curator route availability**

Run:

```powershell
docker exec playlist-curator-test-postgres sh -c 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -tAc "SELECT COUNT(*) FROM score;"'
$status = curl.exe -sS -o NUL -w "%{http_code}" 'http://localhost:18001/playlist_curator/search'
if ([int]$status -notin @(200, 302)) { throw "Unexpected curator route status: $status" }
```

Expected: `score` count is greater than zero; the curator route returns 200 or an expected authentication/setup redirect 302.

- [ ] **Step 4: Prove network, endpoint, and GPU isolation**

Run:

```powershell
$names = @('playlist-curator-test-flask', 'playlist-curator-test-postgres', 'playlist-curator-test-redis')
$results = foreach ($name in $names) {
  $container = (docker inspect $name | ConvertFrom-Json)[0]
  $envKeys = @{}
  foreach ($entry in $container.Config.Env) {
    $parts = $entry -split '=', 2
    $envKeys[$parts[0]] = if ($parts.Count -gt 1) { $parts[1] } else { '' }
  }
  [pscustomobject]@{
    Name = $name
    Project = $container.Config.Labels.'com.docker.compose.project'
    Networks = @($container.NetworkSettings.Networks.PSObject.Properties.Name)
    PostgresHost = $envKeys['POSTGRES_HOST']
    RedisUrl = $envKeys['REDIS_URL']
    DeviceRequests = @($container.HostConfig.DeviceRequests).Count
  }
}
$results | ConvertTo-Json -Depth 5
```

Expected: project is `playlist-curator-test`, only network is `playlist-curator-test_default`, Flask uses `postgres` and `redis://redis:6379/0`, and every `DeviceRequests` count is 0.

- [ ] **Step 5: Reinspect and compare the production worker**

Repeat Task 2 Step 1's redacted inspection.

Expected: container ID, start time, status, network, PostgreSQL host, Redis host, and GPU device IDs exactly match the baseline captured before deployment.

- [ ] **Step 6: Final repository and PR verification**

Run:

```powershell
git status --short --branch
gh pr checks 417 --json name,state,bucket,link,workflow
```

Expected: only the user's `deployment/docker-compose-nvidia.yaml` remains modified; PR checks contain no `fail` bucket. Report the test URL, container names, image, database row count, and production-isolation evidence.

## Rollback

If the new test project fails after creation, stop only its containers:

```powershell
docker compose -p playlist-curator-test -f C:\tmp\playlist-curator-test.compose.yaml down
```

Do not add `--volumes`; preserve the cloned PostgreSQL volume for investigation. Never run `docker compose down` without both `-p playlist-curator-test` and the explicit temporary Compose file.
