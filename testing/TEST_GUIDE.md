# AudioMuse-AI — album_artist Branch Test Guide

Testing the `album_artist` column addition across all providers (Jellyfin, Emby, Navidrome, Lyrion).

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────┐
│                    Host Machine (NVIDIA GPU)                  │
│                                                              │
│  ┌─── docker-compose-test-providers.yaml ──────────────────┐ │
│  │  Jellyfin :8096   Emby :8097                            │ │
│  │  Navidrome :4533  Lyrion :9010                          │ │
│  │         ▲ all mount TEST_MUSIC_PATH read-only           │ │
│  └─────────┼───────────────────────────────────────────────┘ │
│            │  shared network: audiomuse-test-net              │
│  ┌─────────┼── docker-compose-test-audiomuse.yaml ─────────┐ │
│  │         ▼                                                │ │
│  │  AM-Jellyfin :8001  (redis + postgres:5433 + flask+wkr) │ │
│  │  AM-Emby     :8002  (redis + postgres:5434 + flask+wkr) │ │
│  │  AM-Navidrome:8003  (redis + postgres:5435 + flask+wkr) │ │
│  │  AM-Lyrion   :8004  (redis + postgres:5436 + flask+wkr) │ │
│  └──────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
```

| Instance          | Web UI         | Postgres Port | Provider Port |
| ----------------- | -------------- | ------------- | ------------- |
| **Jellyfin AM**   | localhost:8001 | 5433          | 8096          |
| **Emby AM**       | localhost:8002 | 5434          | 8097          |
| **Navidrome AM**  | localhost:8003 | 5435          | 4533          |
| **Lyrion AM**     | localhost:8004 | 5436          | 9010          |

---

## Prerequisites

- Docker & Docker Compose v2+
- NVIDIA Container Toolkit (`nvidia-ctk`)
- `psql` client (for the validation script)
- A directory of test music files (FLAC/MP3/etc.) with proper ID3/Vorbis tags including **Album Artist**

---

## Step 0 — Prepare the environment

```bash
cd AudioMuse-AI/testing/
cp .env.test.example .env.test
```

Edit `.env.test` and set **`TEST_MUSIC_PATH`** to your music directory:

```
TEST_MUSIC_PATH=/mnt/data/test_music
```

Leave the provider credential fields blank for now — you will fill them in during setup.

---

## Step 1 — Start the providers

```bash
docker compose -f docker-compose-test-providers.yaml --env-file .env.test up -d
```

Verify all four are healthy:

```bash
docker compose -f docker-compose-test-providers.yaml --env-file .env.test ps
```

---

## Step 2 — Configure each provider

### 2A. Jellyfin (http://localhost:8096)

1. Open `http://localhost:8096` in a browser.
2. Complete the first-run wizard:
   - Create an admin user (e.g. `admin` / `admin`).
   - Add a media library: type **Music**, folder `/media/music`.
   - Finish the wizard and let the initial scan complete.
3. **Get your User ID:**
   - Go to **Dashboard → Users → click your user**.
   - The URL will contain the user ID:
     `http://localhost:8096/web/#/dashboard/users/profile?userId=<USER_ID>`
   - Copy the `<USER_ID>`.
4. **Get your API token:**
   - Go to **Dashboard → API Keys → +** (add new key).
   - Name it `audiomuse-test` and click OK.
   - Copy the generated token.
5. Update `.env.test`:
   ```
   JELLYFIN_USER_ID=<USER_ID>
   JELLYFIN_TOKEN=<API_TOKEN>
   ```

### 2B. Emby (http://localhost:8097)

1. Open `http://localhost:8097`.
2. Complete the first-run wizard:
   - Create an admin user.
   - Add a media library: type **Music**, folder `/media/music`.
   - Finish and wait for the scan.
3. **Get your User ID:**
   - Go to **Settings → Users → click your user**.
   - The URL contains the user ID:
     `http://localhost:8097/web/index.html?#!/users/user?userId=<USER_ID>`
4. **Get your API token:**
   - Go to **Settings → Advanced → API Keys → New API Key**.
   - Name it `audiomuse-test`, copy the key.
5. Update `.env.test`:
   ```
   EMBY_USER_ID=<USER_ID>
   EMBY_TOKEN=<API_TOKEN>
   ```

### 2C. Navidrome (http://localhost:4533)

1. Open `http://localhost:4533`.
2. Create the initial admin account (first visit auto-prompts).
   - Username: e.g. `admin`
   - Password: e.g. `admin`
3. Navidrome auto-scans `/music` on startup. Verify in the UI that tracks appear.
4. **Navidrome uses username/password auth** (Subsonic API), not tokens.
5. Update `.env.test`:
   ```
   NAVIDROME_USER=admin
   NAVIDROME_PASSWORD=admin
   ```

### 2D. Lyrion Music Server (http://localhost:9010)

1. Open `http://localhost:9010`.
2. On first run, it may prompt for a music folder — confirm `/music`.
3. Go to **Settings → Basic Settings → Media Folders** and verify `/music` is listed.
4. Trigger a rescan: **Settings → Basic Settings → Rescan**.
5. **Lyrion requires no API key.** The AudioMuse compose already points to `http://test-lyrion:9010`.
6. No changes needed in `.env.test` for Lyrion.

---

## Step 3 — Build and start the AudioMuse instances

The compose file builds the NVIDIA image **locally** from the repo's `Dockerfile`
(using `nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04` as the base).  The image is
built once by the `flask-jellyfin` service and reused by all other services via the
shared tag `audiomuse-ai:test-nvidia`.

After filling in all credentials in `.env.test`:

```bash
# Build the image and start everything (first run will take a while)
docker compose -f docker-compose-test-audiomuse.yaml --env-file .env.test up -d --build
```

On subsequent runs (code changes), rebuild with:

```bash
docker compose -f docker-compose-test-audiomuse.yaml --env-file .env.test build
docker compose -f docker-compose-test-audiomuse.yaml --env-file .env.test up -d
```

Verify all containers are running:

```bash
docker compose -f docker-compose-test-audiomuse.yaml --env-file .env.test ps
```

You should see 16 containers (4 × {redis, postgres, flask, worker}).

Check GPU allocation:

```bash
docker exec test-am-flask-jellyfin nvidia-smi
```

---

## Step 4 — Run analysis on each instance

For **each** AudioMuse instance, trigger a full library analysis:

| Instance   | URL                                    |
| ---------- | -------------------------------------- |
| Jellyfin   | http://localhost:8001                  |
| Emby       | http://localhost:8002                  |
| Navidrome  | http://localhost:8003                  |
| Lyrion     | http://localhost:8004                  |

1. Open the web UI for the instance.
2. Navigate to the **Analysis** page.
3. Click **Start Analysis** and wait for it to complete.
4. Monitor progress in the UI or via logs:
   ```bash
   docker logs -f test-am-worker-jellyfin
   docker logs -f test-am-worker-emby
   docker logs -f test-am-worker-navidrome
   docker logs -f test-am-worker-lyrion
   ```

---

## Step 5 — Validate album_artist

Run the validation script from the host:

```bash
cd AudioMuse-AI/testing/
./validate_album_artist.sh
```

The script connects to each Postgres instance and checks:
- Column existence
- Population rate
- Sample data
- `Unknown` fallback ratio
- album_artist vs author (track artist) mismatch count

A passing result means `album_artist` is being stored correctly for that provider.

---

## Step 6 — Manual spot-check via psql

You can also query any instance directly:

```bash
# Jellyfin instance
PGPASSWORD=audiomusepassword psql -h localhost -p 5433 -U audiomuse -d audiomusedb

# Then run:
SELECT title, author, album, album_artist FROM score LIMIT 20;
```

Repeat with ports 5434 (Emby), 5435 (Navidrome), 5436 (Lyrion).

---

## Test Checklist

### Infrastructure
- [ ] All 4 providers start and show their web UI
- [ ] All providers scanned the test music library
- [ ] All 16 AudioMuse containers start without errors
- [ ] GPU is accessible from flask and worker containers

### Provider Setup
- [ ] Jellyfin: library created, API key + user ID obtained
- [ ] Emby: library created, API key + user ID obtained
- [ ] Navidrome: admin created, library scanned
- [ ] Lyrion: music folder configured, rescan complete

### Analysis
- [ ] Jellyfin AM (`:8001`): analysis completes successfully
- [ ] Emby AM (`:8002`): analysis completes successfully
- [ ] Navidrome AM (`:8003`): analysis completes successfully
- [ ] Lyrion AM (`:8004`): analysis completes successfully

### album_artist Validation (run `validate_album_artist.sh`)
- [ ] Jellyfin: `album_artist` column exists
- [ ] Jellyfin: `album_artist` populated for >0 tracks
- [ ] Jellyfin: sample data looks correct (not all `Unknown`)
- [ ] Emby: `album_artist` column exists
- [ ] Emby: `album_artist` populated for >0 tracks
- [ ] Emby: sample data looks correct
- [ ] Navidrome: `album_artist` column exists
- [ ] Navidrome: `album_artist` populated for >0 tracks
- [ ] Navidrome: sample data looks correct
- [ ] Lyrion: `album_artist` column exists
- [ ] Lyrion: `album_artist` populated for >0 tracks
- [ ] Lyrion: sample data looks correct

### Functional Smoke Tests (per instance)
- [ ] Similarity search returns results with no errors
- [ ] Song Alchemy returns results (album_artist used internally)
- [ ] CLAP text search works (if CLAP_ENABLED=true)
- [ ] Path finder returns a valid path
- [ ] Clustering completes without errors

### Edge Cases
- [ ] Compilation albums: `album_artist` ≠ `author` (track artist)
- [ ] Single-artist albums: `album_artist` = `author`
- [ ] Tracks missing album_artist metadata in source: stored as `NULL` (not crash)

---

## Teardown

```bash
# Stop AudioMuse instances
docker compose -f docker-compose-test-audiomuse.yaml --env-file .env.test down -v

# Stop providers
docker compose -f docker-compose-test-providers.yaml --env-file .env.test down -v
```

The `-v` flag removes named volumes (database data, configs). Omit it to preserve state between runs.

---

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `Cannot connect to Postgres on port X` | Check `docker compose ps` — is the postgres container up? |
| `album_artist column NOT found` | The Flask app creates the column on startup. Check flask container logs. |
| `0 tracks in score table` | Analysis hasn't run yet. Trigger it from the UI. |
| `100% Unknown album_artist` | Provider isn't returning the field. Check worker logs for the `OriginalAlbumArtist` value. |
| Flask can't reach provider | Both stacks must share `audiomuse-test-net`. Run `docker network inspect audiomuse-test-net`. |
| GPU not available | Run `nvidia-smi` on host. Check `nvidia-container-toolkit` is installed. |
| Emby won't start on :8097 | Emby internally uses 8096; the host mapping is 8097→8096. Ensure no port conflict. |
