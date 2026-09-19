# End-to-end functional tests

`test/e2e/` boots the real AudioMuse-AI stack and drives it over HTTP, one
module per user-facing functionality:

- Postgres (CI's service container, or a local `pgserver` cluster),
- two real Navidrome instances serving the committed fixture library under
  `library/` (the default server, and a second one for the multi-server and
  provider-migration scenarios),
- `app.py` under gunicorn on port 8000, exactly as in the container,
- the two real queue workers (`python -u -m taskqueue.worker --queue high|default`)
  and the real control listener (`python -u -m taskqueue.control`),
- the supervisor side of the app's control plane: the harness answers on the
  `AUDIOMUSE_CONTROL_SOCKET` the native builds use, so a worker restart the app
  publishes (default-server swap, provider migration) really stops and starts
  the worker processes and is acknowledged in the database,
- a real `fpcalc` for Chromaprint,
- the real lyrics ASR (silero voice detection and Whisper-small, both ONNX) on
  five real CC0 sung tracks (album H of the library); one of them also carries
  its real public-domain text as an `.lrc` sidecar for the provider path,
- a seed catalogue of a few hundred real songs (see below) so clustering,
  search and the indexes run at a realistic library size.

## The seed catalogue

`seed/catalogue.json.gz` holds the analysis rows (score features, MusiCNN
embedding, CLAP embedding, neural fingerprint) the real pipeline produced once
for 289 songs released under CC0 1.0, taken from the Wikimedia Commons list
published with the AudioMuse-AI-DCLAP model (300 were fetched; 8 turned out
to be the same recording as another and were left out, and 3 with real
vocals live on as real clips in album H instead). The seed is
anonymized: every song is "Various Artists - song N" in the placeholder tags
and in the rows alike, no per-song source is kept (the file records the list
and the licence once), and the MusiCNN and CLAP vectors carry a small seeded
Gaussian noise with their norm preserved, so the repository holds a
realistic catalogue without anyone's actual song data. Neither the audio
nor any per-song file is committed: Navidrome only gives an id to a file it
finds on disk, so at boot the harness writes a one-second silent MP3 per
seeded song into `library/`, tagged from the catalogue (those `SCnn` folders
are git-ignored), and the `seeded_catalogue` fixture inserts the rows bound
to the ids Navidrome gave them before the shared analysis. The
app then treats the songs as already analyzed (the analysis only processes
the real clips), and every index build, clustering run, search and migration
works on about 300 catalogue rows instead of a dozen. `seed_builder.py fetch`
and `seed_builder.py build` reproduce the file from the licence list (needs
the stack, about 35 minutes of analysis; the build keeps its state in
`$HOME/audiomuse_e2e_seed_state` so a rerun only exports).

Nothing of the app or of the provider is faked. Every test asserts on HTTP
answers, on the rows the app persisted, and on what it wrote to Navidrome. CI
runs the suite on every pull request to `main` (`.github/workflows/e2e.yml`).

## Running it locally (Linux or WSL)

```
bash test/e2e/run_local.sh --no-browser            # everything but the page smoke
bash test/e2e/run_local.sh --no-browser -k cold    # one module
bash test/e2e/run_local.sh                         # with the Playwright page smoke
```

The script activates the repo `.venv`, checks the packages from
`test/requirements.txt`, and on the first run builds the `unaccent` and
`pg_trgm` PostgreSQL extensions against `pgserver`'s bundled server (the same
script the Linux native build uses, needs `gcc`, `make` and `curl`), because
`init_db` creates them and `pgserver` ships without contrib modules. Navidrome
and `fpcalc` are downloaded once into `test/.cache/` from their pinned release
assets and verified by sha256. Models are taken from `model/` (or
`AUDIOMUSE_E2E_MODEL_DIR`, or `test/models` as CI fills it).

Alternatives to `pgserver`: point `AUDIOMUSE_TEST_DATABASE_URL` at any
disposable PostgreSQL 15+ with the two extensions available, for example
`docker run -d -p 5432:5432 -e POSTGRES_PASSWORD=postgres postgres:15-alpine`.
The database named by that variable is wiped (`DROP SCHEMA public CASCADE`) at
the start of every run. An unreachable value is a failure, never a skip.

Run artifacts land in `test/e2e/.run/`: one log per process under `logs/`,
`env.json` (secrets masked) and, for the page smoke, screenshots and traces of
failed pages under `playwright/`. CI uploads that directory.

### Iterating on one module

`analyzed_library` runs the full analysis once per session (several minutes on
CPU). To keep the analyzed catalogue between runs set a state directory on a
native filesystem:

```
export AUDIOMUSE_E2E_STATE_DIR=$HOME/audiomuse_e2e_state
bash test/e2e/run_local.sh --no-browser -k "clustering or alchemy"
```

With the state kept, the analysis re-run is a no-op and the modules start
within about a minute. `test_00_cold_library.py` asserts an empty catalogue, so
it fails on a reused state by design; delete the directory to start over.

### Keeping the stack up

`python -m test.e2e.hold_stack [--analyze]` boots the same stack outside
pytest, prints its URLs and waits for Ctrl+C (or `AUDIOMUSE_E2E_HOLD_SECONDS`).
Use it to look at the instance the tests see, or to run the page smoke from
another interpreter: with `AUDIOMUSE_E2E_BASE_URL=http://127.0.0.1:8000` the
browser module attaches to that instance instead of booting its own. Chromium
under WSL needs system libraries that `playwright install --with-deps chromium`
installs with sudo; without them, run the page smoke from a Windows Python with
`playwright` and `pytest-playwright` installed against the held stack.

## Layout

- `conftest.py`: the session fixtures (`stack`, `api`, `db`, `navidrome`,
  `library`, `analyzed_library`, `page_base_url`), the liveness check that
  fails the next test when a stack process died, and the terminal summary.
- `stack/`: the harness. `boot.py` owns the bring-up order, `env.py` is the one
  place every environment variable and small-library knob is set (all of them
  names `config.py` already reads), `navidrome.py` runs the server and holds
  the Subsonic client used only to seed plays and assert playlists, `control.py`
  (the control-socket server), `seed.py` (the seed catalogue loader), `postgres.py`,
  `flask.py`, `workers.py`, `binaries.py`, `fpcalc.py`, `models.py`, `library.py`,
  `processes.py`, `ports.py`, `paths.py`.
- `library/`: the committed fixture audio, `manifest.json` (every count the
  tests use comes from it) and `ATTRIBUTION.md` (sources and licences).
- `e2e_helpers.py`: `assert_no_fp_ids` (catalogue ids never leave the API),
  id translation through `track_server_map`, playlist creation.
- `test_library_fixture.py` and `test_stack_smoke.py`: the contract of the
  library and of the harness itself.
- `test_00_cold_library.py`: the empty-library behaviour, before analysis.
- `test_1x` to `test_5x`, `test_90_cleaning.py`: one functionality each,
  including every clustering algorithm on the seeded catalogue, one real cron
  tick, the registry with the second instance and the default-server swap, and
  a provider migration to the second instance and back.
- `golden.py` and `seed/golden/`: the recorded exact answer of every
  deterministic API call (`test_32_golden_api.py` plus the analysis, sonic
  fingerprint, migration and cleaning modules), normalized so provider ids
  become names; clustering, temperature-driven alchemy and the 2D map
  coordinates (the UMAP projection is not seeded) are the only random outputs
  and stay out.
- `test_31_golden_answers.py`: the exact songs the ranking features return
  for three fixed seeded songs (similar songs, path, similar artists, text
  search, hyperbolic neighbours) and the score row of each, recorded in
  `seed/golden.json`; any change in those answers fails the test with the
  expected and the actual song. After an intended change re-record with
  `AUDIOMUSE_E2E_RECORD_GOLDEN=1 bash test/e2e/run_local.sh --no-browser -k golden`.
  A brute-force cosine ranking computed from the database cross-checks the
  similar-song answer independently of the file.
- `seed/` and `seed_builder.py`: the seed catalogue and the script that
  rebuilds it from the DCLAP licence lists.
- `test_33_ui_flows.py` (marker `browser`): the main forms operated in
  headless Chromium the way a person uses them (type into the search box, pick
  the autocomplete suggestion, set the options, press the button) for similar
  songs, song path, alchemy, text search, lyrics search, similar artists,
  hyperbolic neighbours, sonic fingerprint and the library browser. The rows
  on screen must equal the answer the page received, in order, and the songs
  shown are recorded in `seed/golden/test_33_ui_flows.json`; one flow creates
  a playlist from the page and reads it back from Navidrome.
- `test_pages_smoke.py` (marker `browser`): every page in headless Chromium.

## Conventions

- Tests never hard-code library counts or provider ids: `library.counts`,
  `library.pid(key)`.
- Every JSON payload goes through `assert_no_fp_ids`.
- Playlists a test creates on Navidrome are deleted in its finalizer.
- A cancel wipes the whole `task_status` table and a new main task deletes
  terminal rows, so no module asserts on another module's task row.
- The whole suite is Linux only and runs serially; `pytest test/` in one
  process is not supported (the unit suite's psycopg2 mask and the integration
  suite's pgserver instances do not coexist with a live stack).

## Not covered here

Instant playlist chat and AI naming (need an LLM), provider migration between
different provider types (only Navidrome to Navidrome runs here), the setup
wizard's POST (it schedules a web restart), users and auth (covered by
`test/integration/test_auth_barrier_integration.py`), Plex PIN, plugin catalog
installs, and the external lyrics API path.
