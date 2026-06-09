# How-To guide tooling

Generates the per-release user guide under `docs/howto/<version>/` — a
GitHub-readable `howto.md` (table of contents + a section and screenshot per
page) plus a `screenshots/` folder.

```
docs/howto/
  _tooling/                  <- this folder (scripts + prose template + CI stack)
    howto.template.md           the guide prose, with a {{VERSION}} placeholder
    howto_capture.py            drives a browser, screenshots every page
    render_howto.py             template + version -> docs/howto/<version>/howto.md   (stdlib)
    validate_howto.py           checks a rendered folder is complete & correct        (stdlib)
    make_howto.py               convenience: capture + render in one go
    docker-compose.howto.yml    throwaway app + Postgres + Redis used by CI
    _version.py                 reads APP_VERSION from config.py
    requirements.txt            playwright (capture only)
  2.1.4/                     <- a rendered release (howto.md + screenshots/)
```

## Safety (why the output is publishable)

* **Copyright** — track **title / artist / album** never appear. When capturing
  against a real instance every `/api/**` JSON response is intercepted and those
  fields are rewritten to placeholders (`Song Title 1`, …) before the page
  renders. When capturing in CI (`--mock-all`) the data is fabricated as
  placeholders to begin with.
* **Secrets** — server URLs, user IDs, tokens and passwords are blanked on the
  Setup, Sonic Fingerprint, Analysis, Instant Playlist and Users pages.
* Only **read-only** features are exercised; nothing is written to a media
  server. A few "result" screenshots use representative placeholder data.

## Two ways to produce the guide

### A) In CI on each pull request (no real instance) — `.github/workflows/howto.yml`

Runs on every pull request (open + each push) and on manual dispatch. The
version — and therefore the `docs/howto/<version>/` folder name, with the
leading `v` stripped — is read from `APP_VERSION` in `config.py`, **not** from a
git tag. The workflow:

1. starts a throwaway stack from `docker-compose.howto.yml` — the prebuilt
   image + an **empty** Postgres + Redis. Env vars clear the setup/auth barrier
   (no media server is contacted) and seed an `admin` account on boot.
2. waits for `GET /api/health`, then runs `howto_capture.py --mock-all` — which
   logs in and **fabricates every page's data in the browser**, so an empty
   database still yields fully-populated screenshots.
3. renders `howto.md`, validates the folder, and **uploads
   `docs/howto/<version>/` as a build artifact**.

It deliberately does **not** commit anything and never pushes to `main`:
download the artifact from the run, review the screenshots and `howto.md`, and
if you like the result commit the folder yourself (or just run option B locally
and commit). The app image defaults to
`ghcr.io/neptunehub/audiomuse-ai:<version>`, falling back to `:latest`; if that
image is private, add a `docker/login-action` step before "Start app stack".

`--mock-all` notes: data endpoints (`/api/search_tracks`, `/api/map`,
`/api/dashboard/summary`, …) are answered from `build_mock()` in
`howto_capture.py`; config/setup/users endpoints pass through to the real app
(and are masked). Lyrics and DCLAP gate their inputs server-side on built
indexes, which an empty DB doesn't have — so in mock mode those inputs are
re-enabled and the "index not built" banner is hidden before the demo runs.

If you add a page or change an endpoint's response shape, update `build_mock()`
to match.

### B) Locally, against your own analysed instance

```bash
pip install -r docs/howto/_tooling/requirements.txt
playwright install chromium      # or rely on an installed Google Chrome (default channel)

cd docs/howto/_tooling
python make_howto.py --base-url http://192.168.3.204:8000 --user root --password root
```

Writes `docs/howto/<APP_VERSION>/` (version from `config.py`; override with
`--version v2.2.0`). Real metadata is masked on the way through. Commit the
folder. URL/credentials also read from `HOWTO_BASE_URL` / `HOWTO_USER` /
`HOWTO_PASSWORD`.

* Prose change for everyone? Edit `howto.template.md` once, then
  `python render_howto.py --version vX.Y.Z` (no browser).
* Re-render only (no recapture): `make_howto.py --skip-capture`.

## Tweaking individual scripts

* `render_howto.py` / `validate_howto.py` are pure stdlib (no browser, no app) —
  safe and fast to run anywhere, including as a release gate.
* `validate_howto.py` fails if a referenced screenshot is missing/empty, an
  in-page link is broken (GitHub heading-slug rules), or the page isn't stamped
  with the expected version.
