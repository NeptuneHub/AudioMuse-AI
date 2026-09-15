# Model coverage and version APIs

Both endpoints use the existing authentication and initial-setup policy, including
ordinary authenticated users, bearer-token callers, and authentication-disabled
installations. Neither requires an administrator. Responses use
`Cache-Control: no-store`, including authentication/setup errors.

## GET /api/models

Returns all four setup-wizard models. No parameter means global coverage only.
Pass `server_id=<id-or-name>` or `server=<id-or-name>` to additionally request local
coverage. `server` takes precedence; the resolved ID is echoed as `server_id`.
An explicitly empty or invalid selection returns 400 instead of global data.

```json
{
  "server_id": "selected-server",
  "models": {
    "musicnn": {"enabled": true, "global": {"count": 100, "total": 200, "percentage": 50.0}, "local": {"count": 20, "total": 40, "percentage": 50.0}},
    "clap": {"enabled": true, "global": {"count": 80, "total": 200, "percentage": 40.0}, "local": {"count": 10, "total": 40, "percentage": 25.0}},
    "lyrics": {"enabled": false, "global": {"count": 20, "total": 200, "percentage": 10.0}, "local": {"count": 2, "total": 40, "percentage": 5.0}},
    "neural-fingerprint": {"enabled": true, "global": {"count": null, "total": 200, "percentage": null}, "local": {"count": null, "total": 40, "percentage": null}}
  }
}
```

Values are illustrative. The keys match the setup wizard: `clap` is the DCLAP
model. MusiCNN is always enabled; the other enablement values read their effective
runtime configuration flags. Disabled models remain in the response and can
retain coverage from earlier analysis.

Each coverage object contains:

- `count`: indexed tracks, or `null` when unknown.
- `total`: tracks in the relevant catalogue, using `score` globally and existing
  server-availability rules locally. All four percentages use this denominator,
  including Lyrics. The setup wizard retains its existing lyrics-eligible
  denominator and visual bands; its UI response is unchanged.
- `percentage`: `count / total * 100`, rounded to two decimals and bounded to
  0-100. An empty catalogue is 0%; an unknown count gives `null`. A temporarily
  stale index can have a count above total after cleaning; its percentage is
  capped, without hiding the actual index count.

`server_id` and every `local` object are omitted when no server is requested.
Local counts apply the same canonical mapping and legacy-default rules as search;
a server with no indexed tracks never inherits the global count.

Global MusiCNN/DCLAP/Lyrics counts reuse the compact persisted IVF directory
header counts already used by the setup wizard. Global neural coverage uses the
resident fingerprint pack; if unloaded it remains unknown. Local paged-IVF checks
read only directory IDs and source mappings, not embedding/cell blobs, and do not
construct a search index or initialize a model. Scalar counts are cached for up
to 30 seconds, invalidated on source-mapping changes and index replacement.
Local neural coverage reuses the source-scoped resident-index status; loading,
failed or unloaded indexes report unknown counts. Coverage is a snapshot, not a
guarantee of successful inference or media-server playback.

The endpoint contains no version fields or recording limits. Disabled/unknown
models still return 200. Authentication/setup errors retain 401/403. Unexpected
coverage failures return 500 with `{"error": "Could not determine model coverage."}`;
no model paths, credentials, or exception details are exposed.

## GET /api/version

Returns only the actual runtime application version, including any release suffix:

```json
{"app_version": "3.6.1-rc.2"}
```

This route lives in `app.py`, uses the same authentication/setup barrier, and has
no minimum-version policy. It does not query coverage or initialize any models.

## Implementation sharing

`tasks/model_coverage.py` shares the setup wizard's count sources. The model-file
availability/enablement split remains in `tasks/neural_fingerprint.py`. Neither
metadata endpoint performs inference, analysis, or recording warmup. No database
migration, frontend change, or new dependency is required.
