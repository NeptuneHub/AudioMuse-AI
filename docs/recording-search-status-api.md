# Recording-search status API

`GET /api/recording_search/status?server_id=<id>` lets a client check capabilities
before requesting microphone access or uploading a recording. It uses the same
authentication and initial-setup policy as recording search: ordinary session
users and bearer-token clients are supported, and authentication-disabled
installations retain their existing behavior. No administrator role is required.

`server_id` or its existing alias `server` accepts an ID or display name. If both
are supplied, `server` takes precedence. Omission selects the default server;
the response always identifies the resolved source. Invalid selection returns
400. A missing default or unavailable source lookup returns a generic 500, never
an unscoped union-library result.

## Response contract (version 1)

```json
{
  "api_version": 1,
  "app_version": "3.6.0",
  "server_id": "configured-server-id",
  "enabled": true,
  "model_available": true,
  "ready": true,
  "index": {"state": "ready", "indexed_tracks": 12345},
  "recording": {
    "recommended_seconds": 20,
    "max_clip_seconds": 60,
    "max_upload_bytes": 1073741824,
    "default_n_results": 100
  }
}
```

Values above are examples; version, settings, limits and count come from runtime
state. `api_version` versions this contract independently of `app_version`, which
preserves the actual `APP_VERSION`, including any release suffix. The endpoint
does not impose a minimum application version.

- `enabled`: effective `NEURAL_FINGERPRINT_ENABLED`.
- `model_available`: required encoder and codebook files exist, independently of
  enablement. This check does not load/validate the files or guarantee inference.
- `ready`: enabled, model files available, and index state `ready`. An unloaded
  encoder does not prevent readiness; existing warmup/search initializes it.
- `indexed_tracks`: indexed IDs available on the selected server according to
  the existing search availability rules, including legacy default-server IDs.
  Never a global-library count. Unknown is `null`, not zero. This is index
  coverage, not a guarantee that every provider item is currently playable.
- `recording`: effective `RECORDING_SEARCH_RECORD_SECONDS`,
  `RECORDING_SEARCH_MAX_CLIP_SECONDS`, `RECORDING_SEARCH_MAX_UPLOAD_MB` converted
  to bytes, and `RECORDING_SEARCH_DEFAULT_N_RESULTS`, respectively.

| Index state | Meaning | Count |
| --- | --- | --- |
| `not_loaded` | No resident index; coverage unknown | `null` |
| `loading` | Index preparation/loading is in progress | `null` |
| `error` | A known index load/validation failure exists | `null` |
| `ready` | Loaded index contains tracks for the selected source | Positive integer |
| `empty` | Loaded index contains no tracks for the selected source | `0` |

Loading takes precedence over a previous error; a known error takes precedence
over any previously resident pack. The endpoint conservatively reports not-ready
until loading completes successfully. Enablement and file availability are
independent of index state: a disabled feature can retain a resident index.
A never-loaded index cannot establish emptiness and remains `not_loaded`.

## Read-only behavior and errors

GET does not load the encoder/index, renew the recording warmup timer, perform
inference, start analysis, or modify configuration. Availability counts are
cached per source and index build for 30 seconds, invalidated alongside existing
availability masks on mapping changes and index swaps. The first check may read
source mappings and count an in-memory mask; repeated checks reuse the count.
No fingerprint blobs are scanned. The cache is in memory only.

All responses carry `Cache-Control: no-store`, including early auth/setup errors.
Disabled, missing-model, loading, empty and known index-error states return 200.
Invalid source selection returns 400; unauthenticated access returns the existing
401, and initial setup retains the existing 403. Unexpected lookup/status failures
return 500 with `{"error": "Could not determine recording search status."}`.
No credentials, model paths or internal exception text are returned.

Clients can present setup/update guidance based on these fields, use the existing
POST warmup action when appropriate, then recheck status. Readiness is a snapshot:
search must still handle subsequent configuration changes and runtime errors.
