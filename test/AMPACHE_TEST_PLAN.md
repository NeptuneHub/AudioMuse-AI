# Ampache provider - system test plan (PR #810)

Written against the maintainer review on PR #810. Every item is a check that can
fail, with the command or query that decides it. Phases A-C are the three areas
the maintainer called out; phase D maps one test to each fix in the review; phase
E covers the offline/failure paths. Phase F covers running Ampache with local
file access, and applies only to a branch that carries that feature.

Record results in the sign-off table at the bottom and paste it into the PR.

---

## 0. Record the environment first

The review notes the tested-version badge was deliberately left unchanged
because no Ampache version was verified. That makes this section a deliverable,
not preamble - the backend declares `version=8.0.0` on **every** request
(`_handshake`, `_fetch`), so the server's API version is the single most
important fact about this test run.

| Fact | Value |
| --- | --- |
| Ampache version (Admin -> Server Info) | |
| Ampache API version reported by `ping` | |
| Second provider + version (multi-provider / migration tests) | |
| AudioMuse-AI commit under test | |
| Library size (tracks / albums / catalogs) | |
| Auth mode tested (password, API key, both) | |
| Deployment (docker compose, worker replicas) | |
| `LOCAL_FILE_ACCESS` on/off, and the library mount if on | |
| `LOCAL_FILE_ROOTS` / `LOCAL_FILE_PATH_MAP` / `LOCAL_FILE_REQUIRE_READONLY` | |

Get the API version without the UI:

```bash
curl -s "$AMPACHE_URL/server/json.server.php?action=ping&version=8.0.0" | head -c 400
```

**Ampache API 8 is a requirement, not a preference.** The backend declares
`_API_VERSION = '8.0.0'` and gates on `_MIN_API_MAJOR = 8`, because album
browsing needs the `catalog` id on album objects (added in 8) and the `cond`
browse filter. `test_connection` warns when a server reports an older API, so
A9/A10 should also confirm that warning appears when pointed at one.

Confirmed working on Ampache 8 during this round: `cond=catalog,<id>` on the
`albums` action (stable back to API 6), and `sort=addition_time,DESC`, which is
what keeps "recent albums" genuinely newest-first. `cond` takes ONE value per
condition, so several catalogues mean one browse each - and an unrecognised
`cond` is **silently ignored**, returning the whole library, which is why rows
are re-checked against their `catalog` id (see D12a).

---

## 1. Offline checks (no Ampache required)

Run these before touching a live server; they must be green at the commit you test.

```bash
# Ampache backend + the cross-language provider-registration contract
pytest test/unit/test_mediaserver_ampache.py test/unit/test_provider_registration_contract.py -q

# Repo gates that this PR has broken before
pytest test/unit/test_file_header_convention.py test/unit/test_import_architecture.py -q
ruff check tasks/mediaserver/ static/ test/unit/test_mediaserver_ampache.py
codespell tasks/mediaserver/ampache.py

# Only when the branch under test ALSO carries local file access - see phase F.
# tasks/mediaserver/local_file.py does not exist on AmpacheConnector, so this
# line errors with "file or directory not found" there rather than failing.
pytest test/unit/test_local_file_access.py -q
```

Expected: all pass. The contract test is what proves Ampache is registered in
`config.MEDIASERVER_FIELDS_BY_TYPE`, the dispatcher, `app_music_servers`,
`app_provider_migration`, `provider_probe`, both JavaScript files, both HTML
dropdowns and `docs/PARAMETERS.md` - so a missing registration fails offline
rather than mid-demo.

---

## 2. Phase A - Setup wizard, fresh install

Start from an **empty database** (`./ampache-reset.sh reset`), with no
`MEDIASERVER_*` variables in the environment, so the wizard really opens blank.

| # | Step | Expected | How to verify |
| --- | --- | --- | --- |
| A1 | Open `/setup` on a clean DB | Wizard opens, no phantom server row | `SELECT COUNT(*) FROM music_servers;` returns 0 |
| A2 | Step "Media server type" dropdown | **Ampache** listed | Both dropdowns: the multi-server table picker and the basic `MEDIASERVER_TYPE` select (`templates/setup.html`) |
| A3 | Select Ampache | Three fields render: URL, username, password/API key | `serverFields.ampache` in `static/setup.js` |
| A4 | Inspect the password field | `type=password`, value never echoed in plain text | `AMPACHE_PASSWORD` is in `SECRET_FIELDS` (`app_setup.py`) and in `secretKeys` (`static/setup.js`) |
| A5 | Type all three values, switch type to Navidrome, switch back | Ampache values still present | `saveCurrentServerValues()` must list all three `AMPACHE_*` keys |
| A6 | Tooltip text | Explains password-or-API-key, and that username may be blank | |
| A7 | **API-key-only install**: blank username, API key in the password field, Save | Accepted | This is the regression the review calls out; `MEDIASERVER_OPTIONAL_FIELDS_BY_TYPE` makes `AMPACHE_USER` optional and `missing_required_creds()` must not reject it |
| A8 | Test Connection with good creds | OK plus a sample count | `test_connection` samples 100 songs |
| A9 | Test Connection with a **wrong** password | Fails, and reports an auth failure (not a network error) | `auth_failed: true` comes from error kind `auth`; codes 4742/4704 |
| A10 | Test Connection against a catalog added with a **relative** path | Warning about relative paths and manual matching in step 4 | `detect_path_format() != 'absolute'` |
| A11 | Library picker | Real catalog names, checkboxes, not `[object Object]` | `list_libraries` returns lowercase `id`/`name`; the admin JS reads `lib.name` |
| A12 | Save with one catalog selected | Stored as a name list | `SELECT music_libraries FROM music_servers;` shows the catalog name, not `[object Object]` |
| A13 | Finish the wizard, let the app restart | Boots configured, no setup redirect | `GET /api/health` 200, `/` no longer redirects to `/setup` |
| A14 | Start an analysis | Albums are discovered and queued | Worker logs show `Fetched N songs from Ampache` / album jobs; `SELECT COUNT(*) FROM score;` climbs |
| A15 | Analysis with a library filter set | Only the chosen catalog's tracks appear | Compare `score` rows against the catalog's track count in Ampache |
| A16 | Environment-variable bootstrap instead of the wizard | With `MEDIASERVER_TYPE=ampache` + `AMPACHE_URL`/`AMPACHE_PASSWORD` set on a clean DB, the registry self-seeds and the wizard is skipped | `_seed_registry_from_legacy_config` inserts one default row; `AMPACHE_USER` empty must still bootstrap (`_is_valid_server_config` honours the optional field) |

---

## 3. Phase B - Multiple providers

Keep the Ampache server from phase A, then add a second one (any other
provider), and repeat the reverse order in B9.

| # | Step | Expected | How to verify |
| --- | --- | --- | --- |
| B1 | Setup wizard / music-servers admin page -> Add server | Ampache selectable **here too** | `credDefs.ampache` in `static/music_servers_admin.js` |
| B2 | Add the second provider with its own creds | Two rows, one default | `SELECT name, server_type, is_default FROM music_servers;` |
| B3 | Library picker for each server | Each lists its **own** libraries | Per-server `list_libraries` via `POST /api/servers/libraries` |
| B4 | Add a second server whose name duplicates the first | Rejected with a clear message | Unique-name index + `_name_taken` |
| B5 | Switch the default to the other server | Exactly one default; config follows | The partial unique index allows only one `is_default`; `_apply_default_to_config()` runs |
| B6 | Sweep after adding | Sweep task completes, track counts populate | `SELECT name, track_count FROM music_servers;` |
| B7 | Analyse on both servers | Rows mapped per server | `SELECT server_id, COUNT(*) FROM track_server_map GROUP BY 1;` |
| B8 | A track present on both servers | One `score` row, two mappings | Same query as B7 plus `SELECT COUNT(*) FROM score;` |
| B9 | Add Ampache **second** (fresh DB, other provider first) | Works symmetrically | Rules out order-dependent registration |
| B10 | Instant playlist while the **non-default** server is selected | Created on that server, not the default | Ampache id translation happens once, in `_to_server_ids`; check the playlist appears on the intended server only |
| B11 | Delete the non-default server | Its mappings go, `score` rows stay | `track_server_map` has `ON DELETE CASCADE`; `SELECT COUNT(*) FROM score;` unchanged |
| B12 | Per-user credentials (if used) | Playlist writes use the caller's creds, not the default server's | Fixed in this PR: `create_playlist`/`delete_playlist` now take `user_creds`; confirm ownership of the created playlist in Ampache |

---

## 4. Phase C - Provider migration, both directions

Run C1-C8 for **Ampache -> other**, then again for **other -> Ampache**. Take a
`pg_dump` first; migration rewrites `score.item_id`.

| # | Step | Expected | How to verify |
| --- | --- | --- | --- |
| C1 | Open `/provider_migration` | Intro mentions Ampache; Ampache in the target dropdown | `templates/provider_migration.html` |
| C2 | Choose Ampache as target | URL / username / password fields appear, password masked, labels bound to inputs | `data-for="ampache"`, `label for=` / `input id=` pairs |
| C3 | Test Connection (target) | OK plus sample count; relative-path warning when applicable | `provider_probe.test_connection` -> `_normalize_provider_type` accepts `ampache` |
| C4 | Migrate **from** Ampache | Source creds resolve; no `None` provider | The review's fix replaced hardcoded branches in `_current_provider_creds()` with `MEDIASERVER_FIELDS_BY_TYPE`/`MEDIASERVER_CRED_KEY_BY_FIELD`; verify it returns `('ampache', {url,user,password})` |
| C5 | Catalogue fetch for matching | Full track list with absolute paths | `fetch_all_tracks` -> `get_all_songs`; spot-check `Path`/`FilePath` are absolute |
| C6 | Automatic path matching | Most albums matched by path | Session summary counts |
| C7 | Manual match (step 4) | Album search + track list work against Ampache | `search_albums` (`advanced_search` on title), `get_tracks_from_album` (`album_songs`) |
| C8 | Confirm the rewrite | `item_id`s become the new provider's ids; embeddings still joined | `SELECT item_id FROM score LIMIT 5;` (Ampache ids are bare row ids, e.g. `1234`, **not** `so-1234`); `SELECT COUNT(*) FROM embedding;` unchanged |
| C9 | Unmatched tracks | Listed, not silently dropped | `MIGRATION_UNMATCHED_ALBUMS_PAYLOAD_LIMIT` caps the payload at 200 - confirm the total is still reported |
| C10 | Local playlists after migration | Still resolve to real tracks | Open a stored playlist; re-create it on the server |
| C11 | Sonic Fingerprint / Alchemy after migration | Still produce playlists | See D1/D2 |

---

## 5. Phase D - One test per fix in the review

| # | Fix under test | Test | Pass condition |
| --- | --- | --- | --- |
| D1 | `create_or_replace_playlist` arity (3 args) | Run the Sonic Fingerprint cron | Playlist created; **no** `TypeError` in worker logs |
| D2 | Same, via Alchemy Radio | Create an Alchemy anchor + radio, let it run | Playlist created/replaced, no traceback |
| D3 | Return contract is a dict | Chat playlist endpoint + Sonic Fingerprint | No false 500; response carries the playlist name/id (`{'Id','Name'}`) |
| D4 | `_instant` suffix | Create an instant playlist named e.g. `Test Mix` | Ampache shows `Test Mix_instant`; a manually created `Test Mix` is untouched |
| D5 | Instant playlists are strip-normalised | Name with leading/trailing spaces | `Spaced_instant`, no double space |
| D6 | Download checks HTTP status | Analyse an album where one track id no longer exists | No file written for it; log shows the failure; no bogus `score` row |
| D7 | Ampache JSON error under HTTP 200 | Invalidate the session mid-analysis (change the password in Ampache, or expire sessions) | Log: `returned JSON instead of audio` **or** a single silent re-handshake; **no** JSON file saved as audio. Check with `find data/temp-audio-* -size -4k` and `file` on anything found |
| D8 | Session expiry retries once | Same as D7 | Analysis continues after one retry; a second failure ends with `Ampache session could not be renewed` |
| D9 | Play counts, not ratings (`frequent`) | Give one track many plays and a different track 5 stars, then run Sonic Fingerprint | The **most played** track is used, not the highest rated (`stats?type=song&filter=frequent`) |
| D10 | Library picker shape | Phase A11/A12 | Catalog names render; nothing saves `[object Object]` |
| D11 | API-key-only install | Phase A7 | Saves with a blank username |
| D12 | Recent albums honour the library filter | Set `MUSIC_LIBRARIES` to a catalog that is **not** the newest content, then start an analysis with a recent-album limit | The requested number of albums comes back (the backend pages on); the run does not stall at zero |
| D12a | Album filter is applied by the **server** | Analyse with one catalog selected | Album discovery browses `albums` with `cond=catalog,<id>` (one browse per selected catalog, since `cond` takes a single value per condition). Album count matches the catalog's album count in Ampache, not the whole library. Ampache's album objects historically carry no `catalog` field, so a locally-applied filter would drop every album - the regression this replaced |
| D12b | Analysing **every** album is not capped at one page | Set the recent-album count to 0 (all) on a library with more albums than one page | `albums_found` matches Ampache's `total_count`, not the first page. Verify with `curl ".../albums?limit=1"` and compare `total_count` |
| D12c | An empty result is never a silent success | Point the filter at a catalog with no albums, or break the browse | Log carries `AMPACHE RETURNED NO ALBUMS` / `AMPACHE ALBUM FETCH FAILED`; the task does not report `SUCCESS` with zero albums and no explanation |
| D13 | Server-side filtering of the full fetch | Run a sweep with a filter on a large library | Log line `Fetched N songs`, where N is the catalog's size, not the whole library. The request is a `songs` browse with `cond=catalog,<id>`, **one browse per selected catalogue** - not an `advanced_search` with the catalogues OR'd together, which asks the server for the same rows via a much heavier query |
| D14 | Filter fallback is safe | Make a page fail once (e.g. bounce the server mid-page), then make it fail twice | Once: the SAME filtered page is retried, `cond` intact - a single transient error must not cost a full-library walk. Twice: `AMPACHE CATALOGUE FETCH FAILED ... INCOMPLETE` and the partial list is not treated as deletions |
| D14a | An ignored `cond` does not multiply the walk | Point at a server that ignores the condition (or fake it) so a browse returns other catalogues' rows | One unfiltered walk filtered locally, warned once - **not** a full-library walk per selected catalogue. A page that reports no `catalog` field at all is treated the same way, since the filter cannot be verified |
| D14b | Page size honoured | Set `AMPACHE_PAGE_SIZE=100` and sweep | `limit=100` in the browse URLs. Note this cuts request count, not the server's per-row cost |
| D15 | No password in a URL | Configure a **password** (not an API key) and watch the logs | No cleartext secret anywhere: `docker compose logs \| grep -F "$AMPACHE_PASSWORD"` returns nothing; auth params appear as `[REDACTED]`. Note: with a **blank username** the secret is sent as an API key by design - only ever use a real API key in that mode |
| D16 | Rotated password does not reuse a session | Analyse, change the password in Ampache, run Test Connection with the **old** password | Fails (the token cache key includes a hash of the password) |
| D17 | Empty playlist is not reported as created | Point a playlist at ids Ampache rejects | Returns `None`; log names the playlist and the count; UI does not claim success |
| D18 | Failed delete aborts the replace | Make the delete fail (e.g. a playlist owned by another user) | No second playlist with the same name |
| D19 | Playlist lookup uses caller creds | Multi-server or per-user run | The lookup hits the intended server (see B12) |
| D20 | Lyrics handshake is time-bounded | Point at a slow/blackholed Ampache and fetch lyrics | Returns within ~`MUSICSERVER_LYRICS_TIMEOUT` (2.5s default), not 30-60s; the worker is not blocked |
| D21 | Lyrics read | Analyse a track that has lyrics in Ampache | Lyrics stored, from the `lyrics` field the `album_songs` response already carried (see D27) |
| D22 | Connection test warns on relative paths | Phase A10 | Warning shown |
| D23 | Docs | `docs/PARAMETERS.md`, `README.md`, `docs/MULTI_SERVER.md` | All three `AMPACHE_*` documented; Ampache in the supported-provider lists; `AMPACHE_PASSWORD` states an API key is preferred |
| D24 | API key uses the bearer header, not a handshake | Configure a real API key (blank username) and run an analysis | Access log shows **one** `action=ping` per worker process and then no `handshake` at all; every later request carries no `auth=` in the query string. `docker compose logs \| grep "Authorization header"` shows the one-line confirmation with the API version |
| D25 | A server that refuses the header still works | Point a key-shaped secret at a server with `ApiKeyAuthHeader` unavailable, or revoke the key | Log: `would not take the API key as a bearer token`, then the handshake path takes over and the run continues. Probed once per credential set, not once per request |
| D26 | Password mode is unchanged | Configure a password rather than a key | No `ping` probe, no `Authorization` header, straight to `handshake` - a password must never be sent as a bearer token, since `findByApiKey` would only refuse it |
| D27 | The lyrics stage does not refetch each song | Analyse an album with `LYRICS_ENABLED` on and watch the access log | **One** `action=album_songs` and **no** `action=song` for those tracks. Ampache serialises both through the same `Json8_Data::songs_array`, so the album response already carried `lyrics`; the per-track `song` call repeated all of that row's hydration (rating, userflag, art, album and artist lookups) to re-read one field. A track analysed outside an album context still falls back to `action=song` |
| D27a | An absent field is not read as "no lyrics" | Analyse against a server whose `album_songs` rows omit `lyrics` entirely | Falls back to `action=song` per track and still stores lyrics. Only an explicit `lyrics: null` is cached as "this song has none"; a missing key must not silently blank the lyrics stage |
| D28 | The dispatch loop browses instead of fetching every track | Start an analysis and watch the access log during album dispatch | One `action=browse&type=album&filter=<album>&catalog=<id>` per album, **no** `action=album_songs` from the dispatcher. `browse` answers through `Catalog::get_name_array` (id/name/prefix/basename), so it skips the per-song hydration `album_songs` pays for. The album *job* still calls `album_songs` once - it needs the metadata |
| D28a | A truncated browse never skips an album | Force a `total_count` that disagrees with the rows returned | Falls back to `action=album_songs` for that album and the album is still analysed. This is the important one: the dispatcher compares the track count against the number already analysed, so believing a short list would mark an unfinished album complete and skip it silently |
| D28b | A server that refuses the browse still works | Point at a server that rejects `browse` for `type=album` (e.g. one requiring a catalogue the album never reported) | Falls back to `album_songs`; the run continues at the old cost. No album is skipped or double-counted |
| D28c | The catalogue id comes from album discovery | Analyse with and without `MUSIC_LIBRARIES` set | `catalog=<id>` is present on the browse whenever the album row reported one. Stock Ampache 8 **requires** it for a sub-type browse (`BrowseMethod`: `foreach (['filter', 'catalog'] ...)`), so an install that has not relaxed that check depends on it being sent |
| D28d | Other providers are unaffected | Run an analysis on Jellyfin/Navidrome/Emby/Plex/Lyrion | Unchanged behaviour: `get_album_track_ids` is optional, and the dispatcher falls back to `get_tracks_from_album` for a backend that does not implement it |

---

## 6. Phase E - Failure and offline paths

| # | Scenario | Expected |
| --- | --- | --- |
| E1 | Ampache down before an analysis | Clear error, no crash loop, no partial catalogue treated as deletions |
| E2 | Ampache goes down **mid**-analysis | `AMPACHE CATALOGUE FETCH FAILED ... INCOMPLETE - do not treat missing tracks as deleted` in the log; no mass deletion in `score` |
| E3 | Library filter matches no catalog | `matched no catalogs; returning no songs` warning; empty result, not an exception |
| E4 | Wrong URL (typo / missing scheme) | Network-kind error, redacted; test connection fails cleanly |
| E5 | HTTP 500 from the download endpoint | No file written (`raise_for_status`), track skipped, run continues |
| E6 | Worker restart mid-run (`docker compose restart audiomuse-ai-worker`) | Jobs resume; no duplicate playlists; no orphan temp files |
| E7 | Two analyses at once (or an analysis plus a chromaprint backfill) | With scaled workers, confirm the shared temp dir is per-replica (`tmpfs`) - `clean_temp()` wipes the whole `TEMP_DIR` at the start of a main analysis task |
| E8 | Very long track / large FLAC | Downloads and analyses within `AUDIO_LOAD_TIMEOUT` |

---

## 7. Phase F - Ampache with local file access

**Only applies when the branch under test carries `tasks/mediaserver/local_file.py`**
(`LocalFileAccess` / `localfiles`, or the combined `Ampache+localfile`). It is not
present on `AmpacheConnector` alone - skip the phase there and record it as
skipped rather than passed.

Local file access reads each track from a mounted library instead of downloading
it, using the path the provider reported. It couples to the Ampache backend
through exactly one thing: `link_local_copy` reads `item['Path'] or
item['FilePath']`, which the Ampache backend fills from the native `filename`
field - the same field A10 and C5 check for absolute paths. Relative catalogue
paths therefore disable local access as well as weakening migration matching.

**The design is deliberately fail-soft: every failure returns `None` and the
caller downloads as usual.** That is correct behaviour and it is also the whole
testing problem - a completely broken mount yields a *successful* analysis that
is merely slower, with no error anywhere. F2, F3 and F6 are the phase; the rest
are edges.

Deployment used for this round (record it in section 0):

```yaml
# docker-compose.yaml - on BOTH the flask and the worker service
environment:
  LOCAL_FILE_ACCESS: "true"
  LOCAL_FILE_ROOTS: /mnt/files-music/albums
volumes:
  - /host/path/to/albums:/mnt/files-music/albums:ro
```

`LOCAL_FILE_PATH_MAP` is empty here, which is correct **only** when the container
mount point is the same path Ampache reports. Note that the host source and the
container target are independent: the target must match Ampache's reported path,
the source is wherever the files actually live on the host. Setting the source
equal to the target is not a fix for a wrong path - it just moves the problem to
F2.

| # | Step | Expected | How to verify |
| --- | --- | --- | --- |
| F1 | Config reaches **both** services | `LOCAL_FILE_*` and the bind present on flask *and* worker | `docker compose config \| grep -B2 -A2 LOCAL_FILE` - analysis runs on the worker, so a worker-only omission disables the feature while the UI looks configured |
| F2 | The mount is populated **inside** the container | Tracks listed | `docker exec <worker> ls /mnt/files-music/albums \| head`. Empty is the failure to hunt for: Docker **auto-creates a missing bind source as an empty directory** instead of erroring, so a wrong host path mounts cleanly and serves nothing |
| F3 | The root is not writable **by this process** | `False` | `docker exec <worker> python -c "import os; print(os.access('/mnt/files-music/albums', os.W_OK))"`. This is the check `_writable()` makes; its docstring is explicit that the property is "can this process change anything here", not "is the mount flagged ro". `docker inspect ... rw=false` is a different question and not sufficient |
| F4 | A writable root is skipped, not used | Warning, and tracks download | Drop `:ro`, recreate, analyse. Log: `Local file access is SKIPPING ... because this process can write to it`. Confirms `LOCAL_FILE_REQUIRE_READONLY` defaults to **true**, so a writable mount silently costs you the feature |
| F5 | Path map when the mount point differs | Tracks still resolve | Mount at `/music` instead, set `LOCAL_FILE_PATH_MAP=/mnt/files-music/albums=/music`, confirm F6 still fires. Longest server prefix wins, so overlapping pairs are worth one test |
| F6 | Success is announced | At least one info line per replica | `docker compose logs <worker> \| grep "serving tracks from disk"` -> `Local file access is serving tracks from disk (first hit: ...)`. **Absence of this line is the only signal that everything silently downloaded** - there is no error to look for. Expect it REPEATED per replica, not once: `_announced` is a module global and `ReregisteringWorker` (`rq_heartbeat_worker.py:166`) forks a child per job on Linux, so every job announces afresh. Repeats are correct behaviour here, not a leak |
| F7 | Tracks are linked, never copied | Symlinks into the library | `docker exec <worker> ls -la /app/temp_audio` during a run - entries are `l`-mode links pointing at `/mnt/files-music/albums/...`, not files of track size. A hardlink is the documented Windows fallback |
| F8 | The library is never modified or deleted | Files byte-identical after a full analysis | `sha256sum` a sample before and after, and confirm the count is unchanged. This is the item that matters most: the pipeline **deletes whatever `download_track` returns** (the `finally` in `tasks/analysis/album.py`), so local access returning the library file itself would delete the user's music. It returns a TEMP_DIR link precisely so the unlink drops the link only |
| F9 | A path outside the roots is refused | Warning, downloads instead | Narrow `LOCAL_FILE_ROOTS` to a subdirectory and analyse a track outside it. Log: `Local file access refused: ... outside LOCAL_FILE_ROOTS` |
| F10 | A symlink out of the library is refused | Same refusal | Place a symlink inside the library pointing outside it, and analyse that track. `realpath` is resolved *before* the allowlist check, so the literal path passing is not enough - this is what stops a hostile or wrong provider path becoming an arbitrary-file reader |
| F11 | `LOCAL_FILE_ACCESS` on with no roots set | Everything downloads, warned once | Unset `LOCAL_FILE_ROOTS`. Log: `LOCAL_FILE_ACCESS is enabled but no usable root is configured`. Default is empty, i.e. refuse everything |
| F12 | Concurrent workers on one album | No corrupt or half-made links | Scale worker replicas and analyse; links are staged then `os.replace`d, so two workers handed the same track cannot collide. Also confirm E7's per-replica temp dir still holds |
| F13 | Ampache catalogue with **relative** paths | Local access declines, downloads instead | The A10 / C5 catalogue. A relative `filename` cannot resolve inside the roots, so this must fall back rather than half-read |
| F14 | Zero-byte or missing file | Track skipped, downloaded instead | Truncate a file in a writable copy of the library; `isfile`, `R_OK` and `getsize() <= 0` all return `None` silently by design |

---

## 8. Known limitations to state in the PR

Not defects, but they should be written down rather than discovered later:

1. **`get_last_played_time` always returns `None`.** Ampache's `stats` action
   ranks songs by play count but exposes no per-song "last played at", so
   recency-weighted callers (Sonic Fingerprint) fall back to play counts on
   Ampache. Documented in the function docstring.
2. **Ampache API 8 or newer is required**, and stated as such in
   `docs/PARAMETERS.md` and the module docstring. Older servers cannot express
   the library filter (no `catalog` on albums, and an unknown `cond` is ignored
   rather than refused), so album discovery stops with an explicit error instead
   of analysing an unfiltered library. `test_connection` warns before it gets
   that far. The tested-version badge should record the Ampache 8 build used
   here.
3. **Ampache track ids are bare row ids** (`1234`), while the same library read
   through the Subsonic/`navidrome` backend keys as `so-1234`. The prefix is
   Ampache's own doing, not something either backend adds: its OpenSubsonic layer
   stamps a type prefix on every object id - `so-` songs, `al-` albums, `ar-`
   artists, `pl-` playlists, `mf-` catalogs - see the `SUBID_*` constants in
   `src/Module/Api/OpenSubsonic_Api.php`. Nothing in this repo builds or strips
   it, which is why grepping for `so-` here finds only prose. The two paths are
   therefore not interchangeable for an already-analysed library - moving between
   them is a provider migration, not a config change.
4. **Playlists are created `private`, so who can see one depends on which Ampache
   account the credentials belong to.** `create_playlist` sends `type=private`
   explicitly, and Ampache shows a private playlist to its owner and to admins -
   an admin sees every private list on the site. So a playlist created by a
   non-admin API user is invisible to everyone else, and if you browse Ampache as
   a different account you will conclude nothing was created. Check under the
   SAME user the API key or password belongs to, ideally an admin, before calling
   a playlist missing. Then confirm `get_all_playlists` (used by `_automatic`
   cleanup) lists it - otherwise automatic playlists can be orphaned.
5. **This backend puts the secret in the query string, but Ampache does not
   require that** - the limitation is ours, not the server's. `Gatekeeper::getAuth()`
   (`src/Module/Api/Authentication/Gatekeeper.php`) checks three header forms
   *before* it looks at the query string:

   - `Authorization: Bearer <token>`
   - `Authorization: ApiKey <token>`
   - a bare `auth: <apikey>` header

   The GET/POST `auth` parameter is an explicit legacy fallback, and it prefers
   POST over GET - so even without headers a POST keeps the secret out of the URL.
   Header auth engages only when **no** `auth` parameter is sent
   (`ApiHandler.php`: `if (!isset($input['auth']))`), so passing `auth=` in the
   query string is what disables it.

   **With an API key there is also no handshake step at all.** `ApiHandler`
   resolves the user straight from the key (`findByApiKey`) and then creates the
   session itself, rewriting the token to an md5 of the username - the code's own
   comment is "Continue with the new session string to hide your header token".
   Present the key in a header on the first real request and the session is
   established for you.

   **This backend now takes that path for a key-shaped secret** (32+ hex chars):
   one bearer `ping` settles it per credential set, and every request after that
   carries `Authorization: Bearer` with no `auth` in the query string. So an
   API-key install puts no secret in the URL and no secret in the access log.

   A **password** install still handshakes and still carries `auth=` in every URL.
   That value is a time-salted hash rather than the password itself, so it is a
   session credential rather than a reusable one - but it is live for the session
   window and it lands in the web server's access log, as the nginx log from this
   round shows. Hence: prefer an API key with a blank username; if you must use a
   password, treat the access log as holding live session tokens.
6. **Local file access fails silently by design** (only when the branch carries
   it - phase F). Every unusable path returns `None` and the track is downloaded
   instead, so a misconfigured mount, a writable root, a wrong `LOCAL_FILE_ROOTS`
   or an absent path map all present as a normal, slightly slower, *successful*
   analysis. The single positive signal is the info line
   `Local file access is serving tracks from disk`. Anyone testing this should be
   told to assert the presence of that line rather than the absence of errors.

   Note what "once" means for `_announced` and `_warn_once`: they are module
   globals, and on Linux the forking `ReregisteringWorker` gives every job a new
   child process, so both reset per JOB rather than per worker. The announcement
   and the writable-root / outside-roots warnings therefore repeat throughout a
   run - louder than the "logged once per worker" wording in the module suggests.
   The same reset is why each job performs its own handshake.

---

## 9. Why the Navidrome/OpenSubsonic backend does not cover Ampache

Ampache does expose a Subsonic API, so "just use the `navidrome` provider" is a
fair question. It fails for concrete reasons in `tasks/mediaserver/navidrome.py`:

1. **Whole-library enumeration relies on Navidrome-specific behaviour.**
   `get_all_songs` walks the library with `search3` and an **empty** query
   string (`{"query": '', "songCount": 500, ...}`). "Empty query returns
   everything" is a Navidrome convenience, not a Subsonic guarantee; a server
   that treats an empty query as "no matches" yields zero songs and an install
   that never analyses anything. The Ampache backend instead pages the native
   `songs` action, or `advanced_search` when a catalogue filter is active.

2. **Authentication cannot use an Ampache API key.** The Subsonic path sends
   `u=<user>&p=enc:<hex>` with `v=1.16.1` - a username plus a password, on every
   request. Ampache's own handshake accepts **either** a password or an API key
   in one field, which is why `AMPACHE_USER` is optional. The documented
   API-key-only install is simply not expressible over Subsonic. Ampache's
   Subsonic endpoint also has to be enabled server-side, which the AudioMuse
   user may not control.

3. **Library filtering maps onto the wrong concept.** `MUSIC_LIBRARIES` is
   resolved through `getMusicFolders` + `getAlbumList2?musicFolderId=`. Ampache
   organises content in **catalogs**; when the Subsonic folder list does not
   present them under the names the user configured,
   `_get_target_music_folder_ids` returns an empty set and *nothing* is analysed.
   The native backend resolves the same setting against the `catalogs` action and
   pushes it into the query as `rule_*=catalog`.

4. **File paths, which drive migration matching, are weaker.** The Subsonic path
   takes `s.get('path') or s.get('url')`, which is optional and often relative in
   Subsonic responses. The native API returns `filename`, mapped to
   `Path`/`FilePath`, and `test_connection` actively warns when those are not
   absolute. Path quality decides how much of a provider migration matches
   automatically.

5. **Play statistics have a different vocabulary - and a different granularity.**
   The Subsonic path asks for `getAlbumList2?type=frequent`, i.e. albums. The
   native API answers at song level (`stats?type=song&filter=frequent`). This is
   also exactly where the review found a bug: Ampache defines `highest` as
   *highest rated* and `frequent` as *most played*, so the Subsonic-shaped
   assumption produced ratings-based fingerprints.

6. **Lyrics take one call instead of a guess.** Native `song` returns the
   `lyrics` field directly; Subsonic's `getLyrics` is artist+title based and
   unreliable for exact-track lookups.

7. **Fields Subsonic has no room for.** Replay gain, R128, multiple artists and
   the stream format come back from the native API and are dropped by the
   Subsonic shape (see the module docstring).

And the practical consequence, worth repeating: the two paths key the same
library differently (`1234` vs `so-1234`), so they are not two ways of reading
one catalogue - they are two catalogues.

---

## 10. Sign-off

| Phase | Result | Notes / evidence |
| --- | --- | --- |
| 1. Offline checks | | |
| A. Setup wizard, fresh install | | |
| B. Multiple providers | | |
| C. Migration Ampache -> other | | |
| C. Migration other -> Ampache | | |
| D. Fix-by-fix matrix | | |
| E. Failure paths | | |
| F. Local file access (or N/A - not on this branch) | | |
| Ampache version tested | | |
| Local file access: on/off, and mount verified per F2/F3/F6 | | |
