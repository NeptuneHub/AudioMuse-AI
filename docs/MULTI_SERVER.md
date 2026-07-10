# Multiple Music Servers

AudioMuse-AI can talk to several media servers at once - for example a Navidrome
plus two Jellyfins plus a Plex, in any combination, including several instances
of the same type. This is fully backward compatible: an install that only ever
configures one server behaves exactly as it always has.

## The model in one picture

```
Existing APIs / plugins (no server param)
        |
        v
   Default server ---- current config, item_ids, tables and API responses unchanged

Optional ?server=<id> (menu dropdown / API param)
        |
        v
   Server registry -> bound provider client -> that server's catalogue
```

- The **default server** is the one you configure in the Setup wizard. Its track
  ids are the canonical `item_id` used everywhere in the database. Analysis,
  search, similarity and every existing API keep working against it unchanged.
- **Additional servers** are stored in a registry (`music_servers` table). Each
  keeps its own credentials and library filter.
- A **mapping table** (`track_server_map`) records, per analyzed track, the id of
  the same song on each additional server, so a playlist can be translated to any
  server. The default server needs no mapping - its ids are the canonical ones.

## Configuring servers

Open the **Setup** page (admin only). Under **Additional Music Servers** you can:

- See every configured server, which one is the default, and whether it is enabled.
- Add a server: pick a type, fill its credentials and (optionally) a library
  filter, test the connection, and save.
- Edit, enable/disable, set-as-default, delete, or trigger a matching sweep.

Secrets (tokens, passwords) are never sent back to the browser; leave a secret
field blank when editing to keep the stored value.

## How analysis matches additional servers

Analysis always runs against the **default** server. When it finishes, a
background **matching sweep** pulls each additional server's catalogue and pairs
it to the analyzed library using the same tiered matcher as provider migration:

1. MusicBrainz id (when both sides expose one)
2. Normalised file path
3. Path tail (last path components)
4. Exact metadata (title, artist, album)
5. Noise-word-normalised metadata

Confident pairs are written to `track_server_map`. A track that does not match on
an additional server is simply left unmapped - never guessed. You can re-run the
sweep for a single server from the Setup page at any time.

## Selecting a server at runtime

A **Music server** dropdown appears in the menu when more than one server is
configured. It is remembered in your browser and, when set to a non-default
server, is sent as an optional `server=<id>` parameter on API calls.

There is no `v2` API. Every existing endpoint gains one optional `server`
parameter (query string or JSON body). When it is absent, the request targets the
default server exactly as before. When it names another server, playlist creation
translates the selected track ids to that server and creates the playlist there,
reporting how many tracks were unavailable. Tracks that do not exist on the target
server are dropped rather than sent with the wrong id.

## Registry API

All under `/api/servers`. Listing is available to any authenticated user
(credentials masked); every mutation is admin-only.

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/api/servers` | List servers (masked) + default id + feature flag |
| POST | `/api/servers` | Add a server |
| PUT | `/api/servers/<id>` | Update a server (blank secret keeps current) |
| DELETE | `/api/servers/<id>` | Delete a server (not the default) |
| POST | `/api/servers/<id>/default` | Make this server the default |
| POST | `/api/servers/test` | Test a connection (before saving) |
| POST | `/api/servers/<id>/sweep` | Re-run the matching sweep for one server |

## Content fingerprint as the catalogue id (advanced)

By default a track's canonical `item_id` is the default server's id, and a
content fingerprint is computed alongside it (`Audio -> Chromaprint -> 64-bit
SimHash -> BIGINT`, exact lookup) as the cross-server match key. Real Chromaprint
is used when its library is present, otherwise a librosa chroma fingerprint with
the identical pipeline - so every build can fingerprint, no network calls.

Set `CATALOG_FINGERPRINT_AS_ID=true` to go one step further and make the
**fingerprint itself the catalogue id**. After analysis the fingerprinted rows
are relabelled from the media-server id to the canonical fingerprint id, the six
similarity indexes are rebuilt, and the default server's real ids are preserved
in `track_server_map` and translated back whenever a list or playlist is sent to
a server. The result is a server-independent, content-deduplicated catalogue.

Because this rewrites the primary key and rebuilds every index it is opt-in:

- It runs automatically at the end of each analysis when the flag is on, and can
  be triggered on demand with `POST /api/servers/canonicalize` (admin).
- `CATALOG_FINGERPRINT_BACKFILL_PER_RUN=<n>` fingerprints up to `n` legacy rows
  (that predate fingerprinting) per analysis run by re-downloading them.
- `MULTISERVER_SWEEP_FINGERPRINT=true` makes the cross-server sweep download and
  fingerprint each secondary server's tracks so matching is exact-by-content
  (expensive - one download per track - so off by default; matching otherwise
  uses MusicBrainz id / path / metadata tiers).

## Limitations to know

- The default server is treated as the superset library. Music that exists **only**
  on an additional server is not analyzed or searchable; only tracks that match a
  canonical (default-server) track become playable on that additional server.
- Search and similarity results always come from the analyzed (default) library;
  the `server` selection changes only where playlists are created.
- Set `MULTI_SERVER_ENABLED=false` to hide the feature entirely and run as a
  strict single-server install.
