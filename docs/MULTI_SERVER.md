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

A **Music server** dropdown appears in the sidebar menu (under Logout) when more
than one server is configured. It is remembered in your browser and, when set to
a non-default server, is sent as an optional `server` parameter on API calls.

There is no `v2` API. Every existing endpoint gains one optional `server`
parameter (query string or JSON body) that accepts the server's configured
display NAME - the friendly, unique value from the setup wizard, e.g.
`?server=Office%20Jellyfin` - or the internal id. External callers (media-server
plugins, scripts) should use the name, or omit the parameter for the default.
When it is absent, the request targets the default server exactly as before.
When it names another server, similarity/search endpoints over-fetch and return
only tracks available on it, and playlist creation translates the selected track
ids and creates the playlist there, reporting how many tracks were unavailable.
Tracks that do not exist on the target server are dropped rather than sent with
the wrong id.

The matching sweep reports live progress: the setup wizard's Music Servers
section shows a progress bar and a one-line status while it runs, and the
dashboard gains a **Music Server Status** section (visible with more than one
server, refreshed hourly) with the matched-song count per server.

External integrations follow the same rule. ``GET /api/sync`` returns each
track under the selected server's REAL id (default server when no ``server``
param), accepts that server's ids in ``?ids=``, and reports that server's
``provider_type``; tracks the selected secondary server does not have are
omitted. ``/external/get_score``, ``/external/get_embedding`` and
``/external/search`` accept and return the selected server's ids the same way.
``/api/waveform`` and ``/api/sonic_fingerprint/generate`` also honor ``server``
(listening history, downloads and results come from that server).

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

## Content id from the MusiCNN embedding

The canonical catalogue id is derived from data every analyzed track already
has: its MusiCNN embedding. The embedding is projected onto 64 fixed, seeded
random hyperplanes and the sign of each projection becomes one bit of a 64-bit
SimHash-LSH value, encoded as the `fp_<hex>` item_id. Nothing is downloaded, no
external service is contacted, and no extra column is stored - the id itself
encodes the hash.

Because the id comes from the stored embedding, relabelling is a pure database
operation: a 180k-track legacy install canonicalizes in seconds. It happens
automatically at the end of every analysis and at the start of every server
sweep, so a legacy install is fixed either by its next analysis or the moment a
secondary server is added - whichever comes first. The media server's real id is
preserved in `track_server_map` (also for the default/single server) and
translated back whenever a playlist is sent to a server. Rows analyzed without
an embedding keep their provider id and keep working unchanged.

Cross-server matching uses normalized path, path tail, and metadata tiers; in
practice these align 99%+ of a same-library pair instantly with zero downloads.
Secondary tracks that do not match are simply left unmapped - the default
server's catalogue is never touched or reduced.

## Limitations to know

- The default server is treated as the superset library. Music that exists **only**
  on an additional server is not analyzed or searchable; only tracks that match a
  canonical (default-server) track become playable on that additional server.
- Search and similarity results always come from the analyzed (default) library;
  the `server` selection changes only where playlists are created.
- Set `MULTI_SERVER_ENABLED=false` to hide the feature entirely and run as a
  strict single-server install.
