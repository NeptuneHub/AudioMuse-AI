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
   Default server ---- current config and unchanged API defaults

Optional ?server=<id> (menu dropdown / API param)
        |
        v
   Server registry -> bound provider client -> that server's catalogue
```

- The **default server** is the one you configure in the Setup wizard. It is
  analyzed first and remains the target when an API omits `server`.
- **Additional servers** are stored in a registry (`music_servers` table). Each
  keeps its own credentials and library filter.
- A **mapping table** (`track_server_map`) records, per analyzed track, the real
  provider id on every server, including the default. Database `item_id` values
  are content ids, never Jellyfin/Navidrome/Plex ids after canonicalization.

## Configuring servers

Open the **Setup** page (admin only). Under **Music Servers** you can:

- See every configured server, which one is the default, and whether it is enabled.
- Add a server: pick a type, fill its credentials and (optionally) a library
  filter, test the connection, and save.
- Edit, enable/disable, set-as-default, delete, or trigger a matching sweep.

Secrets (tokens, passwords) are never sent back to the browser; leave a secret
field blank when editing to keep the stored value.

## How analysis matches additional servers

Analysis processes every enabled server sequentially, with the **default**
server first. After each source phase a matching sweep aligns the canonical
catalogue to the servers that have not had their own phase yet, and a final
alignment pass across all enabled servers runs once the last phase completes,
before the shared indexes are rebuilt. Tracks already mapped to an analyzed
canonical id are skipped, while tracks unique to the next server are analyzed
once and added to the union catalogue. Alignment uses these tiers:

1. Normalised file path
2. Path tail (last path components)
3. Exact metadata (title, artist, album)
4. Noise-word-normalised metadata

Confident pairs are written to `track_server_map`. A track that does not match on
an additional server is simply left unmapped - never guessed. You can re-run the
sweep for a single server from the Setup page at any time. Manual sweeps (and
the **Align music servers** action) re-fetch the server's full catalogue and
prune mappings whose track is no longer on, or is filtered out of, that server.
Pruning only happens when the fetch looks complete: if the catalogue returns
fewer tracks than half the mappings already stored, the fetch is treated as
partial and pruning is skipped, so a transient provider error never
mass-deletes valid mappings. Only map rows are ever removed, never analyzed
tracks.

Matching runs in bounded memory even on very large libraries: the fetched
catalogue is condensed into a slim lookup index and released, and the local
catalogue streams through it in chunks (20k tracks at a time) with matches
written after every chunk, so neither side is ever held fully in RAM and a
cancelled sweep keeps everything matched so far.

The library cleaning task is multi-server aware: it fetches the current track
set of every enabled server, translates each server's provider ids to canonical
catalogue ids, and deletes only tracks that no server still has. If any
server's catalogue cannot be fully fetched, the run aborts without deleting
anything.

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
When it names another server, the shared index filters candidates to tracks
available on it before distance ranking, and playlist creation translates the selected track
ids and creates the playlist there, reporting how many tracks were unavailable.
Tracks that do not exist on the target server are dropped rather than sent with
the wrong id.

Input ids follow one symmetric contract. Every seed or track id a caller sends
(similar-song seed, Alchemy song or playlist anchor, Path start/end, SemGrove
seed, Sonic Fingerprint listening history) is first resolved through the single
input resolver (`registry.canonical_input_ids`, request-side
`resolve_input_item_id(s)`): the selected server's provider id becomes the
canonical catalogue id before touching any shared index, while canonical or
unknown ids pass through unchanged. On output the same mapping runs in reverse,
so a client can round-trip its own server's ids end to end:

```
provider input id -> canonical id -> shared index (+ availability mask)
                  -> canonical results -> provider ids -> provider action
```

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

## Stable content id from Chromaprint

Every analyzed track is downloaded from its source server when its Chromaprint
is missing. The bundled `fpcalc` tool computes the Chromaprint; the complete,
unchanged value is stored in `score.chromaprint`. A SHA-256 digest of that value
becomes the stable `fp_<hex>` `score.item_id`. MusiCNN/CLAP model changes cannot
mint a new content id.

Legacy rows are backfilled by the next analysis. Relabelling then becomes a
database-only operation. The media server's real id is preserved in
`track_server_map` and translated back whenever a playlist is sent to a server.
If two provider rows produce the same Chromaprint id, their analysis rows are
merged and both provider mappings are retained.

Cross-server matching uses normalized path, path tail, and metadata tiers; in
practice these align 99%+ of a same-library pair instantly with zero downloads.
Tracks unavailable on the selected server are filtered from results and are
never sent to that provider as canonical ids.

## One shared index, bounded build memory

AudioMuse builds one index for the union catalogue, not one index per server.
The index abstraction maintains a small cached availability mask for the active
server and applies it before ranking candidates. Existing search, Path, Alchemy,
Map and similarity call sites continue using the same index API.

Index construction caps the clustering training sample (50,000 rows by default),
writes completed cells to PostgreSQL incrementally instead of retaining every
cell in RAM, and uses a temporary disk-backed matrix for SemGrove. The cap is
configurable with `IVF_TRAIN_SAMPLE_MAX`.

## Scheduled tasks

Analysis, Clustering, Sonic Fingerprint and Radio schedules have an **All music
servers** / **Default server only** selector. Analysis deduplicates work across
sources. Clustering is computed once and its playlists are translated per target
server. Sonic Fingerprint and Radio run inside each selected server context, so
their listening history and results remain valid for that server.

## Limitations to know

- Analysis and indexes cover the union of enabled server catalogues. Each song
  is stored once under its canonical id and may have mappings on one or many
  servers.
- Search and similarity results are scoped to tracks mapped on the selected
  server before provider ids are emitted.
