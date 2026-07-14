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
- **Every server lives in the registry** (`music_servers` table), the default
  included - it is the ONLY persistent home of media-server settings. Each row
  keeps its own credentials and library filter. Legacy installs are migrated at
  first start: any media-server keys still in `app_config` are moved into the
  registry and deleted; environment variables only matter as first-boot seed
  values. The classic `config` values remain readable but are a read-only
  projection of the registry's default row.
- A **mapping table** (`track_server_map`) records, per analyzed track, the real
  provider id on every server, including the default. Database `item_id` values
  are content ids, never Jellyfin/Navidrome/Plex ids after canonicalization.

## Configuring servers

Open the **Setup** page. It is admin-only once the install is set up; during
the very first run - when no admin account exists yet - the wizard reaches it
unauthenticated, so a fresh install can add, test and pick its media servers
right there. Under **Music Servers** you can:

- See every configured server and which one is the default. On a fresh install
  the first server you add automatically becomes the default (the empty seed
  entry it replaces is removed).
- Add a server: pick a type, fill its credentials and (optionally) a library
  filter, test the connection, and save.
- Edit, set-as-default, delete, or trigger a matching sweep. Every
  configured server is always active; there is no enable/disable state.

Secrets (tokens, passwords) are never sent back to the browser; leave a secret
field blank when editing to keep the stored value.

## How analysis keeps servers aligned (no sweeps)

Analysis processes every configured server sequentially, with the **default**
server first, and NEVER runs an alignment sweep: every track resolves against
the shared catalogue at analyze time.

Per server, per album: a track whose provider id already has a mapping is
skipped outright. An unmapped track is downloaded and run through MusiCNN; its
embedding hash id is then checked against the catalogue (Hamming-tolerant). If
the content is already there - analyzed earlier from another server - the track
just gains a `track_server_map` row for this server and everything else is
skipped. Only genuinely new content runs the full pipeline (CLAP, lyrics) and
is stored once under its hash id. The indexes are rebuilt once at the end of
the run. Servers therefore stay aligned by construction.

## The setup-time Align (easy match, zero downloads)

The **Align music servers** action (and the per-server Sweep) exists for the
initial setup moment: it instantly maps a server's catalogue onto tracks the
database already holds, using pure metadata matching - no downloads, no
analysis, no id calculation. Matching uses these tiers:

1. Normalised file path
2. Path tail (last path components)
3. Exact metadata (title, artist, album)
4. Noise-word-normalised metadata

Confident pairs are written to `track_server_map`. The sweep also aligns
everything else the server needs: its artist links (`artist_server_map`) are
upserted from the fetched catalogue, and the catalogue's album, album artist,
year and rating are batch-refreshed for every track mapped to that server (the
file path only from the default server). A track that does not match on
an additional server is simply left unmapped - never guessed. You can re-run the
sweep for a single server from the Setup page at any time. Manual sweeps (and
the **Align music servers** action) re-fetch the server's full catalogue and
prune mappings whose track is no longer on, or is filtered out of, that server.
Pruning only happens when the fetch looks complete: if the catalogue returns
fewer tracks than half the mappings already stored, the fetch is treated as
partial and pruning is skipped, so a transient provider error never
mass-deletes valid mappings. Only map rows are ever removed, never analyzed
tracks. A server's library filter is honoured by every provider: Jellyfin and
Emby fetch only the selected libraries, Plex only the selected sections,
Navidrome only the selected music folders, and Lyrion only the selected paths -
so nothing outside the libraries you picked is ever mapped, counted or pruned.

Matching runs in bounded memory even on very large libraries: the fetched
catalogue is condensed into a slim lookup index and released, and the local
catalogue streams through it in chunks (20k tracks at a time) with matches
written after every chunk, so neither side is ever held fully in RAM and a
cancelled sweep keeps everything matched so far.

Adding or editing servers back to back never leaves a stale alignment running:
each save cancels any queued or running sweep (matches found so far are kept)
and enqueues one fresh alignment covering every server, so the newest
sweep always reflects the full server list. Plex servers can be linked without
hunting for a token: the add/edit form offers the same sign-in-with-Plex
(plex.tv/link PIN) flow as the setup wizard and fills the token automatically.

The library cleaning task is multi-server aware and never shrinks the catalogue:
it fetches the current track set of every server with the same full-catalogue
enumeration the sweeps use, and removes only that server's stale mapping rows
(see "Cleaning never shrinks the catalogue" below).

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
``/api/sonic_fingerprint/generate`` also honors ``server`` (listening history,
downloads and results come from that server).

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

The canonical `score.item_id` IS the content signature (`tasks/simhash.py`, a
home-made similarity hash): one bit per embedding dimension - "is this
dimension above the song's own average" - giving a 200-bit code encoded as the
scheme-versioned `fp_2<50hex>`. No random projections, no external binary, no
extra column, no metadata: the id is the shape of the song's MusiCNN profile.

The signature is similarity-preserving and only PROPOSES identity. The same
song analyzed from two servers (different file, encoder, bitrate) lands within
a few bits, so candidates are found with a Hamming-tolerant lookup - and the
final same/different decision is the EXACT cosine distance between the raw
embeddings, using the same `DUPLICATE_DISTANCE_THRESHOLD_COSINE` the Similar
Songs duplicate filter has always trusted. Everything deciding identity is
derived from the audio itself. Copies of one song become ONE `score` row, and
every provider file that carries it - across servers OR duplicated on a single
server - is recorded as its own `track_server_map` row; two similar-sounding
DIFFERENT songs never merge, even on a signature collision (the second simply
gets the next free id).

Legacy installs migrate ONCE, at Flask container startup, directly on the
Flask container (never through the job queue): item_ids are relabelled from
the already-stored embeddings - a pure database operation, computed vectorized
in chunks across a small bounded thread pool, with a single write pass per table.
The similarity indexes are REPOINTED at the new ids inside the same transaction as
the relabel: no vectors move, nothing is re-clustered, and similarity keeps working
across the migration, so no rebuild is required. Only an index that cannot be
repointed is left for the next analysis to rebuild, as before. Every later boot is
an instant no-op. The media server's real id is preserved in `track_server_map` and
translated back whenever a playlist is sent to a server.

Tracks unavailable on the selected server are filtered from results and are
never sent to that provider as canonical ids.

## Cleaning never shrinks the catalogue

The cleaning task fetches each server's current tracks with the same
full-catalogue enumeration the alignment sweeps use (library filter applied)
and removes
ONLY that server's stale `track_server_map` rows. Analysis rows, embeddings
and other servers' mappings are NEVER deleted: a song that disappeared from
one server keeps playing from the others, and a song on no server at all stays
in the catalogue as unbound (hidden by the per-server availability filter, and
re-bound automatically if it ever comes back). A server whose library cannot
be fully read is skipped, so a partial view can never unbind valid mappings.

## One shared index, bounded build memory

AudioMuse builds one index for the union catalogue, not one index per server.
The index abstraction maintains a small cached availability mask for the active
server and applies it before ranking candidates. Existing search, Path, Alchemy,
Map and similarity call sites continue using the same index API.

Index construction trains k-means on a sample that scales with the library
(50 vectors per cell), streamed through the trainer in small batches so build
RAM stays flat; completed cells are written to PostgreSQL incrementally instead
of being retained in RAM, and SemGrove merges through a temporary disk-backed
matrix. There is no training-sample cap: quality scales with the library, and
hardware sizing for very large libraries is the operator's call.

## Batch tasks always cover every server

**Analysis, Clustering, Cleaning and every scheduled task always run against ALL
your music servers.** There is no scope selector: it does not matter whether you
start them from the page or from a schedule.

They run **one server at a time**, in sequence. A server is finished before the
next one starts: for Clustering that means the whole pipeline (its own
availability-scoped input set, its own search, its own AI naming, its playlists
created on that server) completes before the next server begins. Results are never
computed once and pushed to other servers - each server's playlists are built from
the tracks that server actually has. Analysis deduplicates work across sources, so
a song already analyzed on one server is not analyzed again on the next; it is just
mapped. One server failing is isolated and reported, and the run moves on.

This used to be a per-schedule **All music servers** / **Default server only**
choice, and Clustering from the UI quietly targeted only the server picked in the
dropdown. Both are gone: a narrowed scope silently left the other servers without
playlists, and a "Default server only" Analysis left every other server's exclusive
songs permanently unanalyzed and invisible to every other feature.

## Interactive features follow the server picker

Similar Song, Instant Playlist, Song Path, Song Alchemy, Artist Similarity, Sonic
Fingerprint and the search pages **do** target the one server selected in the
sidebar dropdown, and their results are filtered to tracks available on it. The
Sonic Fingerprint page asks for the credentials of the server you selected.

Every page shows which of the two it is: a **CATALOGUE** or **PER SERVER** tag next
to its title.

## Limitations to know

- Analysis and indexes cover the union of the configured server catalogues. Each song
  is stored once under its canonical id and may have mappings on one or many
  servers.
- Search and similarity results are scoped to tracks mapped on the selected
  server before provider ids are emitted.
