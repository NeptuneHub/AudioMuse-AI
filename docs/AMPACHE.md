# AudioMuse-AI with Ampache

This guide shows how to deploy AudioMuse-AI next to [Ampache](https://ampache.org)
and how to enable the Ampache plugin so that sonic similarity shows up inside
Ampache itself. Once it works you get similar-song results in the Ampache web UI
and, through the Subsonic API, in compatible clients such as Symfonium,
Substreamer and Tempo.

Two projects are involved:

- Core app: https://github.com/NeptuneHub/AudioMuse-AI
- Ampache, which ships the AudioMuse plugin in its own tree: https://github.com/ampache/ampache

## Which connector to use

AudioMuse-AI can talk to Ampache two ways, and they are not equivalent:

- **`ampache`** — the native connector, using Ampache's own JSON API. This is the
  one to pick. It reads catalogs, playlists and play statistics the way Ampache
  models them, so catalog filtering and playlist creation behave the way they do
  in the Ampache UI.
- **`navidrome`** — the OpenSubsonic connector. Ampache serves that API too, so
  this works, but everything is seen through the Subsonic model: ids are the
  prefixed Subsonic form (`so-123`), and Ampache-specific concepts are not
  visible.

Pick one and stay on it. Track ids differ between the two, so switching after an
analysis means the stored results no longer line up with what the server reports.

## What you need

- Ampache **8.0.0 or later** for the full feature set. The connector works
  against Ampache 7, but see *Server version* below for what is missing there.
- Docker and Docker Compose, or one of the native builds.
- Ampache and AudioMuse-AI able to reach each other over the network.

## Step 1 - Deploy AudioMuse-AI

AudioMuse-AI runs as a small stack: the Flask app, a worker, PostgreSQL and
Redis. Follow Step 1 of [the Navidrome guide](NAVIDROME.md), which is identical
here — only the media server chosen in the Setup Wizard differs.

In the Setup Wizard pick **Ampache** as the media server and enter:

- **URL** — the base address of your Ampache server, for example
  `http://192.168.1.50`. Do not include `/server/json.server.php`; the connector
  appends its own paths.
- **User** — an Ampache username.
- **Password** — either that user's password **or** their API key. The connector
  tries the API key form first and falls back to password authentication, so
  whichever you have works without any extra setting.

The equivalent environment variables, if you configure the container directly
rather than through the wizard, are `AMPACHE_URL`, `AMPACHE_USER` and
`AMPACHE_PASSWORD`.

**Run a first analysis before anything else.** AudioMuse-AI can only find similar
songs after it has analysed them. Start an analysis from the main page and let it
finish, then confirm it works with **Similar Song** on any track in the
AudioMuse-AI UI.

The connector downloads each track with Ampache's `download` action rather than
`stream`, so analysis always sees the original file instead of a transcode.

## Step 2 - Enable the AudioMuse plugin in Ampache

The plugin ships with Ampache; there is nothing to download.

1. In Ampache go to **Admin > Modules > Plugins** and activate **AudioMuse**.
2. Open your account preferences and set the **AudioMuse server URL** to the
   address where Ampache reaches the core app, for example
   `http://192.168.1.50:8000`. Use a host and port the Ampache container can
   actually reach, not `localhost`, unless the two share a network namespace.

Ampache advertises the OpenSubsonic `sonicSimilarity` extension only while a
sonic-analysis plugin is enabled for the requesting user, so a server with the
plugin switched off reports the feature as unsupported rather than answering with
metadata similarity, which is a different thing and would give wrong results.

## Step 3 - Check that it works

- In AudioMuse-AI, use **Similar Song** on a track. It should return results.
- In Ampache, the same track should offer similar songs. Over the Subsonic API,
  `getSonicSimilarTracks` should return `sonicMatch` entries; over Ampache's own
  API 8, `sonic_match` (REST: `songs/{song_id}/sonic-match`) returns the same
  matches with a `similarity` score.

Both APIs use the same scale: `0.0`-`1.0` where `1.0` is the same recording, and
`-1` when the backend gives no comparable score for that row.

## Server version

The connector degrades cleanly on an older Ampache, but two things need
**Ampache 8**:

- **Last played time** (`get_last_played_time`) needs database version `800029`
  or later, which added the maintained `last_played` column. Earlier servers omit
  the field and the connector returns `None`, so anything ranking by recency
  simply has nothing to sort on. The value is server-wide rather than per-user,
  matching how Ampache scopes `playcount` — `starred` and `userRating` are its
  per-user pair. An upgraded database backfills the column from the play history
  it already has, so it is populated for old plays too, not just new ones.
- **The `sonic_match` API method** is API 8 only. A client pinned to an older API
  version by the `api_force_version` preference gets `4705 Invalid Request` for
  it, which is the usual first cause when the endpoint works in one client and
  not another.

## Troubleshooting

- **`4701 Session Expired`**: the handshake token has lapsed. The connector
  re-handshakes automatically on that error; seeing it repeatedly means the
  credentials are wrong rather than stale.
- **`4705 Invalid Request` from `sonic-match`**: the session is on an older API
  version. Check the user's `api_force_version` preference and the `version`
  parameter the client sends.
- **Empty or poor results**: the library is not analysed yet, or not fully. Run
  the analysis in AudioMuse-AI and wait for it to finish.
- **Analysis finds nothing to fetch**: check the catalog filter. The connector
  only walks the catalogs it is pointed at, so a library in an unselected catalog
  is invisible to it.
- **Results do not line up with the server**: the analysis was probably run
  through the other connector. Ampache ids and Subsonic ids are different, so
  re-run the analysis after switching.
