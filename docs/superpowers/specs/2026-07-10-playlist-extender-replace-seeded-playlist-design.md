# Playlist Extender: Replace the Seeded Playlist

## Status

Approved design. The selected implementation is the compact, name-based replacement approach that reuses the existing `create_or_replace_playlist` media-server path.

## Problem

Playlist Extender can seed its Workbench from an existing media-server playlist, but the Workbench can only be saved as a newly named playlist. A user who intentionally seeded an existing playlist should also be able to replace that playlist with the current Workbench.

Replacement must remain an explicit option. Creating a new playlist stays available and is the non-destructive default.

## Goals

- Keep the existing create-new playlist flow.
- Offer replacement only after the user seeds from a media-server playlist.
- Replace the seeded playlist with the deduplicated Workbench contents in Workbench order.
- Reuse the existing, tested name-based `create_or_replace_playlist` path used by Sonic Fingerprint automation.
- Make the destructive action clear and require confirmation.
- Keep the Workbench intact when replacement fails.
- Support both the desktop Workbench rail and mobile Workbench sheet.

## Non-goals

- Selecting an arbitrary existing playlist at save time.
- Replacing an AudioMuse cluster playlist.
- Replacing a playlist when the current seed is only the Workbench.
- Adding a new exact-ID replacement API to every media-server adapter.
- Changing the existing Sonic Fingerprint behavior.

## Chosen Approach

The frontend remembers the selected media-server playlist name while it loads that playlist into the Workbench. The Workbench continues to expose its existing name field and create-new button. It additionally shows a contextual replacement button for the remembered seed.

The backend accepts a replacement playlist name as an alternative to a new playlist name. It validates that the named playlist currently exists, deduplicates the submitted track IDs while preserving order, and calls `tasks.mediaserver.create_or_replace_playlist`.

This approach intentionally resolves the replacement target by name. If a media server permits duplicate playlist names, the provider's existing name lookup decides which match is replaced. That trade-off was accepted in favor of the smaller change and reuse of the established replacement path.

Some providers implement replacement by deleting and recreating a playlist. In those providers, the playlist name is preserved but its internal server ID may change.

## User Experience

### Create New

The current playlist-name input and create-new action remain available and behave as they do today. Creating a new playlist never triggers replacement.

### Replace the Seed

When a media-server playlist is selected as the seed:

1. The extender records its name and the count of provider tracks that could not be loaded into the Workbench.
2. The analyzed tracks are added to the Workbench as today.
3. A secondary action appears in both Workbench layouts: `Replace “<playlist name>”`.
4. Clicking it opens a native confirmation that includes the target name and current Workbench track count.
5. If the seed contained unanalyzed tracks, the confirmation also states that those tracks are absent from the Workbench and will be removed by replacement.
6. Confirmation sends the current Workbench IDs to the replacement API.

The replacement action is hidden when no media-server seed is active, when an AudioMuse playlist is selected, or after the Workbench is cleared. Selecting a different media-server seed changes the target to the newly selected playlist.

On success, the UI shows a success toast and clears the Workbench, matching the existing create-new flow. On cancellation or failure, the Workbench and target remain intact.

## Frontend State and Data Flow

`curator-shared.js` owns an in-memory `seededServerPlaylist` object containing:

- `playlistId`, for frontend identity and diagnostics;
- `playlistName`, used by the replacement request;
- `unresolvedTracks`, used in the confirmation warning.

`curator-shared.js` exposes `window.curatorSetSeededPlaylistTarget(target)`. The extender calls that setter before the seed dropdown switches back to Workbench, and the shared Workbench renderer updates both replacement controls. Clearing the Workbench calls the same setter with `null`.

The state is intentionally not persisted across a page reload, which prevents a stale replacement target from surviving after the source playlist may have changed. A user can reselect the seed after a reload.

The replacement payload is:

```json
{
  "replace_playlist_name": "Road Trip",
  "track_ids": ["track-1", "track-2"]
}
```

The create-new payload remains:

```json
{
  "new_playlist_name": "Road Trip Extended",
  "track_ids": ["track-1", "track-2"]
}
```

## Backend Behavior

`POST /api/curator/save_playlist` supports exactly one action per request:

- `new_playlist_name`: create a new playlist through `create_playlist_from_ids` and return HTTP 201;
- `replace_playlist_name`: replace the existing named playlist through `create_or_replace_playlist` and return HTTP 200.

The route rejects requests that provide neither action or both actions. Names must be non-empty strings after trimming, and track IDs must be a non-empty list. Track IDs are converted to strings and deduplicated without changing their first-occurrence order.

Before replacement, the route fetches the current media-server playlist list and requires an exact name match. A missing target returns HTTP 404 rather than silently creating a new playlist. A provider that raises `NotImplementedError` for replacement returns HTTP 501 instead of falling back to create-new in the route.

Successful responses include the action, playlist name, resulting playlist ID, and total song count. A falsey provider result is treated as a failed replacement. Unexpected failures use the existing generic internal-error response and server-side logging.

## Error Handling and Safety

- Replacement requires an explicit confirmation in the browser.
- The confirmation makes the replacement track count visible.
- Unanalyzed source tracks produce an additional removal warning.
- A stale or missing playlist name returns 404.
- Empty Workbenches and invalid payloads return 400.
- Failed saves do not clear the Workbench.
- The create-new and replace code paths remain mutually exclusive.

## Testing

Backend tests will cover:

- the existing create-new behavior and HTTP 201 response;
- replacement calling `create_or_replace_playlist` with the chosen name and ordered, deduplicated IDs;
- replacement returning HTTP 200 and replacement metadata;
- rejection of missing, conflicting, non-string, empty, and nonexistent targets;
- failure behavior without accidental fallback to playlist creation.

Frontend/template tests will cover:

- both desktop and mobile replacement controls;
- target capture before the extender switches back to Workbench;
- replacement controls hidden without a media-server seed;
- create-new remaining available;
- confirmation copy, including the unanalyzed-track warning;
- replacement payload shape;
- clearing only after a successful response.

Browser verification on the isolated test instance will confirm the contextual controls and non-destructive create-new path. Provider replacement itself will be exercised through mocked backend tests so verification does not overwrite an external playlist unintentionally.

## Deployment

After tests and CI pass, rebuild and recreate only the isolated `playlist-curator-test` web container. The test project continues to have no worker and no GPU request. Re-audit the production worker against its existing baseline after deployment.
