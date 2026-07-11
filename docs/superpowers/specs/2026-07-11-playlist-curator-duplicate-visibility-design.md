# Playlist Curator: Visible Duplicate Scans in Workbench and Search Results

## Status

Approved design. Duplicate scans remain non-destructive and operate on an explicit, bounded set of tracks.

## Problem

The existing Workbench duplicate action reaches `POST /api/curator/find_duplicates`, but an empty response appears in a panel below the main results, does not scroll into view, and automatically hides after three seconds. From the Workbench rail this looks like the click did nothing.

Smart Search also has no direct duplicate-scan action in its results overview. Users can scan the Workbench, but cannot inspect duplicate groups among the currently loaded search matches before adding tracks.

## Goals

- Make Workbench duplicate scans visibly start and visibly finish on both curator pages.
- Keep empty and error states visible until the user closes or reruns the scan.
- Add a Search Results action that scans the currently loaded results.
- Cap Search Results scans at 500 tracks to bound the pairwise comparison cost.
- Highlight a recommended keeper and the tracks marked as duplicates.
- Allow duplicate tracks to be hidden from the current Search Results display without changing the library, a media-server playlist, or the Workbench.
- Preserve the current Workbench removal flow for Workbench scans.

## Non-goals

- Scanning every match in an arbitrarily large search query.
- Deleting tracks from the media server or AudioMuse database.
- Removing tracks from a media-server playlist.
- Running duplicate detection as a background worker job.
- Changing the backend similarity algorithm or ranking score.

## Chosen Approach

The shared duplicate finder accepts an explicit scan mode and track list. Workbench mode submits the Workbench tracks exactly as it does today. Search Results mode submits the currently loaded and visible search results, capped at 500 tracks.

Both modes reuse `POST /api/curator/find_duplicates`. The response panel is revealed and scrolled into view as soon as a scan starts. Loading, empty, success, and error states stay visible until the user closes the panel or starts another scan.

The duplicate panel records which mode produced its current groups. In Workbench mode, the existing action removes tracks marked for removal from the Workbench. In Search Results mode, the action is labeled `Hide marked duplicates` and only filters those IDs from the current result display.

## User Experience

### Workbench Scan

The desktop rail and mobile sheet retain their existing `Find duplicates` actions. Clicking either action:

1. Requires at least two Workbench tracks.
2. Opens the duplicate panel immediately.
3. Scrolls the main content to the panel and shows a scanning state.
4. Leaves `No duplicates found at this sensitivity level` visible when the response contains no groups.
5. Shows duplicate groups with one recommended keeper per group when matches exist.
6. Keeps the existing `Remove All Marked` behavior, which removes only marked tracks from the Workbench.

The panel no longer closes itself after an empty result.

### Search Results Scan

Smart Search adds a `Find duplicates` action to the Search Results header. The action is disabled until at least two results are loaded.

Clicking it scans the currently loaded result set. If more than 500 results are loaded, the first 500 in the current display order are scanned and the panel states that the scan was capped. This avoids unbounded quadratic work while keeping the action responsive for normal searches.

When duplicate groups are found, the highest-ranked track remains the default keeper. The panel action changes to `Hide marked duplicates`. Clicking it removes the marked IDs only from the current rendered Search Results. It does not remove tracks from the Workbench, persistent storage, a playlist, or the media library.

A new search, clearing the search, or reloading results resets the hidden-result set and any Search Results duplicate review state.

## Frontend State and Data Flow

`curator-shared.js` continues to own the duplicate panel and backend request. Its duplicate scan entry point receives:

- a mode: `workbench` or `search-results`;
- the track objects to scan;
- optional context describing whether the list was capped.

Workbench buttons call the shared entry point with `getWorkbench().tracks`. Smart Search exposes its currently loaded, visible result list through a page-specific hook and calls the same entry point in `search-results` mode.

The panel stores its active mode alongside `dedupGroups`. Its primary action dispatches by mode:

- `workbench`: remove marked IDs through `workbenchRemove`;
- `search-results`: call a Smart Search hook that hides marked IDs and rerenders the results.

The Search Results button enabled state is updated whenever search results are rendered. The Search Results action depends only on loaded search results, not Workbench size.

## Computational Bound

The backend builds a pairwise similarity matrix, so work and matrix size grow approximately with the square of the submitted track count. A 500-track cap limits the matrix to 250,000 comparisons and keeps request memory predictable. The design intentionally avoids an all-query or all-library scan, which could become expensive for thousands of tracks.

## Error Handling and Safety

- Fewer than two tracks produces an immediate visible message.
- Loading state appears before the request starts.
- Network and backend errors remain visible in the panel.
- Empty results remain visible and do not auto-dismiss.
- Search Results hiding is presentation-only and resets with the search lifecycle.
- Workbench removal still requires the user to review the marked groups before applying it.
- No duplicate action writes to the media server or database.

## Testing

Shared JavaScript behavior tests will cover:

- Workbench scan opens and scrolls the panel before awaiting the response;
- an empty scan stays visible and is not auto-hidden;
- successful Workbench scans retain removal behavior;
- Search Results mode submits the provided result IDs with a maximum of 500;
- Search Results mode changes the primary action to `Hide marked duplicates`;
- applying Search Results duplicates calls the page hook and does not mutate the Workbench;
- errors remain visible.

Smart Search tests will cover:

- the overview button is present and enabled only with at least two loaded results;
- the scan uses the current display order;
- hidden duplicate IDs are filtered from the rendered results;
- running or clearing a search resets hidden duplicate IDs.

Backend characterization tests will continue to cover duplicate grouping and keeper ranking. No backend contract change is required.

## Deployment and Verification

After focused and full checks pass, push the branch and wait for GitHub Actions and SonarCloud. Rebuild the CPU-only image and recreate the isolated `playlist-curator-test` web and worker services only. Verify both duplicate entry points against the cloned test database and re-audit that the production worker identity and start time are unchanged.
