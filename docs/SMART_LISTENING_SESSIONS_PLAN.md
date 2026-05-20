# Smart Listening Sessions - 5 Day Implementation Plan

Smart Listening Sessions is a prompt-driven playlist builder. A user describes a listening intent, optionally adds one or more seed songs or artists, chooses a length and curve, previews the generated queue, reviews why each song was selected, and exports the result to the configured media server.

The feature should reuse AudioMuse-AI's existing analysis layers instead of introducing a new recommendation stack. The first implementation is an MVP focused on a reliable preview and export workflow.

## Goals

- Let users create a playlist from natural language, such as `slow-building evening playlist that starts acoustic and becomes more electronic`.
- Combine prompt similarity, seed-song similarity, mood/path shaping, and diversity rules into one ranked queue.
- Explain each selected track with short, auditable reasons based on available scores and similarities.
- Export the selected queue through the existing media-server playlist path.
- Keep the feature usable when optional subsystems are unavailable, with clear capability warnings.

## Non-goals For The First 5 Days

- Do not train a new recommendation model.
- Do not require an external LLM for the MVP.
- Do not rewrite existing CLAP, SemGrove, Voyager, path, or media-server modules.
- Do not implement collaborative multi-user playlist editing.
- Do not add destructive playlist replacement by default. Export should create a new playlist unless the user explicitly chooses an existing supported replace path in a later iteration.

## Existing Building Blocks

- `app_clap_search.py` exposes `/api/clap/search` for natural-language audio similarity.
- `app_sem_grove.py` exposes `/api/sem_grove/search` for merged lyrics and audio similarity around a seed song.
- `app_path.py` exposes `/api/find_path` and uses `tasks.path_manager`, `tasks.voyager_manager`, and mood centroids for smooth song transitions.
- `app_voyager.py` exposes `/api/create_playlist`, which validates IDs and calls `tasks.voyager_manager.create_playlist_from_ids`.
- `tasks.mediaserver` provides provider-specific playlist creation and instant playlist helpers.
- `templates/includes/layout.html`, `static/style.css`, and `static/menu.css` provide the shared UI shell and theme variables.

## Proposed Files

- `app_smart_sessions.py`: new Flask blueprint for the page and APIs.
- `tasks/smart_session_builder.py`: request validation, candidate retrieval, ranking, curve shaping, explanations, and export helpers.
- `templates/smart_sessions.html`: interactive page for prompt, controls, preview, explanations, and export.
- `static/smart_sessions.js`: client behavior for preview, selection edits, reorder/remove, and export.
- `tests/unit/test_smart_session_builder.py`: unit tests for validation, ranking, deduplication, diversity, and explanation behavior.
- `tests/unit/test_app_smart_sessions.py`: Flask endpoint tests for success and failure paths.

## API Contract

### `GET /smart_sessions`

Render the Smart Listening Sessions page.

### `GET /api/smart_sessions/capabilities`

Return which sources are available and any limits the UI should show.

Example response:

```json
{
  "clap_enabled": true,
  "clap_cache_loaded": true,
  "sem_grove_available": true,
  "lyrics_enabled": true,
  "max_length": 100,
  "default_length": 25
}
```

### `POST /api/smart_sessions/preview`

Generate a preview queue without creating a media-server playlist.

Request shape:

```json
{
  "prompt": "slow-building evening playlist, warm and textured",
  "length": 25,
  "curve": "calm_to_intense",
  "anchors": [
    { "type": "song", "item_id": "abc123", "weight": 0.8 }
  ],
  "avoid": {
    "artists": ["Example Artist"],
    "terms": ["live", "remix"]
  },
  "max_per_artist": 2,
  "include_explanations": true
}
```

Response shape:

```json
{
  "session_id": "optional-preview-token-or-null",
  "playlist_name": "Smart Session - Evening Warmth",
  "tracks": [
    {
      "item_id": "abc123",
      "title": "Track Title",
      "author": "Artist Name",
      "position": 1,
      "scores": {
        "intent": 0.84,
        "anchor": 0.73,
        "curve": 0.66,
        "diversity_penalty": 0.0,
        "final": 0.78
      },
      "reason": "Matches the warm evening prompt and starts close to the selected seed song."
    }
  ],
  "warnings": []
}
```

### `POST /api/smart_sessions/export`

Create a media-server playlist from an approved queue. This endpoint should either delegate to existing playlist creation logic or share the same helper path as `/api/create_playlist`.

Request shape:

```json
{
  "playlist_name": "Smart Session - Evening Warmth",
  "track_ids": ["abc123", "def456"]
}
```

## Ranking Algorithm MVP

The MVP should be deterministic enough to test, but flexible enough to improve later.

1. Validate request values.
   - Require prompt or at least one anchor.
   - Clamp `length` to a configured range, for example 5 to 100.
   - Clamp `max_per_artist` to 1 to 10.

2. Build a candidate pool.
   - If CLAP is enabled and loaded, call `tasks.clap_text_search.search_by_text(prompt, limit=pool_size)`.
   - For song anchors, call `tasks.sem_grove_manager.search_by_song(item_id, limit=anchor_pool_size)` when available.
   - Fall back to Voyager nearest-neighbor lookup from `tasks.voyager_manager` when SemGrove is unavailable.
   - Deduplicate by `item_id` while preserving best source scores.

3. Apply hard filters.
   - Remove avoided artists by normalized artist name.
   - Remove avoided title terms by normalized title string.
   - Remove tracks missing required metadata or media-server IDs.
   - Keep filtered-out counts for UI warnings.

4. Score candidates.
   - `intent_score`: normalized CLAP similarity, or neutral score if no CLAP source is available.
   - `anchor_score`: best seed-song similarity, weighted by anchor weight.
   - `curve_score`: placement suitability based on target position and chosen curve.
   - `freshness_score`: optional small boost for tracks not already selected in recent preview sessions, if this metadata is easy to access.
   - `diversity_penalty`: artist repetition, album repetition, and near-duplicate title penalty.

5. Order the queue.
   - Select the first track from high intent and low-energy/calm candidates when a calm-start curve is chosen.
   - Use greedy constrained reranking for each position: choose the highest final score that does not violate `max_per_artist`.
   - For curve-based sessions, gradually change target mood/energy over positions.
   - Keep ordering stable for identical scores by title and item ID so tests are deterministic.

6. Generate explanations.
   - Use rule-based explanation templates for the MVP.
   - Mention no more than two reasons per track.
   - Avoid claiming unavailable evidence. For example, do not mention lyrics if lyrics or SemGrove data was not used.

## 5 Day Plan

### Day 1 - Product Scope, Data Contract, And Backend Skeleton

Objective: create the feature skeleton and lock the API shape before algorithm work begins.

Tasks:

- Add `tasks/smart_session_builder.py` with request dataclasses or typed dictionaries for preview input, candidate rows, ranked tracks, warnings, and export payloads.
- Add validation helpers:
  - `validate_preview_request(data)`
  - `normalize_avoid_rules(data)`
  - `clamp_session_length(value)`
  - `normalize_anchor(anchor)`
- Add `app_smart_sessions.py` blueprint with routes:
  - `GET /smart_sessions`
  - `GET /api/smart_sessions/capabilities`
  - `POST /api/smart_sessions/preview`
  - `POST /api/smart_sessions/export`
- Register the blueprint in `app.py` near the other feature blueprints.
- Add the navigation entry to `templates/sidebar_navi.html` using the existing `active` pattern.
- Add placeholder `templates/smart_sessions.html` using the shared layout.
- Add placeholder `static/smart_sessions.js` loaded from the page template.
- Return deterministic placeholder previews only behind unit tests or behind a clearly internal helper. The public preview endpoint should call the real builder function even if the builder initially returns an empty result with warnings.

Implementation notes:

- Keep API functions thin. Put business logic in `tasks/smart_session_builder.py` so tests do not need a full Flask request context.
- Avoid introducing an RQ job for preview generation unless performance proves it necessary. Preview should be synchronous with a result cap.
- Export can initially reuse `tasks.voyager_manager.create_playlist_from_ids` or call the existing validation pattern from `app_voyager.py`.

Day 1 deliverables:

- New page reachable at `/smart_sessions`.
- Capability endpoint returns real flags for CLAP, lyrics, SemGrove, and configured limits.
- Preview endpoint validates input and returns structured errors.
- Unit tests cover request validation and API error shapes.

Definition of done:

- Flask starts with the new blueprint registered.
- `POST /api/smart_sessions/preview` rejects empty prompt plus no anchors.
- UI page renders without JavaScript errors.

### Day 2 - Candidate Retrieval And Source Fusion

Objective: produce a useful candidate pool from the existing search systems.

Tasks:

- Implement `get_prompt_candidates(prompt, limit)` using `tasks.clap_text_search.search_by_text` when CLAP is enabled and loaded.
- Implement `get_anchor_candidates(anchors, limit)` using `tasks.sem_grove_manager.search_by_song` for song anchors.
- Add fallback anchor retrieval with Voyager helpers where possible:
  - Load seed vector with `tasks.voyager_manager.get_vector_by_id`.
  - Find neighbors with `tasks.voyager_manager.find_nearest_neighbors_by_vector`.
- Implement `merge_candidate_sources(prompt_candidates, anchor_candidates)`.
- Normalize source scores into common 0.0 to 1.0 fields.
- Preserve source metadata so explanations can say whether a track came from prompt match, anchor match, or both.
- Add hard filters for avoided artists and avoided title terms.
- Return warnings when a source is disabled, empty, or not loaded.

Implementation notes:

- Use larger candidate pools than the requested playlist length, for example `max(length * 8, 100)`, capped to 500.
- Treat missing optional systems as degraded mode, not fatal, as long as at least one source returns candidates.
- Use normalized strings for filtering: lowercase, trim spaces, collapse repeated whitespace.
- Do not expose raw internal exceptions in API responses. Log them and return a user-safe warning.

Day 2 deliverables:

- Preview endpoint returns real track candidates from CLAP and/or anchors.
- Deduplication by `item_id` works across multiple sources.
- Avoid artist and avoid term rules work before ranking.
- Tests mock CLAP and SemGrove helpers to verify merge and filter behavior.

Definition of done:

- A prompt-only request returns up to the requested number of tracks when CLAP cache is available.
- A seed-song request returns candidates when SemGrove or Voyager data is available.
- Warnings accurately explain missing CLAP or SemGrove support.

### Day 3 - Ranking, Curves, Diversity, And Explanations

Objective: turn candidate lists into a polished listening sequence.

Tasks:

- Implement `rank_session_tracks(request, candidates)`.
- Add supported curves:
  - `steady`: preserve strongest overall matches.
  - `calm_to_intense`: gradually increase energy or excitement proxies.
  - `intense_to_calm`: gradually decrease energy or excitement proxies.
  - `near_anchor_then_explore`: start close to anchors and gradually prefer prompt match/diversity.
- Implement `score_candidate_for_position(candidate, position, total, curve)`.
- Use available score-table fields or mood centroid distances where practical. If no suitable feature is available, use source scores and avoid pretending there is an energy model.
- Implement constrained greedy ordering with `max_per_artist`.
- Add duplicate guards for same title and same artist if a title appears multiple times.
- Generate short explanations with rule-based templates.
- Add track-level score breakdowns for UI debugging and future tuning.

Implementation notes:

- Keep weights in module constants first:
  - `INTENT_WEIGHT = 0.45`
  - `ANCHOR_WEIGHT = 0.30`
  - `CURVE_WEIGHT = 0.20`
  - `DIVERSITY_WEIGHT = 0.05`
- Store enough internal details in tests, but keep API responses compact.
- Make ties deterministic. This matters for repeatable tests and user trust.
- Avoid overfitting the MVP to one media server.

Day 3 deliverables:

- Preview returns an ordered queue with score breakdowns and reasons.
- Curves produce visibly different ordering when test fixtures contain feature differences.
- Diversity rules prevent one artist from dominating the session.
- Tests cover scoring, tie-breaking, max-per-artist behavior, and explanations.

Definition of done:

- A 25-song request returns 25 unique item IDs when enough candidates exist.
- Explanations only mention sources that contributed to the score.
- Ranking output is deterministic for identical input fixtures.

### Day 4 - Full UI Preview And Playlist Export

Objective: make the feature usable from the browser from prompt to media-server playlist.

Tasks:

- Expand `templates/smart_sessions.html` with:
  - Prompt textarea.
  - Session length input.
  - Curve selector.
  - Max-per-artist input.
  - Avoid artists and avoid terms fields.
  - Optional seed song input or selector using existing track lookup if available.
  - Preview button.
  - Export button.
- Implement `static/smart_sessions.js` with:
  - Capability loading on page load.
  - Preview request submission.
  - Loading, empty, warning, and error states.
  - Result rendering with position, title, artist, score, and explanation.
  - Remove-track interaction before export.
  - Playlist name editing.
  - Export request submission.
- Implement `/api/smart_sessions/export` by validating `playlist_name` and `track_ids`, then using the existing playlist creation path.
- Add user-safe success and failure messages.
- Style the page using existing theme variables in `static/style.css` and avoid page-specific hard-coded colors.

Implementation notes:

- Keep the first UI operational rather than decorative. The main screen should be the tool itself.
- The export endpoint should not trust the preview response. Revalidate that track IDs are non-empty strings and deduplicated.
- It is acceptable for manual reordering to be deferred if remove-before-export is implemented and the backend remains deterministic.
- Use existing button, card, field, and section conventions from current templates.

Day 4 deliverables:

- A user can generate a preview from `/smart_sessions`.
- A user can remove tracks from the preview.
- A user can export the preview to the media server.
- UI handles disabled CLAP, empty results, and export failures gracefully.

Definition of done:

- Browser workflow works end-to-end against the local Flask app.
- No old hard-coded blue styling is introduced.
- Export uses existing media-server adapters, not a new provider-specific implementation.

### Day 5 - Tests, Tuning, Documentation, And Release Readiness

Objective: harden the implementation and leave clear operating guidance.

Tasks:

- Add unit tests for:
  - Validation and error responses.
  - Candidate merge and deduplication.
  - Source fallback behavior.
  - Avoid rules.
  - Ranking determinism.
  - Max-per-artist constraints.
  - Export payload validation.
- Add endpoint tests for:
  - Capabilities endpoint.
  - Preview success with mocked candidates.
  - Preview degraded mode warnings.
  - Export success and media-server failure.
- Add a short feature section to `docs/ALGORITHM.md` after Instant Playlist or as a new numbered section.
- Add an entry to `docs/FAQ.md` explaining why Smart Sessions may return fewer tracks than requested.
- Add configuration notes to `docs/PARAMETERS.md` if new environment variables are introduced.
- Run the relevant test subset, then the broader test suite if time allows.
- Do a browser smoke test:
  - Prompt-only preview.
  - Anchor-only preview.
  - Prompt plus anchor preview.
  - Avoid artist filter.
  - Export playlist.

Implementation notes:

- Prefer no new environment variables unless the defaults need operator control.
- If new caps are added, use conservative names such as `SMART_SESSION_MAX_LENGTH`, `SMART_SESSION_DEFAULT_LENGTH`, and `SMART_SESSION_CANDIDATE_POOL_LIMIT`.
- Document degraded modes clearly: CLAP disabled, CLAP cache not loaded, SemGrove unavailable, no enough candidates.
- Keep logs useful but avoid logging full user credentials or provider tokens.

Day 5 deliverables:

- Tests for builder and endpoints.
- Updated docs for algorithm, FAQ, and parameters as needed.
- Manual smoke-test notes in the pull request or release checklist.
- Final tuning pass on ranking weights and warning text.

Definition of done:

- Tests pass for the new feature.
- The feature can be used without optional lyrics/SemGrove data if CLAP or Voyager candidates exist.
- The app can create a media-server playlist from a generated Smart Session.
- Documentation explains setup requirements and degraded behavior.

## Testing Matrix

| Area | Test Type | Cases |
|------|-----------|-------|
| Validation | Unit | Missing prompt and anchors, invalid length, invalid anchor, invalid avoid rules |
| Candidate sources | Unit | CLAP only, SemGrove only, combined sources, source unavailable |
| Filtering | Unit | Avoid artist, avoid title term, duplicate item ID, duplicate title |
| Ranking | Unit | Curve differences, deterministic ties, max-per-artist limits |
| Explanations | Unit | Prompt reason, anchor reason, curve reason, no unsupported claims |
| API | Flask tests | Capabilities, preview success, preview empty, export success, export failure |
| UI | Browser smoke | Preview, warnings, remove track, export, responsive layout |

## Rollout Risks And Mitigations

- CLAP cache may not be loaded: return a capability warning and fall back to anchors when possible.
- SemGrove may have sparse coverage: use Voyager nearest neighbors for anchor fallback.
- Ranking may feel repetitive: enforce `max_per_artist` and duplicate title penalties.
- Preview may be slow on large libraries: cap source limits and keep candidate scoring in memory.
- Export behavior differs by media server: reuse existing media-server helper paths and existing tests.
- Explanations may overpromise: use rule-based templates tied only to sources that were actually used.

## Future Iterations

- Add drag-and-drop manual reordering before export.
- Add saved Smart Session presets.
- Add a regenerate button that preserves constraints but reshuffles lower-ranked candidates.
- Add provider-backed AI parsing for freeform avoid rules and desired curve, while keeping rule-based parsing as fallback.
- Add history-aware freshness using media-server playback history when available.
- Add scheduled Smart Sessions that refresh weekly using `create_or_replace_playlist` where supported.