# Playlist Curator Visible Duplicate Scans Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make duplicate scans visibly complete from the Workbench and add a bounded, non-destructive duplicate scan for currently loaded Smart Search results.

**Architecture:** `curator-shared.js` remains the single owner of the duplicate API request and review panel, but gains an explicit scan mode and track-list interface. `curator-search.js` owns Search Results selection, the 500-track cap, presentation-only hiding, and reset behavior; the search template adds the overview entry point. The backend contract and duplicate algorithm remain unchanged.

**Tech Stack:** Flask/Jinja templates, browser JavaScript, Node's built-in test runner, Python/pytest structural template tests.

## Global Constraints

- Workbench scans inspect Workbench tracks only.
- Search Results scans inspect only currently loaded, visible results.
- Search Results scans submit at most 500 tracks in current display order.
- Search Results hiding never changes the Workbench, database, media library, or media-server playlists.
- Empty and error results remain visible until explicitly closed or replaced by another scan.
- Do not modify `deployment/docker-compose-nvidia.yaml`; it contains unrelated user work.
- Reuse `POST /api/curator/find_duplicates`; do not change the backend request or response contract.

---

## File Structure

- Modify `static/playlist_curator/curator-shared.js`: shared scan lifecycle, persistent feedback, scan mode, and mode-specific apply action.
- Modify `static/playlist_curator/curator-search.js`: current-result scan adapter, 500-track cap, hidden duplicate state, rerender/reset hooks.
- Modify `templates/playlist_curator_search.html`: Search Results `Find duplicates` button.
- Modify `test/unit/test_playlist_curator_shared.js`: behavioral coverage of persistent Workbench feedback and mode-specific apply behavior.
- Create `test/unit/test_playlist_curator_search_duplicates.js`: behavioral coverage of result ordering, cap, hiding, and reset behavior.
- Modify `test/unit/test_playlist_curator_templates.py`: structural/accessibility guard for the new overview action and shared interfaces.

### Task 1: Keep Workbench Duplicate Feedback Visible

**Files:**
- Modify: `test/unit/test_playlist_curator_shared.js`
- Modify: `static/playlist_curator/curator-shared.js:575-671`

**Interfaces:**
- Consumes: existing `window.curatorFindDuplicates()` Workbench action.
- Produces: the same public function, but it reveals and scrolls the panel before awaiting the API and never auto-hides an empty result.

- [ ] **Step 1: Extend the shared-script harness with the duplicate panel**

Add these IDs to the harness element list:

```javascript
'curator-dedup-panel',
'curator-dedup-groups',
'curator-dedup-title',
'curator-dedup-threshold',
'curator-dedup-removeall',
'curator-dedup-close',
```

Initialize the panel with `hidden`, give the threshold element value `0.010`, and record scroll calls:

```javascript
const scrollCalls = [];
const panel = elements.get('curator-dedup-panel');
panel.classList.add('hidden');
panel.scrollIntoView = options => scrollCalls.push(options);
elements.get('curator-dedup-threshold').value = '0.010';
```

Return `scrollCalls` from `createHarness()`.

- [ ] **Step 2: Write the failing persistent-empty-result test**

```javascript
test('empty Workbench duplicate scan scrolls to persistent feedback', async () => {
    const harness = createHarness();
    harness.context.window.workbenchAdd(track('track-1'), 'search');
    harness.context.window.workbenchAdd(track('track-2'), 'search');
    harness.setFetchResponse(Promise.resolve(response({
        groups: [],
        total_groups: 0,
        total_duplicate_tracks: 0,
    })));

    await harness.context.window.curatorFindDuplicates();

    const panel = harness.elements.get('curator-dedup-panel');
    assert.equal(panel.classList.contains('hidden'), false);
    assert.equal(harness.scrollCalls.length, 1);
    assert.match(
        harness.elements.get('curator-dedup-groups').innerHTML,
        /No duplicates found/,
    );
    assert.equal(
        harness.elements.get('curator-dedup-title').textContent,
        'No Duplicates Found',
    );
});
```

- [ ] **Step 3: Run the test to verify RED**

Run:

```powershell
node --test test/unit/test_playlist_curator_shared.js
```

Expected: FAIL because the current empty branch never scrolls the panel and schedules it to hide.

- [ ] **Step 4: Implement the minimal persistent feedback change**

In `findDuplicates()` set a fresh title, reveal and scroll the panel before `fetch`:

```javascript
if (!panel || !container) return;
if (titleEl) titleEl.textContent = 'Finding Duplicates';
panel.classList.remove('hidden');
panel.scrollIntoView({ behavior: 'smooth', block: 'start' });
container.innerHTML = '<p class="curator-status loading"><span class="curator-spinner"></span>Scanning for duplicates...</p>';
```

Delete the empty-result auto-hide call:

```javascript
// Remove: setTimeout(() => panel.classList.add('hidden'), 3000);
```

Do not add a replacement timer.

- [ ] **Step 5: Verify GREEN and commit**

Run:

```powershell
node --test test/unit/test_playlist_curator_shared.js
python -m ruff check test/unit/test_playlist_curator_templates.py
git diff --check
```

Expected: all shared JavaScript tests pass, Ruff passes, and `git diff --check` is clean.

Commit:

```powershell
git add static/playlist_curator/curator-shared.js test/unit/test_playlist_curator_shared.js
git commit -m "fix: keep duplicate scan feedback visible"
```

### Task 2: Add Mode-Aware Shared Duplicate Review

**Files:**
- Modify: `test/unit/test_playlist_curator_shared.js`
- Modify: `static/playlist_curator/curator-shared.js:575-692`
- Modify: `templates/includes/_curator_dedup.html:3-7`

**Interfaces:**
- Consumes: `tracks: Array<{item_id: string}>`, mode `'workbench' | 'search-results'`, and options `{capped?: boolean, totalAvailable?: number}`.
- Produces: `window.curatorFindDuplicatesForTracks(tracks, mode, options): Promise<void>`.
- Produces: Search Results apply callback `window.curatorHideSearchDuplicates(ids: string[]): void`, supplied later by `curator-search.js`.
- Preserves: `window.curatorFindDuplicates()` as the Workbench wrapper used by existing rail/sheet buttons.

- [ ] **Step 1: Write the failing Search Results mode tests**

Add a helper for duplicate responses:

```javascript
function duplicateResponse() {
    return response({
        groups: [{
            tracks: [
                { item_id: 'keep', title: 'Keep', score: 5 },
                { item_id: 'hide', title: 'Hide', score: 1 },
            ],
        }],
        total_groups: 1,
        total_duplicate_tracks: 2,
    });
}
```

Add these tests:

```javascript
test('Search Results duplicate scan uses provided tracks and relabels apply action', async () => {
    const harness = createHarness();
    harness.setFetchResponse(Promise.resolve(duplicateResponse()));

    await harness.context.window.curatorFindDuplicatesForTracks(
        [track('keep'), track('hide')],
        'search-results',
        { capped: false, totalAvailable: 2 },
    );

    const body = JSON.parse(harness.fetchCalls[0].options.body);
    assert.deepEqual(Array.from(body.track_ids), ['keep', 'hide']);
    assert.equal(
        harness.elements.get('curator-dedup-removeall').textContent,
        'Hide marked duplicates',
    );
});

test('Search Results apply hides marked ids without changing Workbench', async () => {
    const harness = createHarness();
    const hidden = [];
    harness.context.window.curatorHideSearchDuplicates = ids => hidden.push(...ids);
    harness.setFetchResponse(Promise.resolve(duplicateResponse()));

    await harness.context.window.curatorFindDuplicatesForTracks(
        [track('keep'), track('hide')],
        'search-results',
        {},
    );
    harness.context.window.curatorRemoveAllMarkedDuplicates();

    assert.deepEqual(hidden, ['hide']);
    assert.deepEqual(Array.from(harness.context.window.getWorkbench().tracks), []);
});
```

- [ ] **Step 2: Run the tests to verify RED**

Run:

```powershell
node --test test/unit/test_playlist_curator_shared.js
```

Expected: FAIL because `curatorFindDuplicatesForTracks` does not exist and the apply action always mutates the Workbench.

- [ ] **Step 3: Implement the explicit shared scan interface**

Add mode state beside `dedupGroups`:

```javascript
let dedupGroups = [];
let dedupMode = 'workbench';
```

Extract the request into:

```javascript
async function findDuplicatesForTracks(tracks, mode, options) {
    const scanTracks = Array.isArray(tracks) ? tracks : [];
    const scanMode = mode === 'search-results' ? 'search-results' : 'workbench';
    const scanOptions = options || {};
    if (scanTracks.length < 2) {
        toast('Need at least 2 tracks to scan.', 'error');
        return;
    }
    dedupMode = scanMode;
    // Reuse the existing panel lookup, loading state, fetch, and response render.
    // Submit scanTracks.map(track => track.item_id), never getWorkbench() here.
    // Append ` Scanned the first 500 of ${scanOptions.totalAvailable} loaded results.`
    // to the loading/empty summary only when scanOptions.capped is true.
}

async function findDuplicates() {
    return findDuplicatesForTracks(getWorkbench().tracks, 'workbench', {});
}

window.curatorFindDuplicates = findDuplicates;
window.curatorFindDuplicatesForTracks = findDuplicatesForTracks;
```

Update the primary action label after mode selection:

```javascript
const applyButton = document.getElementById('curator-dedup-removeall');
if (applyButton) {
    applyButton.textContent = dedupMode === 'search-results'
        ? 'Hide marked duplicates'
        : 'Remove All Marked';
}
```

Dispatch apply behavior by mode:

```javascript
if (dedupMode === 'search-results') {
    if (typeof window.curatorHideSearchDuplicates === 'function') {
        window.curatorHideSearchDuplicates(idsToRemove);
    }
    toast(`Hidden ${idsToRemove.length} duplicate${idsToRemove.length === 1 ? '' : 's'} from Search Results.`, 'success');
} else {
    idsToRemove.forEach(id => workbenchRemove(id));
    toast(`Removed ${idsToRemove.length} duplicate${idsToRemove.length === 1 ? '' : 's'} from Workbench.`, 'success');
}
```

Reset `dedupMode` to `workbench` in `closeDedupPanel()`.

- [ ] **Step 4: Update template copy without changing IDs**

Keep `id="curator-dedup-removeall"` and its default text `Remove All Marked`. Add an `aria-live="polite"` region to the groups container so loading, empty, and error results are announced:

```html
<div id="curator-dedup-groups" aria-live="polite"></div>
```

- [ ] **Step 5: Verify GREEN and commit**

Run:

```powershell
node --test test/unit/test_playlist_curator_shared.js
python -m pytest -q test/unit/test_playlist_curator_templates.py
node --check static/playlist_curator/curator-shared.js
git diff --check
```

Expected: all commands pass.

Commit:

```powershell
git add static/playlist_curator/curator-shared.js templates/includes/_curator_dedup.html test/unit/test_playlist_curator_shared.js
git commit -m "feat: support duplicate review modes"
```

### Task 3: Add Bounded Smart Search Duplicate Scans

**Files:**
- Create: `test/unit/test_playlist_curator_search_duplicates.js`
- Modify: `test/unit/test_playlist_curator_templates.py`
- Modify: `templates/playlist_curator_search.html:44-52`
- Modify: `static/playlist_curator/curator-search.js:55-65,235-365,400-435,582-610`

**Interfaces:**
- Consumes: `window.curatorFindDuplicatesForTracks(tracks, 'search-results', options)` from Task 2.
- Produces: `window.curatorHideSearchDuplicates(ids: string[]): void` for Task 2's apply action.
- Produces: overview button `#curator-search-finddups`.

- [ ] **Step 1: Add the failing template guard**

Add `SEARCH_TEMPLATE` and `CURATOR_SEARCH_JS` paths to `test_playlist_curator_templates.py`, then add:

```python
def test_search_results_exposes_bounded_duplicate_action():
    template = SEARCH_TEMPLATE.read_text(encoding="utf-8")
    source = CURATOR_SEARCH_JS.read_text(encoding="utf-8")

    assert 'id="curator-search-finddups"' in template
    assert "const DUPLICATE_SCAN_LIMIT = 500;" in source
    assert "window.curatorFindDuplicatesForTracks(" in source
    assert "window.curatorHideSearchDuplicates = hideSearchDuplicates;" in source
```

- [ ] **Step 2: Create a failing behavioral search test**

Create `test_playlist_curator_search_duplicates.js` with the repository JS legal header. Load `curator-search.js` in an isolated VM using a minimal DOM that supplies the existing initialization controls and `#curator-search-finddups`. Capture the shared scan call:

```javascript
const duplicateCalls = [];
window.curatorFindDuplicatesForTracks = (...args) => duplicateCalls.push(args);
```

Seed 502 loaded results through the mocked `/api/curator/search` response, click the overview action, and assert:

```javascript
assert.equal(duplicateCalls.length, 1);
assert.equal(duplicateCalls[0][0].length, 500);
assert.equal(duplicateCalls[0][1], 'search-results');
assert.deepEqual(duplicateCalls[0][2], {
    capped: true,
    totalAvailable: 502,
});
assert.deepEqual(
    duplicateCalls[0][0].map(track => track.item_id),
    results.slice(0, 500).map(track => track.item_id),
);
```

Add a second test that calls `window.curatorHideSearchDuplicates(['track-2'])`, rerenders, and confirms `track-2` is absent while the Workbench mock is unchanged. Run a new search response and confirm `track-2` appears again.

- [ ] **Step 3: Run the new tests to verify RED**

Run:

```powershell
python -m pytest -q test/unit/test_playlist_curator_templates.py
node --test test/unit/test_playlist_curator_search_duplicates.js
```

Expected: template test fails because the overview button/interface is absent; Node test fails because no search duplicate adapter or hiding hook exists.

- [ ] **Step 4: Add the Search Results overview action**

In `playlist_curator_search.html`, add this button to `.curator-section-head`:

```html
<button type="button" id="curator-search-finddups"
        class="curator-btn" data-kind="secondary" data-size="sm" disabled>
    Find duplicates
</button>
```

The button scans loaded Search Results, not the Workbench; keep that distinction in its `title`:

```html
title="Find duplicates among currently loaded search results"
```

- [ ] **Step 5: Implement bounded result selection and presentation-only hiding**

Add state:

```javascript
const DUPLICATE_SCAN_LIMIT = 500;
let hiddenDuplicateIds = new Set();
```

Filter it in `visibleResults()`:

```javascript
return lastResults.filter(track =>
    !skippedIds.has(track.item_id) && !hiddenDuplicateIds.has(String(track.item_id))
);
```

Update the overview button in `renderResults()`:

```javascript
const findDuplicatesBtn = document.getElementById('curator-search-finddups');
if (findDuplicatesBtn) findDuplicatesBtn.disabled = visible.length < 2;
```

Add the page adapter and hiding hook:

```javascript
function scanSearchResultDuplicates() {
    const visible = visibleResults();
    const tracks = visible.slice(0, DUPLICATE_SCAN_LIMIT);
    return window.curatorFindDuplicatesForTracks(
        tracks,
        'search-results',
        {
            capped: visible.length > DUPLICATE_SCAN_LIMIT,
            totalAvailable: visible.length,
        },
    );
}

function hideSearchDuplicates(ids) {
    (ids || []).forEach(id => hiddenDuplicateIds.add(String(id)));
    renderResults();
}

window.curatorHideSearchDuplicates = hideSearchDuplicates;
```

Attach `scanSearchResultDuplicates` to `#curator-search-finddups` in `init()`.

Reset `hiddenDuplicateIds = new Set()` alongside `skippedIds = new Set()` in `runSearch()`, `fetchPage()`, completed `loadAllPages()`, and the Clear All handler.

- [ ] **Step 6: Verify GREEN and commit**

Run:

```powershell
python -m pytest -q test/unit/test_playlist_curator_templates.py test/unit/test_file_header_convention.py
node --test test/unit/test_playlist_curator_search_duplicates.js test/unit/test_playlist_curator_shared.js
node --check static/playlist_curator/curator-search.js
node --check static/playlist_curator/curator-shared.js
python -m ruff check test/unit/test_playlist_curator_templates.py
git diff --check
```

Expected: all commands pass and the new JS test file passes the repository legal-header policy.

Commit:

```powershell
git add static/playlist_curator/curator-search.js templates/playlist_curator_search.html test/unit/test_playlist_curator_search_duplicates.js test/unit/test_playlist_curator_templates.py
git commit -m "feat: scan duplicates in search results"
```

### Task 4: Full Verification, Review, Push, and Isolated Deployment

**Files:**
- Verify only; no new production files expected.
- Preserve: `deployment/docker-compose-nvidia.yaml` unstaged.

**Interfaces:**
- Consumes: all Task 1-3 behavior.
- Produces: reviewed commits on `feature/playlist-curator-port`, green PR 417 checks, and updated isolated test services.

- [ ] **Step 1: Run focused and repository checks**

```powershell
node --test test/unit/test_playlist_curator_shared.js test/unit/test_playlist_curator_search_duplicates.js test/unit/test_playlist_curator_extender_race.js
python -m pytest -q test/unit -k "playlist_curator"
node --check static/playlist_curator/curator-shared.js
node --check static/playlist_curator/curator-search.js
python -m ruff check test/unit/test_playlist_curator_templates.py
python -m pytest -q test/unit/test_file_header_convention.py test/unit/test_no_emoji_in_source.py test/unit/test_no_em_dash_in_source.py test/unit/test_line_endings_index.py
git diff --check
```

Do not pass JavaScript files to Ruff; use `node --check` for them. Expected: all applicable commands pass.

- [ ] **Step 2: Request code review and resolve findings**

Review the complete Task 1-3 diff for:

- empty/loading/error visibility;
- Workbench versus Search Results mode isolation;
- 500-track order/cap enforcement;
- no Workbench/library mutation from Search Results hiding;
- reset behavior on run, page change, load-all, and clear;
- accessibility and mobile behavior.

Apply only verified findings, rerunning the focused tests after each fix.

- [ ] **Step 3: Push and wait for PR gates**

```powershell
git status --short
git push origin feature/playlist-curator-port
gh pr checks 417 --repo NeptuneHub/AudioMuse-AI --watch --interval 10
```

Expected: unit, integration, security, static checks, and SonarCloud pass. Query the Sonar PR issue API and require zero open new issues.

- [ ] **Step 4: Rebuild and recreate the isolated test services**

```powershell
docker build --build-arg BASE_IMAGE=ubuntu:24.04 -t audiomuse-ai-playlist-curator:test .
$source = (docker inspect 'audiomuse-test-flask-app' | ConvertFrom-Json)[0]
$sourceEnv = @{}
foreach ($entry in $source.Config.Env) {
    $parts = $entry -split '=', 2
    $sourceEnv[$parts[0]] = $parts[1]
}
$keys = @(
    'POSTGRES_USER', 'POSTGRES_PASSWORD', 'POSTGRES_DB', 'TZ',
    'MEDIASERVER_TYPE', 'JELLYFIN_URL', 'JELLYFIN_USER_ID', 'JELLYFIN_TOKEN',
    'AUTH_ENABLED', 'AUDIOMUSE_USER', 'AUDIOMUSE_PASSWORD', 'API_TOKEN', 'JWT_SECRET'
)
foreach ($key in $keys) {
    if ($sourceEnv.ContainsKey($key) -and $sourceEnv[$key]) {
        Set-Item -Path "Env:$key" -Value $sourceEnv[$key]
    }
}
docker compose -p playlist-curator-test -f C:\tmp\playlist-curator-test.compose.yaml up -d --no-deps --force-recreate audiomuse-ai-flask audiomuse-ai-worker
```

- [ ] **Step 5: Verify the deployed interaction and production isolation**

On `http://localhost:18001/playlist_curator/search`:

1. Load at least two results.
2. Confirm the overview `Find duplicates` action becomes enabled.
3. Run a scan and confirm loading/empty/success feedback scrolls into view and remains visible.
4. If groups exist, use `Hide marked duplicates` and confirm only displayed results change.
5. Run a new search and confirm hidden rows reset.

On the Extender, run the Workbench scan and confirm it remains Workbench-only. Reinspect `audiomuse-ai-worker-instance` and require the production container ID, image, and `StartedAt` values to match the saved baseline.
