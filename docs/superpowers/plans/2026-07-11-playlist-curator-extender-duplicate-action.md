# Playlist Curator Extender Duplicate Action Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a visible, non-destructive `Find duplicates` action to Playlist Extender candidate results while retaining the existing Smart Search action.

**Architecture:** Reuse `window.curatorFindDuplicatesForTracks(tracks, 'search-results', options)` and the shared duplicate review panel. Extender owns a presentation-only set of manually hidden candidate IDs, supplies only currently visible candidates to the shared scan, and exposes `window.curatorHideSearchDuplicates(ids)` as the page hook used by the shared panel.

**Tech Stack:** Jinja HTML templates, browser JavaScript, Node.js `node:test` VM harnesses, Python pytest structural tests, Docker Compose.

## Global Constraints

- Keep the standalone Smart Search duplicate action unchanged.
- Scan only currently displayed Extender candidates in display order.
- Cap each manual scan at 500 candidates.
- `Hide marked duplicates` must not mutate the Workbench, seed playlist, database, or media server.
- Reset manually hidden candidates on a new Extender search or seed change.
- Preserve the existing automatic Search-result duplicates Off/Mark/Hide behavior independently.
- Recreate only the isolated `playlist-curator-test` Flask and worker services; production must remain unchanged.
- Do not stage or modify `deployment/docker-compose-nvidia.yaml`.

---

### Task 1: Expose the Extender candidate action

**Files:**
- Modify: `templates/playlist_curator_extender.html:94-103`
- Test: `test/unit/test_playlist_curator_templates.py`

**Interfaces:**
- Consumes: the existing Extender candidate-results section.
- Produces: `#curator-extender-finddups`, disabled by default and available to `curator-extender.js`.

- [ ] **Step 1: Write the failing structural test**

```python
def test_extender_results_exposes_bounded_duplicate_action():
    template = EXTENDER_TEMPLATE.read_text(encoding="utf-8")
    source = CURATOR_EXTENDER_JS.read_text(encoding="utf-8")

    assert 'id="curator-extender-finddups"' in template
    assert "const DUPLICATE_SCAN_LIMIT = 500;" in source
    assert "window.curatorFindDuplicatesForTracks(" in source
    assert "window.curatorHideSearchDuplicates = hideSearchDuplicates;" in source
```

- [ ] **Step 2: Run the test and verify RED**

Run: `python -m pytest -q test/unit/test_playlist_curator_templates.py::test_extender_results_exposes_bounded_duplicate_action`

Expected: FAIL because `curator-extender-finddups` is absent.

- [ ] **Step 3: Add the minimal button markup**

Place this inside `.curator-section-head`, after the heading block:

```html
<button type="button" id="curator-extender-finddups" class="curator-btn"
        data-kind="secondary" data-size="sm" disabled
        title="Find duplicates among currently displayed candidates">
    Find duplicates
</button>
```

- [ ] **Step 4: Run the structural test again**

Expected: it still fails only on the not-yet-added JavaScript contract assertions, proving the template portion is present.

- [ ] **Step 5: Commit the independently reviewable UI contract**

```powershell
git add -- templates/playlist_curator_extender.html test/unit/test_playlist_curator_templates.py
git commit -m "test: specify extender duplicate action"
```

---

### Task 2: Scan and hide visible Extender candidates

**Files:**
- Create: `test/unit/test_playlist_curator_extender_duplicates.js`
- Modify: `static/playlist_curator/curator-extender.js`

**Interfaces:**
- Consumes: `window.curatorFindDuplicatesForTracks(tracks, mode, options)` from `curator-shared.js`.
- Produces: `window.curatorHideSearchDuplicates(ids: string[])`, presentation-only candidate hiding, and a 500-item scan adapter.

- [ ] **Step 1: Write failing behavior tests with an isolated DOM harness**

The harness executes the checked-in `curator-extender.js` in `vm.runInNewContext`, supplies all controls read by `init()` and `runExtend()`, returns deterministic `/api/curator/search` results, records calls to `curatorFindDuplicatesForTracks`, and exposes button clicks plus seed changes.

```javascript
test('Extender duplicate scan preserves visible order and caps at 500', async () => {
    const results = tracks(502);
    const harness = await createHarness(results);
    await harness.elements.get('curator-extender-run').click();
    await harness.elements.get('curator-extender-finddups').click();

    assert.equal(harness.duplicateCalls.length, 1);
    assert.equal(harness.duplicateCalls[0][0].length, 500);
    assert.equal(harness.duplicateCalls[0][1], 'search-results');
    assert.deepEqual({ ...harness.duplicateCalls[0][2] }, {
        capped: true,
        totalAvailable: 502,
    });
    assert.deepEqual(
        Array.from(harness.duplicateCalls[0][0], track => track.item_id),
        results.slice(0, 500).map(track => track.item_id),
    );
});

test('Extender duplicate hiding is presentation-only and resets on a new search', async () => {
    const results = tracks(3);
    const harness = await createHarness(results);
    await harness.elements.get('curator-extender-run').click();

    harness.context.window.curatorHideSearchDuplicates(['track-2']);
    assert.doesNotMatch(harness.elements.get('curator-results-cards').innerHTML, /data-row-id="track-2"/);
    assert.deepEqual(harness.workbenchAdds, []);

    await harness.elements.get('curator-extender-run').click();
    assert.match(harness.elements.get('curator-results-cards').innerHTML, /data-row-id="track-2"/);
});

test('Extender seed change clears manually hidden candidates', async () => {
    const results = tracks(3);
    const harness = await createHarness(results);
    await harness.elements.get('curator-extender-run').click();
    harness.context.window.curatorHideSearchDuplicates(['track-2']);

    harness.changeSeed('');
    await harness.elements.get('curator-extender-run').click();
    assert.match(harness.elements.get('curator-results-cards').innerHTML, /data-row-id="track-2"/);
});
```

- [ ] **Step 2: Run the new test and verify RED**

Run: `node --test test/unit/test_playlist_curator_extender_duplicates.js`

Expected: FAIL because the button has no handler and `window.curatorHideSearchDuplicates` is undefined.

- [ ] **Step 3: Add bounded scan and presentation state**

Add state and helpers to `curator-extender.js`:

```javascript
const DUPLICATE_SCAN_LIMIT = 500;
let hiddenDuplicateIds = new Set();

function isAutomaticallyHiddenDuplicate(track) {
    const mode = document.getElementById('curator-dup-mode');
    const slider = document.getElementById('curator-dup-threshold');
    const distance = track.duplicate_of && typeof track.duplicate_of.distance === 'number'
        ? track.duplicate_of.distance
        : null;
    return !!mode && !!slider && mode.value === 'hide'
        && distance !== null && distance < parseFloat(slider.value);
}

function displayedResults() {
    return lastResults.filter(track => !hiddenDuplicateIds.has(String(track.item_id)));
}

function duplicateScanResults() {
    return displayedResults().filter(track => !isAutomaticallyHiddenDuplicate(track));
}

function updateDuplicateButton() {
    const button = document.getElementById('curator-extender-finddups');
    if (button) button.disabled = duplicateScanResults().length < 2;
}

function scanResultDuplicates() {
    const visible = duplicateScanResults();
    return window.curatorFindDuplicatesForTracks(
        visible.slice(0, DUPLICATE_SCAN_LIMIT),
        'search-results',
        {
            capped: visible.length > DUPLICATE_SCAN_LIMIT,
            totalAvailable: visible.length,
        },
    );
}

function hideSearchDuplicates(ids) {
    ids.forEach(id => hiddenDuplicateIds.add(String(id)));
    renderResults();
}

window.curatorHideSearchDuplicates = hideSearchDuplicates;
```

Render `displayedResults()` rather than `lastResults`, call `updateDuplicateButton()` after rendering and after automatic duplicate visibility changes, and bind `#curator-extender-finddups` to `scanResultDuplicates` during `init()`.

- [ ] **Step 4: Reset state at both lifecycle boundaries**

Before a validated Extender request starts and inside the seed-change listener, clear only the presentation state:

```javascript
hiddenDuplicateIds.clear();
```

Do not modify `lastResults`, Workbench tracks, `duplicate_of`, or backend data beyond the existing lifecycle behavior.

- [ ] **Step 5: Run focused tests and verify GREEN**

```powershell
node --test test/unit/test_playlist_curator_extender_duplicates.js
node --test test/unit/test_playlist_curator_extender_race.js
node --test test/unit/test_playlist_curator_shared.js
python -m pytest -q test/unit/test_playlist_curator_templates.py
```

Expected: all tests pass with no warnings or failures.

- [ ] **Step 6: Commit the behavior**

```powershell
git add -- static/playlist_curator/curator-extender.js test/unit/test_playlist_curator_extender_duplicates.js test/unit/test_playlist_curator_templates.py
git commit -m "feat: find duplicates in extender results"
```

---

### Task 3: Verify, publish, and refresh the isolated test instance

**Files:**
- Verify only: all files changed in Tasks 1-2.
- Preserve: `deployment/docker-compose-nvidia.yaml`.

**Interfaces:**
- Consumes: the completed Extender UI and tests.
- Produces: a green PR 417 commit deployed only to `playlist-curator-test`.

- [ ] **Step 1: Run the complete focused regression suite**

```powershell
python -m pytest -q @(rg --files test | Where-Object { $_ -match 'playlist_curator.*\.py$' })
node --test test/unit/test_playlist_curator_shared.js
node --test test/unit/test_playlist_curator_search_duplicates.js
node --test test/unit/test_playlist_curator_extender_duplicates.js
node --test test/unit/test_playlist_curator_extender_race.js
git diff --check
```

Expected: zero failures; `git status --short` shows only intentional commits plus the user's unstaged Docker Compose edit.

- [ ] **Step 2: Push and require all PR checks plus exact Sonar zero**

```powershell
git push origin feature/playlist-curator-port
gh pr checks 417 --repo NeptuneHub/AudioMuse-AI --watch
$uri='https://sonarcloud.io/api/issues/search?componentKeys=NeptuneHub_AudioMuse-AI&pullRequest=417&issueStatuses=OPEN%2CCONFIRMED&sinceLeakPeriod=true&ps=100'
(Invoke-RestMethod -Uri $uri).total
```

Expected: every required check passes and Sonar prints `0`.

- [ ] **Step 3: Rebuild and recreate only isolated web and worker services**

```powershell
docker build --build-arg BASE_IMAGE=ubuntu:24.04 -t audiomuse-ai-playlist-curator:test .
docker compose -p playlist-curator-test -f C:\tmp\playlist-curator-test.compose.yaml up -d --no-deps --force-recreate audiomuse-ai-flask audiomuse-ai-worker
```

Carry forward only the existing test Compose variables in-memory. Do not recreate Postgres, Redis, or any production service.

- [ ] **Step 4: Verify the deployed UI, worker, and production isolation**

Confirm:

```powershell
docker inspect playlist-curator-test-flask --format '{{.State.Health.Status}}|{{.Image}}|{{json .HostConfig.DeviceRequests}}'
docker top playlist-curator-test-worker -eo pid,ppid,comm,args
docker exec playlist-curator-test-redis redis-cli SCARD rq:workers
curl.exe -sS --max-time 45 -o NUL -w '%{http_code}' http://localhost:18001/api/health
docker inspect audiomuse-ai-worker-instance --format '{{.Id}}|{{.Image}}|{{.State.StartedAt}}|{{.State.Status}}'
```

Expected: test Flask is healthy, the test worker has two RQ workers plus the janitor and no Flask process, Redis reports two registrations, HTTP is `200`, and the production fingerprint exactly matches the pre-deployment baseline.
