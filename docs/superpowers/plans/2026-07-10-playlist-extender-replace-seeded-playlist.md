# Playlist Extender Seed Replacement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a user replace the media-server playlist that seeded Playlist Extender while preserving the existing create-new playlist flow.

**Architecture:** Extend the existing curator save endpoint with a mutually exclusive `replace_playlist_name` action that validates the target and delegates to the established name-based `create_or_replace_playlist` helper. Keep replacement target state in `curator-shared.js`; `curator-extender.js` supplies that state when a media-server seed is loaded, and the shared Workbench renders and handles contextual replacement controls on desktop and mobile.

**Tech Stack:** Python 3, Flask, pytest/unittest.mock, vanilla JavaScript, Jinja templates, CSS, Docker Compose.

## Global Constraints

- Creating a new playlist remains available and is the non-destructive default.
- Replacement is available only after loading a media-server playlist seed.
- Replacement uses the current Workbench order after de-duplicating IDs by first occurrence.
- The selected approach is name-based replacement through `tasks.mediaserver.create_or_replace_playlist`.
- A duplicate playlist name may be resolved by the provider's existing first-match behavior; do not add exact-ID adapter APIs.
- Replacement requires explicit confirmation and warns when unanalyzed seed tracks will be removed.
- Failed or cancelled saves preserve the Workbench and replacement target.
- An unsupported media-server replacement returns HTTP 501 and never falls back to create-new.
- Replacement target state is in memory only and must not survive page reload.
- Do not modify or stage `deployment/docker-compose-nvidia.yaml`; it is an unrelated user change.
- Deployment must recreate only the isolated `playlist-curator-test` web container; do not create a worker or request a GPU.

---

## File Structure

- Create `test/unit/test_playlist_curator_save.py`: focused Flask route tests for create-new, replace, validation, and failures.
- Modify `app_playlist_curator.py`: implement the mutually exclusive create/replace save contract.
- Modify `templates/includes/_curator_workbench.html`: add desktop and mobile contextual replacement buttons.
- Modify `static/playlist_curator/curator-shared.js`: own seeded-target state, render replacement actions, confirm, submit, and clear safely.
- Modify `static/playlist_curator/curator-extender.js`: retain the selected server playlist name and unresolved-track count when loading the seed.
- Modify `static/playlist_curator/curator.css`: make the new controls fit both Workbench layouts.
- Modify `test/unit/test_playlist_curator_templates.py`: assert the frontend save contract and accessible controls.

### Task 1: Add the Backend Replacement Action

**Files:**
- Create: `test/unit/test_playlist_curator_save.py`
- Modify: `app_playlist_curator.py:743-779`

**Interfaces:**
- Consumes: `tasks.mediaserver.create_or_replace_playlist(playlist_name: str, item_ids: list[str], user_creds=None) -> dict | None` and `_fetch_server_playlists() -> list[dict]`.
- Produces: `POST /api/curator/save_playlist` accepting exactly one of `new_playlist_name` or `replace_playlist_name`, plus `track_ids: list`, returning an `action` of `created` or `replaced`.

- [ ] **Step 1: Write focused failing route tests**

Create `test/unit/test_playlist_curator_save.py` with:

```python
"""Playlist Curator save-new and replace-seeded-playlist API tests."""

from unittest.mock import patch

import pytest
from flask import Flask

import app_playlist_curator


@pytest.fixture
def client():
    app = Flask(__name__)
    app.config.update(TESTING=True)
    app.register_blueprint(app_playlist_curator.playlist_curator_bp)
    return app.test_client()


def test_save_playlist_create_new_remains_available(client):
    with patch(
        "tasks.ivf_manager.create_playlist_from_ids", return_value="new-123"
    ) as create:
        response = client.post(
            "/api/curator/save_playlist",
            json={"new_playlist_name": "New Mix", "track_ids": ["a", "a", "b"]},
        )

    assert response.status_code == 201
    assert response.get_json() == {
        "action": "created",
        "message": "Playlist 'New Mix' created with 2 songs!",
        "playlist_id": "new-123",
        "playlist_name": "New Mix",
        "total_songs": 2,
    }
    create.assert_called_once_with("New Mix", ["a", "b"])


def test_save_playlist_replaces_existing_name(client):
    playlists = [{"Id": "seed-1", "Name": "Road Trip"}]
    replaced = {"Id": "seed-1", "Name": "Road Trip"}
    with (
        patch.object(app_playlist_curator, "_fetch_server_playlists", return_value=playlists),
        patch(
            "tasks.mediaserver.create_or_replace_playlist", return_value=replaced
        ) as replace,
        patch("tasks.ivf_manager.create_playlist_from_ids") as create,
    ):
        response = client.post(
            "/api/curator/save_playlist",
            json={"replace_playlist_name": "Road Trip", "track_ids": [1, 1, 2]},
        )

    assert response.status_code == 200
    assert response.get_json() == {
        "action": "replaced",
        "message": "Playlist 'Road Trip' replaced with 2 songs!",
        "playlist_id": "seed-1",
        "playlist_name": "Road Trip",
        "total_songs": 2,
    }
    replace.assert_called_once_with("Road Trip", ["1", "2"])
    create.assert_not_called()


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"track_ids": ["a"]}, "Provide exactly one playlist save action"),
        (
            {
                "new_playlist_name": "New",
                "replace_playlist_name": "Old",
                "track_ids": ["a"],
            },
            "Provide exactly one playlist save action",
        ),
        (
            {"replace_playlist_name": 42, "track_ids": ["a"]},
            "Playlist name must be a non-empty string",
        ),
        (
            {"replace_playlist_name": "   ", "track_ids": ["a"]},
            "Playlist name must be a non-empty string",
        ),
        (
            {"replace_playlist_name": "Road Trip", "track_ids": "a"},
            "Track IDs must be a non-empty list",
        ),
        (
            {"replace_playlist_name": "Road Trip", "track_ids": []},
            "Track IDs must be a non-empty list",
        ),
    ],
)
def test_save_playlist_rejects_invalid_action_payloads(client, payload, message):
    response = client.post("/api/curator/save_playlist", json=payload)

    assert response.status_code == 400
    assert response.get_json() == {"error": message}


def test_save_playlist_replacement_requires_existing_exact_name(client):
    with (
        patch.object(
            app_playlist_curator,
            "_fetch_server_playlists",
            return_value=[{"Id": "seed-1", "Name": "Road Trip"}],
        ),
        patch("tasks.mediaserver.create_or_replace_playlist") as replace,
    ):
        response = client.post(
            "/api/curator/save_playlist",
            json={"replace_playlist_name": "road trip", "track_ids": ["a"]},
        )

    assert response.status_code == 404
    assert response.get_json() == {"error": "Playlist 'road trip' no longer exists"}
    replace.assert_not_called()


def test_save_playlist_reports_unsupported_replacement(client):
    with (
        patch.object(
            app_playlist_curator,
            "_fetch_server_playlists",
            return_value=[{"id": "seed-1", "name": "Road Trip"}],
        ),
        patch(
            "tasks.mediaserver.create_or_replace_playlist",
            side_effect=NotImplementedError,
        ),
    ):
        response = client.post(
            "/api/curator/save_playlist",
            json={"replace_playlist_name": "Road Trip", "track_ids": ["a"]},
        )

    assert response.status_code == 501
    assert response.get_json() == {
        "error": "Replacing playlists is not supported by this media server"
    }


def test_save_playlist_treats_falsey_provider_result_as_failure(client):
    with (
        patch.object(
            app_playlist_curator,
            "_fetch_server_playlists",
            return_value=[{"Id": "seed-1", "Name": "Road Trip"}],
        ),
        patch("tasks.mediaserver.create_or_replace_playlist", return_value=None),
    ):
        response = client.post(
            "/api/curator/save_playlist",
            json={"replace_playlist_name": "Road Trip", "track_ids": ["a"]},
        )

    assert response.status_code == 502
    assert response.get_json() == {"error": "Media server failed to replace playlist"}
```

- [ ] **Step 2: Run the new tests and verify the missing contract fails**

Run:

```powershell
python -m pytest -q test/unit/test_playlist_curator_save.py
```

Expected: failures because the route still requires `new_playlist_name`, does not call `create_or_replace_playlist`, and does not return `action` or `playlist_name`.

- [ ] **Step 3: Implement the mutually exclusive create/replace route**

Replace `save_playlist_api` in `app_playlist_curator.py` with:

```python
@playlist_curator_bp.route('/api/curator/save_playlist', methods=['POST'])
def save_playlist_api():
    """Create a new curator playlist or replace the named server-playlist seed."""
    from tasks.ivf_manager import create_playlist_from_ids
    from tasks.mediaserver import create_or_replace_playlist

    payload = request.get_json(silent=True) or {}
    has_new_name = 'new_playlist_name' in payload
    has_replace_name = 'replace_playlist_name' in payload
    if has_new_name == has_replace_name:
        return jsonify({"error": "Provide exactly one playlist save action"}), 400

    action = 'replaced' if has_replace_name else 'created'
    raw_name = payload.get('replace_playlist_name' if has_replace_name else 'new_playlist_name')
    if not isinstance(raw_name, str) or not raw_name.strip():
        return jsonify({"error": "Playlist name must be a non-empty string"}), 400
    playlist_name = raw_name.strip()

    track_ids = payload.get('track_ids')
    if not isinstance(track_ids, list) or not track_ids:
        return jsonify({"error": "Track IDs must be a non-empty list"}), 400

    seen = set()
    final_ids = []
    for track_id in track_ids:
        normalized = str(track_id)
        if normalized not in seen:
            seen.add(normalized)
            final_ids.append(normalized)

    try:
        if has_replace_name:
            existing_names = {
                str(playlist.get('Name') or playlist.get('name') or '').strip()
                for playlist in (_fetch_server_playlists() or [])
            }
            if playlist_name not in existing_names:
                return jsonify({
                    "error": f"Playlist '{playlist_name}' no longer exists"
                }), 404
            try:
                replaced = create_or_replace_playlist(playlist_name, final_ids)
            except NotImplementedError:
                return jsonify({
                    "error": "Replacing playlists is not supported by this media server"
                }), 501
            if not replaced:
                return jsonify({"error": "Media server failed to replace playlist"}), 502
            playlist_id = replaced.get('Id') or replaced.get('id')
            if not playlist_id:
                return jsonify({"error": "Media server replacement returned no playlist ID"}), 502
            status = 200
            message = f"Playlist '{playlist_name}' replaced with {len(final_ids)} songs!"
        else:
            playlist_id = create_playlist_from_ids(playlist_name, final_ids)
            status = 201
            message = f"Playlist '{playlist_name}' created with {len(final_ids)} songs!"

        return jsonify({
            "action": action,
            "message": message,
            "playlist_id": playlist_id,
            "playlist_name": playlist_name,
            "total_songs": len(final_ids),
        }), status
    except Exception:
        logger.exception("Save curator playlist failed")
        return jsonify({"error": INTERNAL_ERROR_MESSAGE}), 500
```

- [ ] **Step 4: Run backend tests and lint**

Run:

```powershell
python -m pytest -q test/unit/test_playlist_curator_save.py
python -m ruff check app_playlist_curator.py test/unit/test_playlist_curator_save.py
```

Expected: all save tests pass and Ruff reports `All checks passed!`.

- [ ] **Step 5: Commit the backend action**

```powershell
git add app_playlist_curator.py test/unit/test_playlist_curator_save.py
git commit -m "feat: replace named curator playlist"
```

### Task 2: Add Contextual Replacement Controls to the Shared Workbench

**Files:**
- Modify: `templates/includes/_curator_workbench.html:34-47,76-85`
- Modify: `static/playlist_curator/curator-shared.js:54-151,183-293,454-477,599-655`
- Modify: `static/playlist_curator/curator.css:886-895,985-992`
- Modify: `test/unit/test_playlist_curator_templates.py`

**Interfaces:**
- Consumes: Task 1's `replace_playlist_name` payload and `action: "replaced"` response.
- Produces: `window.curatorSetSeededPlaylistTarget(target)` where `target` is `null` or `{playlistId: string, playlistName: string, unresolvedTracks: number}`, plus `window.curatorReplaceSeededPlaylist()`.

- [ ] **Step 1: Write failing template and shared-script contract tests**

Append to `test/unit/test_playlist_curator_templates.py`:

```python
WORKBENCH_TEMPLATE = REPO_ROOT / "templates" / "includes" / "_curator_workbench.html"
CURATOR_SHARED_JS = REPO_ROOT / "static" / "playlist_curator" / "curator-shared.js"


def test_workbench_keeps_create_new_and_adds_contextual_replace_controls():
    template = WORKBENCH_TEMPLATE.read_text(encoding="utf-8")

    assert 'id="curator-wb-name"' in template
    assert 'id="curator-wb-save-btn"' in template
    assert 'id="curator-sheet-name"' in template
    assert 'id="curator-sheet-save-btn"' in template
    assert 'id="curator-wb-replace-btn"' in template
    assert 'id="curator-sheet-replace-btn"' in template
    assert template.count('Replace seeded playlist') == 2


def test_shared_workbench_owns_nonpersistent_seed_target_and_replace_payload():
    source = CURATOR_SHARED_JS.read_text(encoding="utf-8")

    assert "let seededServerPlaylist = null;" in source
    assert "window.curatorSetSeededPlaylistTarget = setSeededPlaylistTarget;" in source
    assert "replace_playlist_name: seededServerPlaylist.playlistName" in source
    assert "unresolvedTracks" in source
    assert "confirm(message)" in source
    assert "window.curatorReplaceSeededPlaylist = replaceSeededPlaylist;" in source
    assert "localStorage.setItem(STORAGE_KEY, JSON.stringify(workbench))" in source
    assert "JSON.stringify(seededServerPlaylist)" not in source
```

The final persistence assertion protects the existing Workbench-only storage payload; do not add the replacement target to `workbench`.

- [ ] **Step 2: Run the frontend contract tests and verify they fail**

Run:

```powershell
python -m pytest -q test/unit/test_playlist_curator_templates.py
```

Expected: failures for the missing replacement controls, setter, confirmation, and replacement payload.

- [ ] **Step 3: Add desktop and mobile replacement buttons**

In the desktop `.curator-wb-save-card`, keep the existing input/save button and add:

```html
<button type="button" id="curator-wb-replace-btn"
        class="curator-btn hidden" data-kind="secondary" data-full="true" disabled>
    Replace seeded playlist
</button>
```

In `.curator-sheet-save`, after the existing save button, add:

```html
<button type="button" id="curator-sheet-replace-btn"
        class="curator-btn hidden" data-kind="secondary" data-full="true" disabled>
    Replace seeded playlist
</button>
```

- [ ] **Step 4: Add target ownership, rendering, confirmation, and submission**

In `curator-shared.js`, define state beside `workbench`:

```javascript
let seededServerPlaylist = null;

function setSeededPlaylistTarget(target) {
    if (!target || !target.playlistName) {
        seededServerPlaylist = null;
    } else {
        seededServerPlaylist = {
            playlistId: String(target.playlistId || ''),
            playlistName: String(target.playlistName),
            unresolvedTracks: Math.max(0, Number(target.unresolvedTracks) || 0),
        };
    }
    renderWorkbench();
}
window.curatorSetSeededPlaylistTarget = setSeededPlaylistTarget;
```

Update `workbenchClear()` so an explicit clear or successful save also clears the target:

```javascript
function workbenchClear() {
    setSeededPlaylistTarget(null);
    if (workbench.tracks.length === 0) return;
    workbench = { tracks: [] };
    commit({ changedIds: null });
}
```

In both `renderRail()` and `renderSheet()`, obtain the corresponding replacement button and apply this rendering contract (use the correct button variable in each function):

```javascript
if (replaceBtn) {
    const hasReplaceTarget = Boolean(seededServerPlaylist);
    replaceBtn.classList.toggle('hidden', !hasReplaceTarget);
    replaceBtn.disabled = !hasReplaceTarget || total === 0;
    if (hasReplaceTarget) {
        replaceBtn.textContent = `Replace “${seededServerPlaylist.playlistName}”`;
    }
}
```

Replace the current duplicated save request body with these complete functions:

```javascript
async function submitPlaylistSave(payload, successMessage) {
    try {
        const res = await fetch('/api/curator/save_playlist', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || 'Save failed');
        toast(successMessage, 'success');
        workbenchClear();
        return true;
    } catch (e) {
        toast(e.message || 'Save failed', 'error');
        return false;
    }
}

async function savePlaylist(name) {
    const wb = getWorkbench();
    if (!name || !name.trim()) {
        toast('Please enter a playlist name.', 'error');
        return false;
    }
    if (wb.tracks.length === 0) {
        toast('Workbench is empty.', 'error');
        return false;
    }
    const trimmedName = name.trim();
    const trackIds = wb.tracks.map(track => track.item_id);
    return submitPlaylistSave(
        { new_playlist_name: trimmedName, track_ids: trackIds },
        `Saved "${trimmedName}" - ${trackIds.length} tracks`
    );
}

async function replaceSeededPlaylist() {
    const wb = getWorkbench();
    if (!seededServerPlaylist) {
        toast('Choose a media-server playlist seed first.', 'error');
        return false;
    }
    if (wb.tracks.length === 0) {
        toast('Workbench is empty.', 'error');
        return false;
    }

    const count = wb.tracks.length;
    const name = seededServerPlaylist.playlistName;
    let message = `Replace all tracks in "${name}" with ${count} Workbench ${count === 1 ? 'track' : 'tracks'}?`;
    if (seededServerPlaylist.unresolvedTracks > 0) {
        const unresolved = seededServerPlaylist.unresolvedTracks;
        message += `\n\n${unresolved} ${unresolved === 1 ? 'track was' : 'tracks were'} not analyzed and will be removed.`;
    }
    message += '\n\nThis cannot be undone.';
    if (!confirm(message)) return false;

    const trackIds = wb.tracks.map(track => track.item_id);
    return submitPlaylistSave(
        { replace_playlist_name: seededServerPlaylist.playlistName, track_ids: trackIds },
        `Replaced "${name}" - ${trackIds.length} tracks`
    );
}

window.curatorSavePlaylist = savePlaylist;
window.curatorReplaceSeededPlaylist = replaceSeededPlaylist;
```

In `attachWorkbenchHandlers()`, bind both replacement buttons:

```javascript
const railReplaceBtn = document.getElementById('curator-wb-replace-btn');
if (railReplaceBtn) {
    railReplaceBtn.addEventListener('click', replaceSeededPlaylist);
}

const sheetReplaceBtn = document.getElementById('curator-sheet-replace-btn');
if (sheetReplaceBtn) {
    sheetReplaceBtn.addEventListener('click', async () => {
        const ok = await replaceSeededPlaylist();
        if (ok) closeSheet();
    });
}
```

- [ ] **Step 5: Make the second action fit both layouts**

Add to `curator.css`:

```css
.curator-wb-save-card .curator-btn + .curator-btn { margin-top: 8px; }
.curator-sheet-save { flex-wrap: wrap; }
.curator-sheet-save .curator-btn[data-full="true"] { flex-basis: 100%; }
```

- [ ] **Step 6: Run template tests and inspect the focused diff**

Run:

```powershell
python -m pytest -q test/unit/test_playlist_curator_templates.py
git diff --check
```

Expected: template tests pass and `git diff --check` prints no whitespace errors.

- [ ] **Step 7: Commit the shared Workbench controls**

```powershell
git add templates/includes/_curator_workbench.html static/playlist_curator/curator-shared.js static/playlist_curator/curator.css test/unit/test_playlist_curator_templates.py
git commit -m "feat: add seeded playlist replacement controls"
```

### Task 3: Capture the Media-Server Seed for Replacement

**Files:**
- Modify: `static/playlist_curator/curator-extender.js:160-245,574-584`
- Modify: `test/unit/test_playlist_curator_templates.py`

**Interfaces:**
- Consumes: `window.curatorSetSeededPlaylistTarget(target)` from Task 2 and `/api/curator/server_playlist_tracks` response fields `unresolved_tracks` and `tracks`.
- Produces: contextual target `{playlistId, playlistName, unresolvedTracks}` before switching the seed select back to `__workbench__`.

- [ ] **Step 1: Write failing extender seed-state contract tests**

Append to `test/unit/test_playlist_curator_templates.py`:

```python
CURATOR_EXTENDER_JS = REPO_ROOT / "static" / "playlist_curator" / "curator-extender.js"


def test_extender_retains_server_seed_name_and_unresolved_count_for_replacement():
    source = CURATOR_EXTENDER_JS.read_text(encoding="utf-8")

    assert "opt.dataset.playlistName = pl.playlist_name;" in source
    assert "window.curatorSetSeededPlaylistTarget({" in source
    assert "playlistId," in source
    assert "playlistName," in source
    assert "unresolvedTracks: data.unresolved_tracks || 0" in source
    assert source.index("window.curatorSetSeededPlaylistTarget({") < source.index(
        "select.value = SEED_WORKBENCH;"
    )


def test_extender_clears_replacement_target_for_non_server_seed():
    source = CURATOR_EXTENDER_JS.read_text(encoding="utf-8")

    assert "window.curatorSetSeededPlaylistTarget(null);" in source
    assert "seedValue.startsWith('__server__')" in source
```

- [ ] **Step 2: Run the contract tests and verify they fail**

Run:

```powershell
python -m pytest -q test/unit/test_playlist_curator_templates.py
```

Expected: failures because server options do not retain `playlist_name` and the extender never calls the shared target setter.

- [ ] **Step 3: Attach the playlist name to each server option**

In `loadServerPlaylists()`, add the dataset assignment before appending the option:

```javascript
opt.value = '__server__' + pl.playlist_id;
opt.dataset.playlistName = pl.playlist_name;
opt.textContent = `${pl.playlist_name} (${pl.song_count} songs)`;
```

- [ ] **Step 4: Capture or clear the target during seed loading**

At the start of `loadSeedIntoWorkbench(seedValue)`, clear stale state for the new non-Workbench choice, then capture server metadata only after a successful fetch:

```javascript
async function loadSeedIntoWorkbench(seedValue) {
    if (!seedValue || seedValue === SEED_WORKBENCH) return;
    const statusId = 'curator-extender-status';
    const select = document.getElementById('curator-seed-select');
    const selectedOption = select ? select.options[select.selectedIndex] : null;
    const isServerSeed = seedValue.startsWith('__server__');
    window.curatorSetSeededPlaylistTarget(null);

    let tracks = null;
    let serverSeed = null;
    if (isServerSeed) {
        window.curatorSetStatus(statusId, 'Loading playlist tracks…', 'loading');
        const data = await fetchServerPlaylistTracks(seedValue);
        if (!data) return;
        tracks = data.tracks || [];
        const playlistId = seedValue.replace('__server__', '');
        const playlistName = selectedOption ? selectedOption.dataset.playlistName : '';
        serverSeed = {
            playlistId,
            playlistName,
            unresolvedTracks: data.unresolved_tracks || 0,
        };
    } else if (clusterPlaylistsCache[seedValue]) {
        tracks = clusterPlaylistsCache[seedValue];
    }

    if (!tracks || tracks.length === 0) {
        window.curatorSetStatus(statusId, 'Playlist is empty.', 'error');
        return;
    }

    if (serverSeed && serverSeed.playlistName) {
        window.curatorSetSeededPlaylistTarget(serverSeed);
    }
    const added = window.workbenchAddBulk(tracks, 'extend');
    if (select) {
        refreshWorkbenchOption();
        select.value = SEED_WORKBENCH;
    }
    const skipped = tracks.length - added;
    const msg = added > 0
        ? `Loaded ${added} track${added === 1 ? '' : 's'} into Workbench${skipped > 0 ? ` (${skipped} already there)` : ''}. Tune influence on the right →`
        : 'All tracks were already in the Workbench.';
    window.curatorSetStatus(statusId, msg, 'success');
    setTimeout(() => window.curatorSetStatus(statusId, '', ''), 4000);
}
```

In the seed change handler, clear the target when the user returns to the empty choice, but keep it when the programmatic or manual choice is Workbench:

```javascript
const v = seedSelect.value;
if (!v) window.curatorSetSeededPlaylistTarget(null);
if (v && v !== SEED_WORKBENCH) loadSeedIntoWorkbench(v);
```

- [ ] **Step 5: Run all focused curator tests and lint**

Run:

```powershell
python -m pytest -q test/unit/test_playlist_curator_save.py test/unit/test_playlist_curator_templates.py test/unit/test_playlist_curator_security.py test/unit/test_playlist_curator_merge_compat.py
python -m ruff check app_playlist_curator.py test/unit/test_playlist_curator_save.py test/unit/test_playlist_curator_templates.py
git diff --check
```

Expected: all focused tests pass, Ruff reports `All checks passed!`, and the diff check is clean.

- [ ] **Step 6: Commit extender target capture**

```powershell
git add static/playlist_curator/curator-extender.js test/unit/test_playlist_curator_templates.py
git commit -m "feat: retain extender playlist seed target"
```

### Task 4: Verify, Push, and Redeploy the Isolated Test App

**Files:**
- Verify: all files changed in Tasks 1-3
- Preserve: `deployment/docker-compose-nvidia.yaml`
- Use without committing: `C:\tmp\playlist-curator-test.compose.yaml`

**Interfaces:**
- Consumes: the completed API/UI behavior and the existing `playlist-curator-test` Compose project.
- Produces: a pushed PR head, green checks, and a healthy isolated test app at `http://localhost:18001/playlist_curator/extender` with no test worker or GPU request.

- [ ] **Step 1: Run final focused verification**

```powershell
python -m pytest -q test/unit/test_playlist_curator_save.py test/unit/test_playlist_curator_templates.py test/unit/test_playlist_curator_security.py test/unit/test_playlist_curator_merge_compat.py
python -m ruff check app_playlist_curator.py test/unit/test_playlist_curator_save.py test/unit/test_playlist_curator_templates.py
git diff --check
git status --short
```

Expected: tests and lint pass; the only unrelated unstaged path is `deployment/docker-compose-nvidia.yaml`.

- [ ] **Step 2: Push the feature branch and inspect PR 417**

```powershell
git push origin feature/playlist-curator-port
gh pr checks 417
```

Expected: push succeeds. Wait for checks to settle; no check may remain failing or pending before completion.

- [ ] **Step 3: Capture the production worker identity before deployment**

Read and retain, without printing secrets:

```powershell
$prodBefore = (docker inspect 'audiomuse-ai-worker-instance' | ConvertFrom-Json)[0]
if ($prodBefore.State.Status -ne 'running') { throw "Production worker is not running" }
$baseline = [pscustomobject]@{
    Id = $prodBefore.Id
    StartedAt = $prodBefore.State.StartedAt
    Networks = @($prodBefore.NetworkSettings.Networks.PSObject.Properties.Name)
}
$json = $baseline | ConvertTo-Json -Compress
[IO.File]::WriteAllText('C:\tmp\playlist-curator-production-baseline.json', $json)
```

Expected: production worker status is `running`.

- [ ] **Step 4: Build the CPU-only test image**

```powershell
docker build --build-arg BASE_IMAGE=ubuntu:24.04 -t audiomuse-ai-playlist-curator:test .
```

Expected: build succeeds and `docker image inspect audiomuse-ai-playlist-curator:test` shows no `NVIDIA_*` image environment variables.

- [ ] **Step 5: Recreate only the isolated test web container**

Load the stopped `audiomuse-test-flask-app` environment into the current process without printing values, then recreate only the web service:

```powershell
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
    Set-Item -Path "Env:$key" -Value $sourceEnv[$key]
}
docker compose -p playlist-curator-test -f C:\tmp\playlist-curator-test.compose.yaml up -d --no-deps --force-recreate audiomuse-ai-flask
```

Expected: only `playlist-curator-test-flask` is recreated; Postgres, Redis, and production containers retain their identities.

- [ ] **Step 6: Verify container health and isolation**

```powershell
$test = (docker inspect 'playlist-curator-test-flask' | ConvertFrom-Json)[0]
$deviceRequestCount = if ($null -eq $test.HostConfig.DeviceRequests) { 0 } else { @($test.HostConfig.DeviceRequests).Count }
$testNetworks = @($test.NetworkSettings.Networks.PSObject.Properties.Name)
$testProjectContainers = @(docker ps -a --filter 'label=com.docker.compose.project=playlist-curator-test' --format '{{.Names}}')
$testWorkers = @($testProjectContainers | Where-Object { $_ -match 'worker' })

if ($test.State.Health.Status -ne 'healthy') { throw "Test web is not healthy" }
if ($deviceRequestCount -ne 0) { throw "Test web requested a device" }
if (($testNetworks -join ',') -ne 'playlist-curator-test_default') { throw "Unexpected test network" }
if ($testWorkers.Count -ne 0) { throw "Unexpected test worker" }
```

Expected: healthy test web, zero device requests, one isolated network, and zero workers.

- [ ] **Step 7: Verify the UI without replacing an external playlist**

Open `http://localhost:18001/playlist_curator/extender`. Confirm:

- the create-new input and button remain visible;
- the replacement buttons are initially hidden;
- calling `window.curatorSetSeededPlaylistTarget({playlistId: 'test', playlistName: 'Test Seed', unresolvedTracks: 2})` in the local page context makes both replacement controls contextual and enabled when the Workbench has a track;
- intercepting `window.confirm` to return `false` and clicking replacement shows the warning path without issuing a network save;
- refreshing the page clears the replacement target while leaving the persisted Workbench intact.

Expected: the contextual UI behaves as designed and no media-server playlist is mutated during browser verification.

- [ ] **Step 8: Re-audit production and PR state**

```powershell
$baseline = Get-Content -LiteralPath 'C:\tmp\playlist-curator-production-baseline.json' -Raw | ConvertFrom-Json
$prodAfter = (docker inspect 'audiomuse-ai-worker-instance' | ConvertFrom-Json)[0]
if ($prodAfter.Id -ne $baseline.Id) { throw "Production worker identity changed" }
if ($prodAfter.State.StartedAt -ne $baseline.StartedAt) { throw "Production worker restarted" }
if ($prodAfter.State.Status -ne 'running') { throw "Production worker is not running" }
if ((@($prodAfter.NetworkSettings.Networks.PSObject.Properties.Name) -join ',') -ne (@($baseline.Networks) -join ',')) { throw "Production worker network changed" }

gh pr checks 417
git status --short --branch
```

Expected: production is unchanged; PR checks have zero failures/pending checks; branch matches its remote; the Docker Compose user edit remains untouched and unstaged.
