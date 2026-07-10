/*
 * AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
 * Copyright (C) 2025 NeptuneHub
 * SPDX-License-Identifier: AGPL-3.0-only
 */

'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { test } = require('node:test');

const SHARED_JS = path.resolve(
    __dirname,
    '../../static/playlist_curator/curator-shared.js',
);
const STORAGE_KEY = 'audiomuse:curator:workbench';

function deferred() {
    let resolve;
    const promise = new Promise(resolvePromise => {
        resolve = resolvePromise;
    });
    return { promise, resolve };
}

function response(body, ok = true) {
    return {
        ok,
        async json() {
            return body;
        },
    };
}

function classList(initial = []) {
    const values = new Set(initial);
    return {
        add(...names) {
            names.forEach(name => values.add(name));
        },
        remove(...names) {
            names.forEach(name => values.delete(name));
        },
        toggle(name, force) {
            if (force === undefined) {
                if (values.has(name)) values.delete(name);
                else values.add(name);
                return values.has(name);
            }
            if (force) values.add(name);
            else values.delete(name);
            return force;
        },
        contains(name) {
            return values.has(name);
        },
    };
}

function element(id, classes = []) {
    const listeners = new Map();
    return {
        id,
        className: '',
        classList: classList(classes),
        dataset: {},
        disabled: false,
        innerHTML: '',
        style: {},
        textContent: '',
        value: '',
        addEventListener(type, handler) {
            if (!listeners.has(type)) listeners.set(type, []);
            listeners.get(type).push(handler);
        },
        appendChild() {},
        remove() {},
    };
}

function createHarness() {
    const documentListeners = new Map();
    const windowListeners = new Map();
    const storage = new Map();
    const fetchCalls = [];

    const ids = [
        'curator-wb-list',
        'curator-wb-total',
        'curator-wb-from-search',
        'curator-wb-from-extend',
        'curator-wb-save-btn',
        'curator-wb-replace-btn',
        'curator-wb-finddups-btn',
        'curator-wb-clear-btn',
        'curator-wb-name',
        'curator-workbench-sheet',
        'curator-sheet-handle',
        'curator-sheet-title',
        'curator-sheet-sub',
        'curator-sheet-list',
        'curator-sheet-save-btn',
        'curator-sheet-replace-btn',
        'curator-sheet-finddups-btn',
        'curator-sheet-clear-btn',
        'curator-sheet-name',
        'curator-sheet-backdrop',
    ];
    const elements = new Map(ids.map(id => [
        id,
        element(id, id.endsWith('replace-btn') ? ['hidden'] : []),
    ]));

    const document = {
        readyState: 'complete',
        body: {
            dataset: { curatorPage: 'search' },
            appendChild(node) {
                if (node.id) elements.set(node.id, node);
            },
        },
        addEventListener(type, handler) {
            if (!documentListeners.has(type)) documentListeners.set(type, []);
            documentListeners.get(type).push(handler);
        },
        createElement() {
            return element('');
        },
        dispatchEvent(event) {
            for (const handler of documentListeners.get(event.type) || []) handler(event);
            return true;
        },
        getElementById(id) {
            return elements.get(id) || null;
        },
        querySelectorAll() {
            return [];
        },
    };

    const window = {
        addEventListener(type, handler) {
            if (!windowListeners.has(type)) windowListeners.set(type, []);
            windowListeners.get(type).push(handler);
        },
    };

    const context = {
        CustomEvent: class CustomEvent {
            constructor(type, options = {}) {
                this.type = type;
                this.detail = options.detail;
            }
        },
        clearTimeout() {},
        confirm() {
            return true;
        },
        console,
        document,
        fetch(url, options) {
            fetchCalls.push({ url, options });
            throw new Error('Set harness fetch response before saving');
        },
        localStorage: {
            getItem(key) {
                return storage.has(key) ? storage.get(key) : null;
            },
            setItem(key, value) {
                storage.set(key, value);
            },
        },
        setTimeout() {
            return 1;
        },
        window,
    };

    const source = fs.readFileSync(SHARED_JS, 'utf8');
    vm.runInNewContext(source, context, { filename: SHARED_JS });

    return {
        buttons: [
            elements.get('curator-wb-save-btn'),
            elements.get('curator-wb-replace-btn'),
            elements.get('curator-sheet-save-btn'),
            elements.get('curator-sheet-replace-btn'),
        ],
        context,
        elements,
        fetchCalls,
        setExternalWorkbench(tracks) {
            storage.set(STORAGE_KEY, JSON.stringify({ tracks }));
            for (const handler of windowListeners.get('storage') || []) {
                handler({ key: STORAGE_KEY });
            }
        },
        setFetchResponse(promise) {
            context.fetch = (url, options) => {
                fetchCalls.push({ url, options });
                return promise;
            };
        },
    };
}

function track(itemId) {
    return { item_id: itemId, title: itemId, author: 'Artist' };
}

function prepareSaveableWorkbench(harness) {
    harness.elements.get('curator-wb-name').value = 'New Mix';
    harness.elements.get('curator-sheet-name').value = 'New Mix';
    harness.context.window.workbenchAdd(track('track-1'), 'search');
    harness.context.window.curatorSetSeededPlaylistTarget({
        playlistId: 'seed-1',
        playlistName: 'Road Trip',
        unresolvedTracks: 0,
    });
    harness.context.window.renderWorkbench();
}

test('external clear followed by repopulation cannot reactivate an old replacement target', () => {
    const harness = createHarness();
    prepareSaveableWorkbench(harness);

    harness.setExternalWorkbench([]);
    harness.setExternalWorkbench([track('other-tab-track')]);

    for (const id of ['curator-wb-replace-btn', 'curator-sheet-replace-btn']) {
        const button = harness.elements.get(id);
        assert.equal(button.classList.contains('hidden'), true);
        assert.equal(button.disabled, true);
    }
});

test('save is single-flight and disables every create and replace control while pending', async () => {
    const harness = createHarness();
    prepareSaveableWorkbench(harness);
    const pending = deferred();
    harness.setFetchResponse(pending.promise);

    const firstSave = harness.context.window.curatorSavePlaylist('New Mix');
    const secondSave = harness.context.window.curatorSavePlaylist('Second Mix');

    assert.equal(harness.fetchCalls.length, 1);
    for (const button of harness.buttons) assert.equal(button.disabled, true);

    pending.resolve(response({ action: 'created' }));
    await firstSave;
    assert.equal(await secondSave, false);
});

test('ordinary successful save clears the submitted Workbench and replacement target', async () => {
    const harness = createHarness();
    prepareSaveableWorkbench(harness);
    harness.setFetchResponse(Promise.resolve(response({ action: 'created' })));

    assert.equal(await harness.context.window.curatorSavePlaylist('New Mix'), true);

    assert.deepEqual(Array.from(harness.context.window.getWorkbench().tracks), []);
    for (const id of ['curator-wb-replace-btn', 'curator-sheet-replace-btn']) {
        assert.equal(harness.elements.get(id).classList.contains('hidden'), true);
    }
});

test('failed save restores every create and replace control', async () => {
    const harness = createHarness();
    prepareSaveableWorkbench(harness);
    const pending = deferred();
    harness.setFetchResponse(pending.promise);

    const save = harness.context.window.curatorSavePlaylist('New Mix');
    pending.resolve(response({ error: 'Provider failed' }, false));

    assert.equal(await save, false);
    for (const button of harness.buttons) assert.equal(button.disabled, false);
});

test('successful save preserves edits and target made after the submitted snapshot', async () => {
    const harness = createHarness();
    prepareSaveableWorkbench(harness);
    const pending = deferred();
    harness.setFetchResponse(pending.promise);

    const save = harness.context.window.curatorSavePlaylist('New Mix');
    harness.context.window.workbenchAdd(track('track-2'), 'search');
    pending.resolve(response({ action: 'created' }));

    assert.equal(await save, true);
    assert.deepEqual(
        Array.from(harness.context.window.getWorkbench().tracks, item => item.item_id),
        ['track-1', 'track-2'],
    );
    assert.equal(
        harness.elements.get('curator-wb-replace-btn').classList.contains('hidden'),
        false,
    );
    assert.deepEqual(
        JSON.parse(harness.fetchCalls[0].options.body).track_ids,
        ['track-1'],
    );
});

test('successful save preserves externally synchronized tracks while pending', async () => {
    const harness = createHarness();
    prepareSaveableWorkbench(harness);
    const pending = deferred();
    harness.setFetchResponse(pending.promise);

    const save = harness.context.window.curatorSavePlaylist('New Mix');
    harness.setExternalWorkbench([track('other-tab-track')]);
    pending.resolve(response({ action: 'created' }));

    assert.equal(await save, true);
    assert.deepEqual(
        Array.from(harness.context.window.getWorkbench().tracks, item => item.item_id),
        ['other-tab-track'],
    );
    assert.equal(
        harness.elements.get('curator-wb-replace-btn').classList.contains('hidden'),
        true,
    );
});
