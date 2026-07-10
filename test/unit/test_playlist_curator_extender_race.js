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

const EXTENDER_JS = path.resolve(
    __dirname,
    '../../static/playlist_curator/curator-extender.js',
);

function deferred() {
    let resolve;
    const promise = new Promise(resolvePromise => {
        resolve = resolvePromise;
    });
    return { promise, resolve };
}

function jsonResponse(body) {
    return {
        ok: true,
        async json() {
            return body;
        },
    };
}

async function flushAsyncWork() {
    await new Promise(resolve => setImmediate(resolve));
    await new Promise(resolve => setImmediate(resolve));
}

async function createHarness() {
    const documentListeners = new Map();
    const selectListeners = new Map();
    const targetWrites = [];
    const addedBatches = [];
    let workbenchTracks = [];
    let workbenchOption = null;

    const select = {
        value: '',
        selectedIndex: 0,
        options: [{ dataset: { playlistName: 'Server playlist' } }],
        firstChild: null,
        addEventListener(type, handler) {
            selectListeners.set(type, handler);
        },
        appendChild() {},
        insertBefore(option) {
            workbenchOption = option;
            this.firstChild = option;
        },
        querySelector(selector) {
            return selector === 'option[value="__workbench__"]' ? workbenchOption : null;
        },
        querySelectorAll() {
            return [];
        },
    };

    const resultsSection = {
        classList: {
            add() {},
        },
    };

    const document = {
        readyState: 'loading',
        addEventListener(type, handler) {
            documentListeners.set(type, handler);
        },
        createElement(tagName) {
            const element = {
                tagName,
                children: [],
                dataset: {},
                value: '',
                textContent: '',
                appendChild(child) {
                    this.children.push(child);
                },
                remove() {},
            };
            return element;
        },
        getElementById(id) {
            if (id === 'curator-seed-select') return select;
            if (id === 'curator-results-section') return resultsSection;
            return null;
        },
    };

    const initialResponses = new Map([
        ['/api/curator/filter_options', {}],
        ['/api/playlists', { Cluster: [{ item_id: 'cluster-track' }] }],
        ['/api/curator/server_playlists', []],
    ]);

    const context = {
        clearTimeout() {},
        console,
        document,
        fetch(url) {
            if (!initialResponses.has(url)) {
                throw new Error(`Unexpected fetch during init: ${url}`);
            }
            return Promise.resolve(jsonResponse(initialResponses.get(url)));
        },
        requestAnimationFrame(callback) {
            callback();
        },
        setTimeout() {
            return 0;
        },
        window: {
            CURATOR_ICONS: {},
            curatorSetSeededPlaylistTarget(target) {
                targetWrites.push(target);
            },
            curatorSetStatus() {},
            escHtml: String,
            getInfluenceInfo() {
                return {};
            },
            getWorkbench() {
                return { tracks: workbenchTracks };
            },
            workbenchAddBulk(tracks) {
                addedBatches.push(Array.from(tracks, track => track.item_id));
                workbenchTracks = [...workbenchTracks, ...tracks];
                return tracks.length;
            },
        },
    };

    const source = fs.readFileSync(EXTENDER_JS, 'utf8');
    vm.runInNewContext(source, context, { filename: EXTENDER_JS }); // NOSONAR -- Executes a checked-in script in an isolated test VM.

    const initialize = documentListeners.get('DOMContentLoaded');
    assert.equal(typeof initialize, 'function');
    await initialize();

    const changeSeed = selectListeners.get('change');
    assert.equal(typeof changeSeed, 'function');

    return {
        addedBatches,
        changeSeed,
        context,
        select,
        targetWrites,
    };
}

async function startServerLoad(harness) {
    const pendingResponse = deferred();
    harness.context.fetch = url => {
        assert.equal(url, '/api/curator/server_playlist_tracks');
        return pendingResponse.promise;
    };
    harness.select.value = '__server__server-1';
    harness.select.selectedIndex = 0;
    harness.changeSeed();
    return pendingResponse;
}

test('empty selection invalidates an in-flight server seed load', async () => {
    const harness = await createHarness();
    const pendingResponse = await startServerLoad(harness);

    harness.select.value = '';
    harness.changeSeed();

    pendingResponse.resolve(jsonResponse({
        tracks: [{ item_id: 'server-track' }],
        unresolved_tracks: 2,
    }));
    await flushAsyncWork();

    assert.deepEqual(harness.targetWrites, [null, null]);
    assert.deepEqual(harness.addedBatches, []);
    assert.equal(harness.select.value, '');
});

test('cluster selection invalidates an in-flight server seed load', async () => {
    const harness = await createHarness();
    const pendingResponse = await startServerLoad(harness);

    harness.select.value = 'Cluster';
    harness.changeSeed();

    pendingResponse.resolve(jsonResponse({
        tracks: [{ item_id: 'server-track' }],
        unresolved_tracks: 2,
    }));
    await flushAsyncWork();

    assert.deepEqual(harness.targetWrites, [null, null]);
    assert.deepEqual(harness.addedBatches, [['cluster-track']]);
    assert.equal(harness.select.value, '__workbench__');
});

test('current server seed load completes normally', async () => {
    const harness = await createHarness();
    const pendingResponse = await startServerLoad(harness);

    pendingResponse.resolve(jsonResponse({
        tracks: [{ item_id: 'server-track' }],
        unresolved_tracks: 2,
    }));
    await flushAsyncWork();

    assert.equal(harness.targetWrites.length, 2);
    assert.equal(harness.targetWrites[0], null);
    assert.equal(harness.targetWrites[1].playlistId, 'server-1');
    assert.equal(harness.targetWrites[1].playlistName, 'Server playlist');
    assert.equal(harness.targetWrites[1].unresolvedTracks, 2);
    assert.deepEqual(harness.addedBatches, [['server-track']]);
    assert.equal(harness.select.value, '__workbench__');
});

test('Workbench selection invalidates an in-flight server seed load', async () => {
    const harness = await createHarness();
    const pendingResponse = await startServerLoad(harness);

    harness.select.value = '__workbench__';
    harness.changeSeed();

    pendingResponse.resolve(jsonResponse({
        tracks: [{ item_id: 'server-track' }],
        unresolved_tracks: 2,
    }));
    await flushAsyncWork();

    assert.deepEqual(harness.targetWrites, [null]);
    assert.deepEqual(harness.addedBatches, []);
    assert.equal(harness.select.value, '__workbench__');
});
