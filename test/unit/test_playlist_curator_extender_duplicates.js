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

function classList() {
    const values = new Set();
    return {
        add(...names) { names.forEach(name => values.add(name)); },
        remove(...names) { names.forEach(name => values.delete(name)); },
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
        contains(name) { return values.has(name); },
    };
}

function element(id = '', tagName = 'div') {
    const listeners = new Map();
    const children = [];
    const node = {
        id,
        tagName,
        children,
        options: [],
        selectedIndex: 0,
        className: '',
        classList: classList(),
        dataset: {},
        disabled: false,
        innerHTML: '',
        isConnected: true,
        max: '',
        min: '',
        style: {},
        textContent: '',
        value: '',
        addEventListener(type, handler) {
            if (!listeners.has(type)) listeners.set(type, []);
            listeners.get(type).push(handler);
        },
        appendChild(child) {
            children.push(child);
            if (tagName === 'select') node.options.push(child);
            return child;
        },
        insertAdjacentHTML(_position, html) { node.innerHTML += html; },
        querySelector() { return null; },
        querySelectorAll() { return []; },
        remove() {},
        async dispatch(type) {
            for (const handler of listeners.get(type) || []) {
                await handler({ target: node, preventDefault() {} });
            }
        },
        async click() { await node.dispatch('click'); },
    };
    return node;
}

function response(body, ok = true) {
    return { ok, async json() { return body; } };
}

async function createHarness(initialResults) {
    const documentListeners = new Map();
    const duplicateCalls = [];
    const workbenchAdds = [];
    let searchResults = initialResults;

    const ids = [
        'curator-seed-select',
        'curator-seed-status',
        'curator-tune-toggle',
        'curator-tune-grid',
        'curator-similarity',
        'curator-similarity-value',
        'curator-min-rating',
        'curator-min-rating-value',
        'curator-max-songs',
        'curator-year-min',
        'curator-year-max',
        'curator-year-value',
        'curator-year-track',
        'curator-dup-mode',
        'curator-dup-threshold',
        'curator-dup-threshold-value',
        'curator-dup-sens',
        'curator-extender-run',
        'curator-extender-status',
        'curator-extender-finddups',
        'curator-results-section',
        'curator-results-count',
        'curator-results-table-wrap',
        'curator-results-cards',
    ];
    const elements = new Map(ids.map(id => [id, element(id)]));
    const seedSelect = elements.get('curator-seed-select');
    let workbenchOption = null;
    seedSelect.value = '__workbench__';
    seedSelect.querySelector = selector => (
        selector === 'option[value="__workbench__"]' ? workbenchOption : null
    );
    seedSelect.querySelectorAll = () => [];
    seedSelect.insertBefore = option => {
        workbenchOption = option;
        seedSelect.options.unshift(option);
        return option;
    };

    elements.get('curator-similarity').value = '0.5';
    elements.get('curator-min-rating').value = '0';
    elements.get('curator-max-songs').value = '50';
    elements.get('curator-year-min').min = '1900';
    elements.get('curator-year-min').max = '2030';
    elements.get('curator-year-min').value = '1900';
    elements.get('curator-year-max').min = '1900';
    elements.get('curator-year-max').max = '2030';
    elements.get('curator-year-max').value = '2030';
    elements.get('curator-dup-mode').value = 'mark';
    elements.get('curator-dup-threshold').value = '0.05';
    elements.get('curator-dup-threshold').max = '0.1';
    elements.get('curator-results-section').classList.add('hidden');

    const document = {
        readyState: 'loading',
        addEventListener(type, handler) {
            if (!documentListeners.has(type)) documentListeners.set(type, []);
            documentListeners.get(type).push(handler);
        },
        createElement(tagName) { return element('', tagName); },
        getElementById(id) { return elements.get(id) || null; },
        querySelector() { return null; },
    };

    const window = {
        CURATOR_ICONS: { check: '', plus: '', play: '', warn: '', x: '' },
        curatorFindDuplicatesForTracks(...args) { duplicateCalls.push(args); },
        curatorSetSeededPlaylistTarget() {},
        curatorSetStatus() {},
        escHtml: String,
        getInfluenceInfo() { return { label: '', tip: '' }; },
        getWorkbench() {
            return { tracks: [{ item_id: 'seed-track', influence: 0 }] };
        },
        matchMedia() { return { matches: false }; },
        workbenchAdd(track) { workbenchAdds.push(track.item_id); },
        workbenchAddBulk() { return 0; },
        workbenchGetInfluence() { return 0; },
        workbenchHas() { return false; },
    };

    const context = {
        CSS: { escape: String },
        clearTimeout() {},
        console,
        document,
        encodeURIComponent,
        fetch(url) {
            if (url === '/api/curator/filter_options') return Promise.resolve(response({}));
            if (url === '/api/playlists') return Promise.resolve(response({}));
            if (url === '/api/curator/server_playlists') return Promise.resolve(response([]));
            if (url === '/api/curator/search') {
                return Promise.resolve(response({ results: searchResults }));
            }
            throw new Error(`Unexpected fetch: ${url}`);
        },
        requestAnimationFrame(callback) { callback(); },
        setTimeout() { return 0; },
        window,
    };

    const source = fs.readFileSync(EXTENDER_JS, 'utf8');
    vm.runInNewContext(source, context, { filename: EXTENDER_JS }); // NOSONAR -- Executes a checked-in script in an isolated test VM.
    for (const handler of documentListeners.get('DOMContentLoaded') || []) await handler();

    return {
        context,
        duplicateCalls,
        elements,
        async changeSeed(value) {
            seedSelect.value = value;
            await seedSelect.dispatch('change');
        },
        setSearchResults(results) { searchResults = results; },
        workbenchAdds,
    };
}

function tracks(count) {
    return Array.from({ length: count }, (_value, index) => ({
        item_id: `track-${index + 1}`,
        title: `Track ${index + 1}`,
        author: 'Artist',
    }));
}

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
    assert.doesNotMatch(
        harness.elements.get('curator-results-cards').innerHTML,
        /data-row-id="track-2"/,
    );
    assert.deepEqual(harness.workbenchAdds, []);

    await harness.elements.get('curator-extender-run').click();
    assert.match(
        harness.elements.get('curator-results-cards').innerHTML,
        /data-row-id="track-2"/,
    );
});

test('Extender seed change clears manually hidden candidates', async () => {
    const results = tracks(3);
    const harness = await createHarness(results);
    await harness.elements.get('curator-extender-run').click();
    harness.context.window.curatorHideSearchDuplicates(['track-2']);

    await harness.changeSeed('__workbench__');
    await harness.elements.get('curator-extender-run').click();
    assert.match(
        harness.elements.get('curator-results-cards').innerHTML,
        /data-row-id="track-2"/,
    );
});

test('Extender scan excludes candidates hidden by automatic duplicate mode', async () => {
    const results = tracks(3);
    results[1].duplicate_of = { distance: 0.01 };
    const harness = await createHarness(results);
    harness.elements.get('curator-dup-mode').value = 'hide';
    await harness.elements.get('curator-extender-run').click();
    await harness.elements.get('curator-extender-finddups').click();

    assert.deepEqual(
        Array.from(harness.duplicateCalls[0][0], track => track.item_id),
        ['track-1', 'track-3'],
    );
});
