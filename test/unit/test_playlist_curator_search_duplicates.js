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

const SEARCH_JS = path.resolve(
    __dirname,
    '../../static/playlist_curator/curator-search.js',
);

function classList(initial = []) {
    const values = new Set(initial);
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
        querySelector(selector) {
            if (selector === 'tbody') return null;
            if (selector === '.filter-value') {
                return children.find(child => child.className === 'filter-value') || null;
            }
            return null;
        },
        remove() {},
        async click() {
            for (const handler of listeners.get('click') || []) {
                await handler({ target: node, preventDefault() {} });
            }
        },
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
        'curator-filter-rows',
        'curator-match-mode',
        'curator-add-rule',
        'curator-search-run',
        'curator-search-clear',
        'curator-search-send',
        'curator-search-finddups',
        'curator-search-status',
        'curator-results-section',
        'curator-results-count',
        'curator-results-total',
        'curator-results-skipped',
        'curator-results-table-wrap',
        'curator-results-cards',
        'curator-pagination',
        'curator-page-indicator',
        'curator-page-prev',
        'curator-page-next',
        'curator-page-loadall',
        'curator-page-loadall-cancel',
        'curator-loadall-status',
    ];
    const elements = new Map(ids.map(id => [id, element(id)]));
    elements.get('curator-match-mode').value = 'all';
    elements.get('curator-results-section').classList.add('hidden');

    const filterRow = {
        querySelector(selector) {
            if (selector === '.filter-field') return { value: 'artist' };
            if (selector === '.filter-operator') return { value: 'contains' };
            if (selector === '.filter-value') return { value: 'Artist' };
            return null;
        },
    };

    const document = {
        readyState: 'loading',
        addEventListener(type, handler) {
            if (!documentListeners.has(type)) documentListeners.set(type, []);
            documentListeners.get(type).push(handler);
        },
        createElement(tagName) { return element('', tagName); },
        getElementById(id) { return elements.get(id) || null; },
        querySelector() { return null; },
        querySelectorAll(selector) {
            if (selector === '#curator-filter-rows .curator-filter-row') return [filterRow];
            return [];
        },
    };

    const window = {
        CURATOR_ICONS: { check: '', plus: '', play: '', x: '' },
        curatorFindDuplicatesForTracks(...args) { duplicateCalls.push(args); },
        curatorSetStatus() {},
        curatorToast() {},
        location: { href: '' },
        workbenchAdd(track) { workbenchAdds.push(track.item_id); },
        workbenchAddBulk() { return 0; },
        workbenchHas() { return false; },
    };

    const context = {
        CSS: { escape: String },
        console,
        document,
        encodeURIComponent,
        fetch(url) {
            if (url === '/api/curator/filter_options') return Promise.resolve(response({}));
            if (url === '/api/curator/search') {
                return Promise.resolve(response({
                    results: searchResults,
                    total: searchResults.length,
                    page: 1,
                    per_page: 500,
                }));
            }
            throw new Error(`Unexpected fetch: ${url}`);
        },
        requestAnimationFrame(callback) { callback(); },
        window,
    };
    window.escHtml = String;

    const source = fs.readFileSync(SEARCH_JS, 'utf8');
    vm.runInNewContext(source, context, { filename: SEARCH_JS }); // NOSONAR -- Executes a checked-in script in an isolated test VM.
    for (const handler of documentListeners.get('DOMContentLoaded') || []) await handler();

    return {
        context,
        duplicateCalls,
        elements,
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

test('Search Results duplicate scan preserves order and caps the request at 500', async () => {
    const results = tracks(502);
    const harness = await createHarness(results);
    await harness.elements.get('curator-search-run').click();
    await harness.elements.get('curator-search-finddups').click();

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

test('hiding Search Results duplicates is presentation-only and resets on new search', async () => {
    const results = tracks(3);
    const harness = await createHarness(results);
    await harness.elements.get('curator-search-run').click();

    harness.context.window.curatorHideSearchDuplicates(['track-2']);
    assert.doesNotMatch(
        harness.elements.get('curator-results-cards').innerHTML,
        /data-row-id="track-2"/,
    );
    assert.deepEqual(harness.workbenchAdds, []);

    harness.setSearchResults(results);
    await harness.elements.get('curator-search-run').click();
    assert.match(
        harness.elements.get('curator-results-cards').innerHTML,
        /data-row-id="track-2"/,
    );
});
