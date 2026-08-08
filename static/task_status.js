/* AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
 * Copyright (C) 2025 NeptuneHub
 * SPDX-License-Identifier: AGPL-3.0-only
 *
 * This program is free software: you can redistribute it and/or modify it under
 * the terms of the GNU Affero General Public License v3.0. See the LICENSE file
 * in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>
 *
 * The one task-status vocabulary the whole frontend reads.
 *
 * The backend can emit exactly five values - NEW, RUNNING, SUCCESS, FAIL,
 * REVOKED - and config.py says so in one place. The frontend used to re-implement
 * that vocabulary as a hardcoded array in five different files, each of which had
 * grown to seven or more entries by UNIONING the current spellings with older ones
 * ('FINISHED', 'FAILED', 'CANCELED', 'PENDING', 'STARTED', 'PROGRESS', 'FAILURE',
 * and lowercase 'finished'/'failed'). Every one of those is now unreachable, but a
 * reader cannot tell a dead branch from a live one, so the next status added became
 * a twelfth edit across five files rather than one.
 *
 * The risk that creates is silent and page-specific: a terminal check that misses
 * a spelling leaves the page polling a finished task forever behind a frozen
 * progress bar, and the cleaning page went further and treated a final status it
 * could not read as SUCCESS - reporting a clean run for a task that may have
 * failed. Both are now one function that cannot disagree with itself.
 *
 * Note the migration SESSION status ('completed' / 'failed' / 'dry_run_ready') is
 * a genuinely different vocabulary from task status and is deliberately NOT
 * handled here.
 *
 * Task status and worker liveness are different domains and get different mappers
 * here rather than one if-chain that conflates them - a worker answers busy/idle,
 * which is not a task status.
 */
(function () {
    var NEW = 'NEW';
    var RUNNING = 'RUNNING';
    var SUCCESS = 'SUCCESS';
    var FAIL = 'FAIL';
    var REVOKED = 'REVOKED';

    var LIVE = [NEW, RUNNING];
    var TERMINAL = [SUCCESS, FAIL, REVOKED];

    function normalize(status) {
        return String(status === null || status === undefined ? '' : status).toUpperCase();
    }

    function isLive(status) {
        return LIVE.indexOf(normalize(status)) !== -1;
    }

    function isTerminal(status) {
        return TERMINAL.indexOf(normalize(status)) !== -1;
    }

    function isSuccess(status) {
        return normalize(status) === SUCCESS;
    }

    function isFailure(status) {
        return normalize(status) === FAIL;
    }

    function isRevoked(status) {
        return normalize(status) === REVOKED;
    }

    function badgeClass(status) {
        switch (normalize(status)) {
            case SUCCESS: return 'badge-success';
            case FAIL: return 'badge-failure';
            case NEW: return 'badge-pending';
            case RUNNING: return 'badge-started';
            case REVOKED: return 'badge-revoked';
            default: return 'badge-unknown';
        }
    }

    function statusClass(status) {
        var value = normalize(status);
        if (value === SUCCESS) { return 'status-success'; }
        if (value === FAIL || value === REVOKED) { return 'status-failure'; }
        if (value === 'IDLE') { return 'status-idle'; }
        return 'status-pending';
    }

    function title(status) {
        var value = normalize(status);
        if (value === SUCCESS) { return 'Task Completed'; }
        if (value === FAIL) { return 'Task Failed'; }
        if (value === REVOKED) { return 'Task Canceled'; }
        return 'Task Update';
    }

    function workerStateClass(state) {
        var value = normalize(state);
        if (value === 'BUSY') { return 'badge-progress'; }
        if (value === 'IDLE') { return 'badge-ok'; }
        return 'badge-unknown';
    }

    window.AudioMuseTaskStatus = {
        NEW: NEW,
        RUNNING: RUNNING,
        SUCCESS: SUCCESS,
        FAIL: FAIL,
        REVOKED: REVOKED,
        LIVE: LIVE,
        TERMINAL: TERMINAL,
        normalize: normalize,
        isLive: isLive,
        isTerminal: isTerminal,
        isSuccess: isSuccess,
        isFailure: isFailure,
        isRevoked: isRevoked,
        badgeClass: badgeClass,
        statusClass: statusClass,
        title: title,
        workerStateClass: workerStateClass
    };
})();
