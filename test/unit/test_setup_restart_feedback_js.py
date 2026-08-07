# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Structural guards for the shared restart wait and the setup wizard's feedback.

There is no javascript runner in this repo, so the behaviours that were silently
lost once are pinned by reading the sources - but reading them as CODE, not as
text. A substring search over a whole file still passes when the handler that was
supposed to contain the string has been deleted and the string survives in a
comment or in an unrelated function, which is exactly the failure this file used
to be blind to. So every check here first blanks comments and then walks matching
braces to isolate the one function body or `if` branch that owns the behaviour,
and asserts inside that block only.

Main Features:
* Comments are blanked before anything is matched, so prose can never satisfy a check
* Blocks are located by brace matching, so a check fails when its handler is deleted
* The wizard consumes the partial-save fields inside the branch that guards on them
* The partial-save warning never lands in the element the countdown rewrites
* waitAndGo observes `until` when it is called, never from inside the tick
* A rejected `until` returns out of the tick before anything renders
* A resolved `until` reaches finish() only through the floorReached guard
* `restartScheduled: false` only picks the wording inside the countdown and only
  skips the probe; the redirect keeps its cache buster
"""

import os
import re

REPO_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
)

_QUOTES = '\'"`'
_REGEX_PRECEDERS = set('(,=:[!&|?{};+-*%^<>~')


def _read(rel_path):
    with open(os.path.join(REPO_ROOT, rel_path), encoding='utf-8') as handle:
        return handle.read()


def _skip_literal(source, start):
    closer = source[start]
    index = start + 1
    size = len(source)
    while index < size:
        char = source[index]
        index += 1
        if char == '\\':
            index += 1
            continue
        if char == closer:
            break
    return index


def _blank_comments(source):
    out = []
    index = 0
    previous = ''
    size = len(source)
    while index < size:
        char = source[index]
        following = source[index + 1:index + 2]
        if char == '/' and following == '/':
            while index < size and source[index] != '\n':
                out.append(' ')
                index += 1
            continue
        if char == '/' and following == '*':
            while index < size and not (
                source[index] == '*' and source[index + 1:index + 2] == '/'
            ):
                out.append('\n' if source[index] == '\n' else ' ')
                index += 1
            out.append('  ')
            index += 2
            continue
        if char in _QUOTES or (char == '/' and previous in _REGEX_PRECEDERS):
            end = _skip_literal(source, index)
            out.append(source[index:end])
            previous = char
            index = end
            continue
        out.append(char)
        if not char.isspace():
            previous = char
        index += 1
    return ''.join(out)


def _braced(code, brace):
    depth = 0
    index = brace
    size = len(code)
    while index < size:
        char = code[index]
        if char in _QUOTES:
            index = _skip_literal(code, index)
            continue
        if char == '{':
            depth += 1
        elif char == '}':
            depth -= 1
            if depth == 0:
                return code[brace + 1:index], index
        index += 1
    raise AssertionError('unbalanced braces from offset %d' % brace)


def _closing_paren(code, opener):
    depth = 0
    index = opener
    size = len(code)
    while index < size:
        char = code[index]
        if char in _QUOTES:
            index = _skip_literal(code, index)
            continue
        if char == '(':
            depth += 1
        elif char == ')':
            depth -= 1
            if depth == 0:
                return index
        index += 1
    raise AssertionError('unbalanced parentheses from offset %d' % opener)


def _body(code, marker, start=0):
    at = code.find(marker, start)
    assert at != -1, 'no block introduced by %r' % marker
    brace = code.find('{', at)
    assert brace != -1, 'the block introduced by %r has no body' % marker
    return _braced(code, brace)


def _guarded(code, needle, start=0):
    for match in re.finditer(r'\bif\s*\(', code[start:]):
        opener = start + match.end() - 1
        close = _closing_paren(code, opener)
        if needle not in code[opener + 1:close]:
            continue
        brace = code.find('{', close)
        assert brace != -1 and not code[close + 1:brace].strip(), (
            'the branch guarding on %r is not a braced block' % needle
        )
        return _braced(code, brace)
    raise AssertionError('no if-statement guards on %r' % needle)


def _call_arguments(code, name, start=0):
    at = code.find(name + '(', start)
    assert at != -1, 'no call to %s(' % name
    opener = at + len(name)
    return code[opener + 1:_closing_paren(code, opener)]


def _setup_code():
    return _blank_comments(_read(os.path.join('static', 'setup.js')))


def _restart_wait_code():
    return _blank_comments(_read(os.path.join('static', 'restart_wait.js')))


def test_the_wizard_reads_the_partial_save_fields_off_the_response():
    handler, _end = _body(_setup_code(), 'saved.then(function(data) {')
    branch, _branch_end = _guarded(handler, 'worker_restart_acknowledged === false')

    assert 'showSaveRestartWarning(' in branch
    assert 'data.warning' in branch


def test_the_partial_save_warning_is_not_written_into_the_countdown_element():
    code = _setup_code()
    renderer, _renderer_end = _body(code, 'function showSaveRestartWarning(')
    handler, _handler_end = _body(code, 'saved.then(function(data) {')
    branch, _branch_end = _guarded(handler, 'worker_restart_acknowledged === false')

    assert 'saveRestartWarning.textContent' in renderer
    assert 'saveFeedback.textContent' not in renderer
    assert 'saveFeedback' not in branch


def test_the_save_response_handler_still_reports_failures():
    code = _setup_code()
    _handler, end = _body(code, 'saved.then(function(data) {')
    failure, _failure_end = _body(code, '.catch(function(', end)

    assert 'err.message' in failure
    assert "saveFeedback.className = 'status-failure inline-feedback'" in failure
    assert 'saveFeedback.textContent' in failure


def test_until_is_observed_when_wait_and_go_is_called_not_when_the_floor_expires():
    outer, _outer_end = _body(_restart_wait_code(), 'function waitAndGo(options) {')
    watcher, _watcher_end = _guarded(outer, 'opts.until')
    tick, _tick_end = _body(outer, 'function tick() {')

    assert 'opts.until.then(untilSucceeded, untilRejected);' in watcher
    assert 'opts.until.then' not in tick
    assert outer.index('opts.until.then') < outer.index('function tick() {')


def test_a_rejected_until_stops_the_countdown_instead_of_overwriting_it():
    tick, _tick_end = _body(_restart_wait_code(), 'function tick() {')
    guard, _guard_end = _guarded(tick, 'untilFailed')

    assert tick.strip().startswith('if (untilFailed)')
    assert guard.strip() == 'return;'
    assert tick.index('untilFailed') < tick.index('render(')


def test_a_resolved_until_still_waits_for_the_whole_countdown():
    code = _restart_wait_code()
    resolved, _resolved_end = _body(code, 'function untilSucceeded() {')
    floor_guard, _floor_end = _guarded(resolved, 'floorReached')
    tick, _tick_end = _body(code, 'function tick() {')
    late_guard, _late_end = _guarded(tick, 'untilResolved')

    assert 'finish();' in floor_guard
    assert resolved.count('finish()') == 1
    assert 'finish();' in late_guard
    assert tick.index('floorReached = true') < tick.index('untilResolved')


def test_a_false_restart_scheduled_skips_only_the_probe():
    outer, _outer_end = _body(_restart_wait_code(), 'function waitAndGo(options) {')
    finish, _finish_end = _body(outer, 'function finish() {')
    probe_branch, _probe_end = _guarded(finish, 'willRestart')
    tick, _tick_end = _body(outer, 'function tick() {')
    countdown, _countdown_end = _guarded(tick, 'remaining > 0')
    wording = _call_arguments(countdown, 'render')

    assert 'var willRestart = opts.restartScheduled !== false;' in outer
    assert 'reconnecting();' in probe_branch
    assert finish.count('reconnecting()') == 1
    assert 'redirect(target);' in finish
    assert 'remaining -= 1;' in countdown
    assert 'window.setTimeout(tick, 1000);' in countdown
    assert re.search(r'\bif\s*\([^)]*willRestart', tick) is None
    assert tick.count('willRestart') == 1
    assert wording.count('willRestart') == 1


def test_the_redirect_keeps_its_cache_buster():
    redirect, _redirect_end = _body(_restart_wait_code(), 'function redirect(target) {')
    branch, branch_end = _guarded(redirect, 'window.appRedirect')
    fallback = redirect[branch_end:]

    assert "'_reloaded=' + Date.now()" in redirect
    assert 'window.appRedirect(fresh);' in branch
    assert 'window.location.href = fresh;' in fallback
    assert 'target' not in fallback
