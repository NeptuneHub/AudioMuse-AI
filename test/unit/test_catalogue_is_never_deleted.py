# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""The catalogue never deletes a song that is still on a server.

`score` holds one row per distinct recording, keyed by the fp hash of its own
audio, and the row carries the analysis - MusiCNN, CLAP and lyrics embeddings that
cost real compute to produce. So losing ONE server never deletes the row: it is
only unbound from that server (its track_server_map row goes), stays in the
catalogue, and is hidden from that server's results by the availability filter; a
later sweep re-binds it with no re-analysis. Removing a server or migrating
providers change only `track_server_map`.

Two deleters are sanctioned, and both only remove a row whose audio is gone from
every library or already preserved elsewhere:
* the fingerprint canonicalizer's duplicate MERGE, which folds a row into the
  canonical row that already holds the same audio's analysis (nothing lost);
* the cleaning pass, which deletes a row bound to NO server (an orphan) - the file
  is gone from every library, so the analysis is re-created if it ever returns -
  and only when every server was read completely, so an incomplete view can never
  delete a track still on a server.

Main Features:
* Only the canonicalizer merge and the cleaning orphan-delete may DELETE FROM score
* embedding / clap_embedding / lyrics_embedding cascade from score, so they are
  covered by the same rule
"""

import os
import pathlib
import re

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

# Pruned during the walk, not filtered after it: descending into .venv costs ~90s.
_SKIP_DIRS = {
    '.venv', '.venv-windows', '.git', 'dist', 'build', 'native-build', 'test',
    'node_modules', '__pycache__',
}

# The sanctioned deleters: the canonicalizer's duplicate merge (deletes a row only
# after folding it into the canonical row for the same audio), and the cleaning pass
# (deletes only orphans bound to no server, and only on a complete server view).
_SANCTIONED = {
    'tasks/fingerprint_canonicalize.py': 'duplicate merge into the canonical row',
    'tasks/cleaning.py': 'orphan delete: tracks bound to no server, complete view only',
}

_DELETE_SCORE = re.compile(r'DELETE\s+FROM\s+score\b', re.IGNORECASE)


def _tracked_python_files():
    for dirpath, dirnames, filenames in os.walk(REPO_ROOT):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
        for name in filenames:
            if not name.endswith('.py'):
                continue
            path = pathlib.Path(dirpath) / name
            yield path.relative_to(REPO_ROOT).as_posix(), path


def test_only_sanctioned_deleters_may_delete_from_score():
    offenders = []
    for rel, path in _tracked_python_files():
        if rel in _SANCTIONED:
            continue
        text = path.read_text(encoding='utf-8', errors='ignore')
        if _DELETE_SCORE.search(text):
            offenders.append(rel)

    assert not offenders, (
        "These modules delete from the centralized catalogue:\n  "
        + "\n  ".join(offenders)
        + "\n\nA song's analysis must survive losing a server. Unbind it instead: "
        "delete its track_server_map row and leave the score row alone. "
        "The availability filter then hides it from that server's results, and a "
        "later sweep re-binds it with no re-analysis."
    )


def test_the_canonicalizer_deletes_only_rows_it_has_already_merged():
    """Its DELETE is joined to duplicate_item_id_map: every row it drops has just been
    folded into the canonical row holding the same audio, so no analysis is lost."""
    text = (REPO_ROOT / 'tasks' / 'fingerprint_canonicalize.py').read_text(encoding='utf-8')
    deletes = [
        line.strip()
        for line in text.splitlines()
        if _DELETE_SCORE.search(line)
    ]
    assert deletes, "expected the canonicalizer's merge delete to still exist"
    for stmt in deletes:
        assert 'duplicate_item_id_map' in stmt, (
            "the canonicalizer may only delete rows it has merged: " + stmt
        )


def test_provider_migration_unbinds_instead_of_deleting():
    """A provider swap changes where a song LIVES, not whether it is known."""
    text = (REPO_ROOT / 'tasks' / 'provider_migration_tasks.py').read_text(encoding='utf-8')
    assert not _DELETE_SCORE.search(text)
    assert 'DELETE FROM track_server_map' in text
    # And it must never rewrite the canonical id, which is derived from the audio.
    assert 'UPDATE score s SET item_id' not in text
    assert 'SET item_id = m.new_id' not in text
