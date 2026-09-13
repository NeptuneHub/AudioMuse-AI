# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only

"""Shared model coverage for the setup wizard and authenticated clients.

Main Features:
* Reuses directory-header counts and the resident neural fingerprint pack.
* Keeps wizard eligibility denominators while API percentages use the catalogue.
* Adds explicit-server coverage without loading encoders or search indexes.
"""

import config


def _count_rows(cur, sql, params=None):
    cur.execute(sql, params)
    row = cur.fetchone()
    return int(row[0]) if row and row[0] is not None else 0


def model_coverage_pairs(db):
    """Index counts and wizard-eligible totals; unknown index counts stay None."""
    from tasks.paged_ivf import paged_ivf_item_count
    from tasks.neural_fingerprint_index import indexed_track_count

    with db.cursor() as cur:
        total = _count_rows(cur, 'SELECT COUNT(*) FROM score')
        lyrics_total = _count_rows(cur, 'SELECT COUNT(*) FROM lyrics_embedding WHERE embedding IS NOT NULL')
    return {
        'musicnn': (paged_ivf_item_count(db, config.INDEX_NAME), total),
        'clap': (paged_ivf_item_count(db, 'clap_index'), total),
        'lyrics': (paged_ivf_item_count(db, 'lyrics_index'), lyrics_total),
        'neural-fingerprint': (indexed_track_count(), total),
    }


def _coverage(count, total):
    return {
        'count': count,
        'total': total,
        'percentage': None if count is None else (
            round(min(100.0, max(0.0, count * 100.0 / total)), 2) if total else 0.0
        ),
    }


def get_model_coverage(server_id=None):
    """All four models; local coverage is present only for an explicit source."""
    from database import get_db
    from tasks.mediaserver import registry
    from tasks.neural_fingerprint_index import get_scoped_status
    from tasks.paged_ivf import paged_ivf_scoped_item_count

    db = get_db()
    try:
        pairs = model_coverage_pairs(db)
        total = pairs['musicnn'][1]
        enabled = {
            'musicnn': True,  # The setup wizard's always-on model has no flag.
            'clap': bool(config.CLAP_ENABLED),
            'lyrics': bool(config.LYRICS_ENABLED),
            'neural-fingerprint': bool(config.NEURAL_FINGERPRINT_ENABLED),
        }
        models = {
            model: {'enabled': enabled[model], 'global': _coverage(pair[0], total)}
            for model, pair in pairs.items()
        }
        result = {'models': models}
        if server_id is not None:
            include_legacy = server_id == registry.get_default_server_id()
            with db.cursor() as cur:
                local_total = _count_rows(
                    cur, 'SELECT COUNT(*) FROM score s WHERE ' + registry.availability_sql('s'),
                    (server_id, include_legacy),
                )
            for model, name in (('musicnn', config.INDEX_NAME), ('clap', 'clap_index'), ('lyrics', 'lyrics_index')):
                count = paged_ivf_scoped_item_count(db, name, server_id)
                models[model]['local'] = _coverage(count, local_total)
            count = get_scoped_status(server_id)['indexed_tracks']
            models['neural-fingerprint']['local'] = _coverage(count, local_total)
            result['server_id'] = server_id
        return result
    except Exception:
        db.rollback()
        raise
