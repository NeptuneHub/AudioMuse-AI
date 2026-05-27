"""Tests for app_cron.py — sonic_fingerprint cron branch (issue #336).

Verifies:
- Empty fingerprint results → previous playlist is preserved (no upsert call)
- Non-empty results → create_or_replace_playlist called with the constant name
- Backend that raises NotImplementedError (e.g. MPD) → falls back to legacy
  date-suffixed create_playlist_from_ids path
"""
from unittest.mock import MagicMock, patch


def _make_cron_row(task_type='sonic_fingerprint'):
    """Build a row dict matching the DictCursor shape used by run_due_cron_jobs."""
    return {
        'id': 1,
        'name': 'Sonic Fingerprint',
        'task_type': task_type,
        'cron_expr': '* * * * *',
        'enabled': True,
        'last_run': 0,
    }


def _setup_db_mock():
    """Mock a get_db() that returns one enabled sonic_fingerprint cron row."""
    cur = MagicMock()
    cur.fetchall.return_value = [_make_cron_row()]
    db = MagicMock()
    db.cursor.return_value = cur
    return db, cur


@patch('app_cron.cron_matches_now', return_value=True)
@patch('app_cron.get_db')
def test_sonic_fingerprint_branch_skips_on_empty_results(mock_get_db, _matches):
    """If generate_sonic_fingerprint returns [], do NOT call create_or_replace_playlist."""
    from app_cron import run_due_cron_jobs

    db, _cur = _setup_db_mock()
    mock_get_db.return_value = db

    with patch('tasks.sonic_fingerprint_manager.generate_sonic_fingerprint', return_value=[]) as gen, \
         patch('tasks.mediaserver.create_or_replace_playlist') as upsert, \
         patch('tasks.voyager_manager.create_playlist_from_ids') as legacy:
        run_due_cron_jobs()

    gen.assert_called_once()
    upsert.assert_not_called()
    legacy.assert_not_called()


@patch('app_cron.cron_matches_now', return_value=True)
@patch('app_cron.get_db')
def test_sonic_fingerprint_branch_calls_upsert_with_constant_name(mock_get_db, _matches):
    """Non-empty results → upsert called once with SONIC_FINGERPRINT_CRON_PLAYLIST_NAME."""
    from app_cron import run_due_cron_jobs
    from config import SONIC_FINGERPRINT_CRON_PLAYLIST_NAME

    db, _cur = _setup_db_mock()
    mock_get_db.return_value = db

    fp = [{'item_id': 'a'}, {'item_id': 'b'}, {'item_id': 'c'}]

    with patch('tasks.sonic_fingerprint_manager.generate_sonic_fingerprint', return_value=fp), \
         patch('tasks.mediaserver.create_or_replace_playlist', return_value={'Id': 'pl-x'}) as upsert, \
         patch('tasks.voyager_manager.create_playlist_from_ids') as legacy:
        run_due_cron_jobs()

    upsert.assert_called_once_with(SONIC_FINGERPRINT_CRON_PLAYLIST_NAME, ['a', 'b', 'c'])
    legacy.assert_not_called()


@patch('app_cron.cron_matches_now', return_value=True)
@patch('app_cron.get_db')
def test_sonic_fingerprint_branch_falls_back_for_mpd(mock_get_db, _matches):
    """Backend raising NotImplementedError → legacy create_playlist_from_ids called."""
    from app_cron import run_due_cron_jobs

    db, _cur = _setup_db_mock()
    mock_get_db.return_value = db

    fp = [{'item_id': 'a'}]

    with patch('tasks.sonic_fingerprint_manager.generate_sonic_fingerprint', return_value=fp), \
         patch('tasks.mediaserver.create_or_replace_playlist', side_effect=NotImplementedError), \
         patch('tasks.voyager_manager.create_playlist_from_ids', return_value='legacy-id') as legacy:
        run_due_cron_jobs()

    legacy.assert_called_once()
    legacy_name = legacy.call_args[0][0]
    assert legacy_name.startswith('Sonic Fingerprint (Cron ')
    assert legacy.call_args[0][1] == ['a']


@patch('app_cron.cron_matches_now', return_value=True)
@patch('app_cron.get_db')
def test_alchemy_anchors_branch_refreshes_anchor_playlists(mock_get_db, _matches):
    """Each saved alchemy anchor refreshes a stable <anchor>_automatic playlist."""
    from app_cron import run_due_cron_jobs

    cur = MagicMock()
    cur.fetchall.return_value = [_make_cron_row('alchemy_anchors')]
    db = MagicMock()
    db.cursor.return_value = cur
    mock_get_db.return_value = db

    anchors = [{'id': 7, 'name': 'Jazz Night'}]
    alchemy = {'results': [{'item_id': 'a'}, {'item_id': 'b'}]}

    with patch('app_helper.get_alchemy_anchors', return_value=anchors), \
         patch('tasks.song_alchemy.song_alchemy', return_value=alchemy) as run_alchemy, \
         patch('tasks.mediaserver.create_or_replace_playlist', return_value={'Id': 'pl-1'}) as upsert, \
         patch('tasks.voyager_manager.create_playlist_from_ids') as legacy:
        run_due_cron_jobs()

    run_alchemy.assert_called_once_with(add_items=[{'type': 'anchor', 'id': 7}])
    upsert.assert_called_once_with('Jazz Night_automatic', ['a', 'b'])
    legacy.assert_not_called()
