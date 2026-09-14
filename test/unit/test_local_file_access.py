# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Direct file access: reading a mounted library instead of downloading it.

The two properties worth protecting are that the library file is never what the
pipeline deletes, and that a path reported by the media server cannot be used to
read outside the configured roots. Everything else is a fallback: any failure
must return None so the caller downloads the track as before.

Main Features:
* A returned path is always a symlink in the temp dir, and deleting it (as the
  analysis pipeline does) leaves the library file intact
* Paths escaping LOCAL_FILE_ROOTS are refused, including via a symlink planted
  inside the library and via ../ traversal
* No roots configured, feature disabled, missing file and empty file all fall
  back to downloading
* LOCAL_FILE_PATH_MAP rewrites the server's prefix onto the mount point,
  longest prefix first
"""

import os

import pytest


@pytest.fixture
def library(tmp_path):
    """A library root holding one track, plus a secret file outside it."""
    root = tmp_path / 'library'
    (root / 'Artist' / 'Album').mkdir(parents=True)
    track = root / 'Artist' / 'Album' / 'song.flac'
    track.write_bytes(b'fLaC' + b'\x00' * 64)
    secret = tmp_path / 'secret.txt'
    secret.write_text('do not read me')
    return {'root': root, 'track': track, 'secret': secret, 'tmp': tmp_path}


@pytest.fixture
def configured(monkeypatch, library):
    from tasks.mediaserver import local_file

    monkeypatch.setattr(local_file.config, 'LOCAL_FILE_ACCESS', True, raising=False)
    monkeypatch.setattr(
        local_file.config, 'LOCAL_FILE_ROOTS', str(library['root']), raising=False
    )
    monkeypatch.setattr(local_file.config, 'LOCAL_FILE_PATH_MAP', '', raising=False)
    monkeypatch.setattr(local_file.config, 'LOCAL_FILE_REQUIRE_READONLY', True, raising=False)
    # A pytest tmp dir is writable, so stand in for the read-only mount the
    # feature requires. TestTheLibraryMustBeReadOnly checks the real primitive.
    monkeypatch.setattr(local_file, '_writable', lambda path: False)
    local_file._warned.clear()
    return local_file


def _temp_dir(library):
    path = library['tmp'] / 'temp_audio'
    path.mkdir(exist_ok=True)
    return str(path)


class TestTheLibraryFileIsNeverAtRisk:
    def test_the_returned_path_is_a_link_in_the_temp_dir(self, configured, library):
        link = configured.link_local_copy(
            _temp_dir(library), {'Id': '42', 'Path': str(library['track'])}
        )

        assert link is not None
        assert os.path.dirname(link) == _temp_dir(library)
        assert os.path.samefile(link, str(library['track']))

    def test_deleting_the_returned_path_leaves_the_library_untouched(self, configured, library):
        """The analysis pipeline removes what download_track returns."""
        link = configured.link_local_copy(
            _temp_dir(library), {'Id': '42', 'Path': str(library['track'])}
        )

        os.remove(link)

        assert not os.path.lexists(link)
        assert library['track'].exists()
        assert library['track'].read_bytes().startswith(b'fLaC')

    def test_the_link_keeps_the_real_extension(self, configured, library):
        link = configured.link_local_copy(
            _temp_dir(library), {'Id': '42', 'Path': str(library['track'])}
        )

        assert os.path.basename(link) == '42.flac'

    def test_a_second_call_replaces_the_link_rather_than_failing(self, configured, library):
        item = {'Id': '42', 'Path': str(library['track'])}
        first = configured.link_local_copy(_temp_dir(library), item)
        second = configured.link_local_copy(_temp_dir(library), item)

        assert first == second
        assert os.path.samefile(second, str(library['track']))


class TestTheLibraryMustBeReadOnly:
    """Nothing here writes to the library; a read-only root means nothing CAN."""

    def test_the_writable_check_is_real(self, library):
        """Guards the simulation in the fixture from making the suite vacuous."""
        from tasks.mediaserver import local_file

        assert local_file._writable(str(library['root'])) is True

    def test_a_writable_root_is_skipped_by_default(self, configured, library, monkeypatch):
        monkeypatch.setattr(configured, '_writable', lambda path: True)

        assert configured.link_local_copy(
            _temp_dir(library), {'Id': '1', 'Path': str(library['track'])}
        ) is None

    def test_the_requirement_can_be_waived(self, configured, library, monkeypatch):
        monkeypatch.setattr(configured, '_writable', lambda path: True)
        monkeypatch.setattr(
            configured.config, 'LOCAL_FILE_REQUIRE_READONLY', False, raising=False
        )

        assert configured.link_local_copy(
            _temp_dir(library), {'Id': '1', 'Path': str(library['track'])}
        ) is not None


class TestUntrustedPathsAreRefused:
    def test_a_path_outside_the_roots_is_refused(self, configured, library):
        assert configured.link_local_copy(
            _temp_dir(library), {'Id': '1', 'Path': str(library['secret'])}
        ) is None

    def test_traversal_out_of_the_root_is_refused(self, configured, library):
        escape = str(library['root'] / 'Artist' / '..' / '..' / 'secret.txt')

        assert configured.link_local_copy(_temp_dir(library), {'Id': '1', 'Path': escape}) is None

    def test_a_symlink_inside_the_library_cannot_point_out_of_it(self, configured, library):
        """realpath is checked, not the path as given."""
        planted = library['root'] / 'Artist' / 'escape.flac'
        try:
            os.symlink(str(library['secret']), str(planted))
        except (OSError, NotImplementedError):
            pytest.skip('symlink creation is not permitted in this environment')

        assert configured.link_local_copy(
            _temp_dir(library), {'Id': '1', 'Path': str(planted)}
        ) is None

    def test_no_configured_root_refuses_everything(self, configured, library, monkeypatch):
        monkeypatch.setattr(configured.config, 'LOCAL_FILE_ROOTS', '', raising=False)

        assert configured.link_local_copy(
            _temp_dir(library), {'Id': '1', 'Path': str(library['track'])}
        ) is None


class TestFallingBackToDownloading:
    def test_disabled_returns_none_without_touching_the_path(self, configured, library, monkeypatch):
        monkeypatch.setattr(configured.config, 'LOCAL_FILE_ACCESS', False, raising=False)

        assert configured.link_local_copy(
            _temp_dir(library), {'Id': '1', 'Path': str(library['track'])}
        ) is None

    def test_a_missing_file_returns_none(self, configured, library):
        missing = str(library['root'] / 'Artist' / 'Album' / 'gone.flac')

        assert configured.link_local_copy(_temp_dir(library), {'Id': '1', 'Path': missing}) is None

    def test_an_empty_file_returns_none(self, configured, library):
        empty = library['root'] / 'Artist' / 'Album' / 'empty.flac'
        empty.write_bytes(b'')

        assert configured.link_local_copy(
            _temp_dir(library), {'Id': '1', 'Path': str(empty)}
        ) is None

    def test_an_item_with_no_path_returns_none(self, configured, library):
        assert configured.link_local_copy(_temp_dir(library), {'Id': '1'}) is None


class TestPathMapping:
    def test_the_server_prefix_is_rewritten_onto_the_mount_point(
        self, configured, library, monkeypatch
    ):
        monkeypatch.setattr(
            configured.config,
            'LOCAL_FILE_PATH_MAP',
            f"/srv/music={library['root']}",
            raising=False,
        )

        link = configured.link_local_copy(
            _temp_dir(library), {'Id': '7', 'Path': '/srv/music/Artist/Album/song.flac'}
        )

        assert link is not None
        assert os.path.samefile(link, str(library['track']))

    def test_the_longest_matching_prefix_wins(self, configured, library, monkeypatch):
        monkeypatch.setattr(
            configured.config,
            'LOCAL_FILE_PATH_MAP',
            f"/srv={library['tmp'] / 'wrong'},/srv/music={library['root']}",
            raising=False,
        )

        link = configured.link_local_copy(
            _temp_dir(library), {'Id': '7', 'Path': '/srv/music/Artist/Album/song.flac'}
        )

        assert link is not None
        assert os.path.samefile(link, str(library['track']))

    def test_a_file_url_is_understood(self, configured, library):
        url = 'file://' + str(library['track']).replace(os.sep, '/')
        if not url.startswith('file:///'):
            url = url.replace('file://', 'file:///', 1)

        link = configured.link_local_copy(_temp_dir(library), {'Id': '7', 'Path': url})

        assert link is not None
        assert os.path.samefile(link, str(library['track']))

    def test_windows_separators_survive_the_rewrite(self, configured, library, monkeypatch):
        monkeypatch.setattr(
            configured.config,
            'LOCAL_FILE_PATH_MAP',
            f"D:\\Media={library['root']}",
            raising=False,
        )

        link = configured.link_local_copy(
            _temp_dir(library), {'Id': '7', 'Path': 'D:\\Media\\Artist\\Album\\song.flac'}
        )

        assert link is not None
        assert os.path.samefile(link, str(library['track']))


class TestDispatcherIntegration:
    def test_a_local_hit_never_calls_the_provider(self, configured, library, monkeypatch):
        from unittest.mock import MagicMock

        from tasks import mediaserver

        provider = MagicMock()
        monkeypatch.setattr(mediaserver, '_provider', lambda *a, **k: provider)

        path = mediaserver.download_track(
            _temp_dir(library), {'Id': '42', 'Path': str(library['track'])}
        )

        assert os.path.samefile(path, str(library['track']))
        provider.download_track.assert_not_called()

    def test_a_local_miss_falls_through_to_the_provider(self, configured, library, monkeypatch):
        from unittest.mock import MagicMock

        from tasks import mediaserver

        downloaded = os.path.join(_temp_dir(library), 'downloaded.mp3')
        with open(downloaded, 'wb') as handle:
            handle.write(b'ID3')
        provider = MagicMock()
        provider.download_track.return_value = downloaded
        monkeypatch.setattr(mediaserver, '_provider', lambda *a, **k: provider)

        path = mediaserver.download_track(
            _temp_dir(library), {'Id': '42', 'Path': str(library['secret'])}
        )

        assert path == downloaded
        provider.download_track.assert_called_once()
