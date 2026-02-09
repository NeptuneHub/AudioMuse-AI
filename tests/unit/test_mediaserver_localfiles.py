"""Unit tests for tasks/mediaserver_localfiles.py

Tests cover the LocalFiles media provider:
- Path normalization (POSIX conversion, relative paths)
- File path hashing (SHA-256 stability)
- Supported format filtering
- Metadata extraction (tags, fallbacks)
- Rating extraction (POPM, TXXX, Vorbis, M4A)
- M3U playlist management (create, list, delete)
- Directory scanning (recursive, flat)
- Connection testing
"""
import os
import sys
import hashlib
import pytest
from unittest.mock import Mock, MagicMock, patch, mock_open
from pathlib import Path, PurePosixPath


# ---------------------------------------------------------------------------
# Import helpers (bypass tasks/__init__.py -> pydub -> audioop chain)
# ---------------------------------------------------------------------------

def _import_localfiles():
    """Load tasks.mediaserver_localfiles directly without triggering tasks/__init__.py."""
    import importlib.util
    import sys
    mod_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', '..', 'tasks', 'mediaserver_localfiles.py'
    )
    mod_path = os.path.normpath(mod_path)
    mod_name = 'tasks.mediaserver_localfiles'
    if mod_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(mod_name, mod_path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[mod_name]


# ---------------------------------------------------------------------------
# Path Normalization
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestPathNormalization:
    """Test normalize_file_path()."""

    def test_posix_conversion(self):
        mod = _import_localfiles()
        result = mod.normalize_file_path('Artist\\Album\\song.mp3')
        assert '\\' not in result
        assert 'Artist/Album/song.mp3' == result

    @pytest.mark.skipif(sys.platform == 'win32', reason='POSIX absolute paths not valid on Windows')
    def test_relative_to_base(self):
        mod = _import_localfiles()
        result = mod.normalize_file_path('/music/Artist/Album/song.mp3', '/music')
        assert result == 'Artist/Album/song.mp3'

    @pytest.mark.skipif(sys.platform == 'win32', reason='POSIX absolute paths not valid on Windows')
    def test_no_base_keeps_absolute(self):
        mod = _import_localfiles()
        result = mod.normalize_file_path('/music/Artist/song.mp3', '')
        # Without base_path, absolute path stays (converted to POSIX)
        assert result.startswith('/')

    def test_whitespace_stripped(self):
        mod = _import_localfiles()
        result = mod.normalize_file_path('  Artist/song.mp3  ')
        assert result == 'Artist/song.mp3'

    def test_different_base_keeps_original(self):
        mod = _import_localfiles()
        # If path is not relative to base, keep as-is
        result = mod.normalize_file_path('/other/Artist/song.mp3', '/music')
        assert 'Artist/song.mp3' in result


# ---------------------------------------------------------------------------
# File Path Hash
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestFilePathHash:
    """Test file_path_hash() SHA-256 generation."""

    def test_deterministic(self):
        mod = _import_localfiles()
        h1 = mod.file_path_hash('Artist/Album/song.mp3')
        h2 = mod.file_path_hash('Artist/Album/song.mp3')
        assert h1 == h2

    def test_different_paths_different_hashes(self):
        mod = _import_localfiles()
        h1 = mod.file_path_hash('Artist/Album/song1.mp3')
        h2 = mod.file_path_hash('Artist/Album/song2.mp3')
        assert h1 != h2

    def test_is_sha256_hex(self):
        mod = _import_localfiles()
        h = mod.file_path_hash('test/path.mp3')
        assert len(h) == 64  # SHA-256 hex = 64 chars
        assert all(c in '0123456789abcdef' for c in h)

    def test_matches_manual_sha256(self):
        mod = _import_localfiles()
        path = 'Artist/Album/song.mp3'
        expected = hashlib.sha256(path.encode('utf-8')).hexdigest()
        assert mod.file_path_hash(path) == expected

    def test_utf8_paths(self):
        mod = _import_localfiles()
        h = mod.file_path_hash('Artiste/Café/chanson.mp3')
        assert len(h) == 64


# ---------------------------------------------------------------------------
# Supported Formats
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestFormatFiltering:
    """Test SUPPORTED_FORMATS constant and format-related logic."""

    def test_supported_formats_exist(self):
        mod = _import_localfiles()
        assert '.mp3' in mod.SUPPORTED_FORMATS
        assert '.flac' in mod.SUPPORTED_FORMATS
        assert '.ogg' in mod.SUPPORTED_FORMATS
        assert '.m4a' in mod.SUPPORTED_FORMATS

    def test_wav_supported(self):
        mod = _import_localfiles()
        assert '.wav' in mod.SUPPORTED_FORMATS

    def test_opus_supported(self):
        mod = _import_localfiles()
        assert '.opus' in mod.SUPPORTED_FORMATS

    def test_unsupported_format_excluded(self):
        mod = _import_localfiles()
        assert '.pdf' not in mod.SUPPORTED_FORMATS
        assert '.txt' not in mod.SUPPORTED_FORMATS
        assert '.jpg' not in mod.SUPPORTED_FORMATS

    def test_get_config_default_formats(self):
        """get_config returns SUPPORTED_FORMATS as default."""
        mod = _import_localfiles()
        with patch.dict(os.environ, {}, clear=False):
            cfg = mod.get_config()
            # Formats should be a list of supported extensions
            assert isinstance(cfg['supported_formats'], list)
            assert len(cfg['supported_formats']) > 0

    def test_get_config_override(self):
        """get_config accepts overrides."""
        mod = _import_localfiles()
        cfg = mod.get_config(overrides={'music_directory': '/custom/path'})
        assert cfg['music_directory'] == '/custom/path'


# ---------------------------------------------------------------------------
# Metadata Extraction
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestMetadataExtraction:
    """Test extract_metadata() with mocked mutagen."""

    def test_fallback_title_from_filename(self):
        """When mutagen returns None, title defaults to filename."""
        mod = _import_localfiles()
        with patch.object(mod, 'MUTAGEN_AVAILABLE', False):
            meta = mod.extract_metadata('/music/Artist/My Song.mp3')
            assert meta['title'] == 'My Song'
            assert meta['artist'] == 'Unknown Artist'
            assert meta['album'] == 'Unknown Album'

    def _inject_mutagen_mock(self, mod, mock_file):
        """Inject MutagenFile into module if mutagen isn't installed."""
        if not hasattr(mod, 'MutagenFile'):
            mod.MutagenFile = Mock()
        return patch.object(mod, 'MutagenFile', mock_file)

    def test_mutagen_extracts_tags(self):
        """When mutagen is available, tags are extracted."""
        mod = _import_localfiles()
        mock_audio = MagicMock()
        mock_audio.tags = {
            'title': ['Test Song'],
            'artist': ['Test Artist'],
            'album': ['Test Album'],
            'albumartist': ['Album Artist'],
            'date': ['2023'],
            'tracknumber': ['5/12'],
            'genre': ['Rock'],
        }
        mock_audio.info = MagicMock()
        mock_audio.info.length = 180.5
        mock_mutagen = Mock(return_value=mock_audio)
        with patch.object(mod, 'MUTAGEN_AVAILABLE', True), \
             self._inject_mutagen_mock(mod, mock_mutagen), \
             patch.object(mod, '_extract_rating', return_value=None):
            meta = mod.extract_metadata('/music/test.mp3')
            assert meta['title'] == 'Test Song'
            assert meta['artist'] == 'Test Artist'
            assert meta['album'] == 'Test Album'
            assert meta['album_artist'] == 'Album Artist'
            assert meta['year'] == 2023
            assert meta['track_number'] == 5
            assert meta['genre'] == 'Rock'
            assert meta['duration'] == 180.5

    def test_track_number_slash_format(self):
        """Track number '3/12' extracts as 3."""
        mod = _import_localfiles()
        mock_audio = MagicMock()
        mock_audio.tags = {'tracknumber': ['3/12']}
        mock_audio.info = None
        mock_mutagen = Mock(return_value=mock_audio)
        with patch.object(mod, 'MUTAGEN_AVAILABLE', True), \
             self._inject_mutagen_mock(mod, mock_mutagen), \
             patch.object(mod, '_extract_rating', return_value=None):
            meta = mod.extract_metadata('/music/test.mp3')
            assert meta['track_number'] == 3

    def test_performer_fallback_for_artist(self):
        """If 'artist' tag missing but 'performer' present, use performer."""
        mod = _import_localfiles()
        mock_audio = MagicMock()
        mock_audio.tags = {'performer': ['Performer Name']}
        mock_audio.info = None
        mock_mutagen = Mock(return_value=mock_audio)
        with patch.object(mod, 'MUTAGEN_AVAILABLE', True), \
             self._inject_mutagen_mock(mod, mock_mutagen), \
             patch.object(mod, '_extract_rating', return_value=None):
            meta = mod.extract_metadata('/music/test.mp3')
            assert meta['artist'] == 'Performer Name'

    def test_mutagen_returns_none(self):
        """When MutagenFile returns None, defaults are used."""
        mod = _import_localfiles()
        mock_mutagen = Mock(return_value=None)
        with patch.object(mod, 'MUTAGEN_AVAILABLE', True), \
             self._inject_mutagen_mock(mod, mock_mutagen):
            meta = mod.extract_metadata('/music/test.mp3')
            assert meta['title'] == 'test'
            assert meta['artist'] == 'Unknown Artist'

    def test_exception_returns_defaults(self):
        """Exceptions during extraction return default metadata."""
        mod = _import_localfiles()
        mock_mutagen = Mock(side_effect=Exception('corrupt file'))
        with patch.object(mod, 'MUTAGEN_AVAILABLE', True), \
             self._inject_mutagen_mock(mod, mock_mutagen):
            meta = mod.extract_metadata('/music/test.mp3')
            assert meta['title'] == 'test'
            assert meta['artist'] == 'Unknown Artist'


# ---------------------------------------------------------------------------
# Rating Extraction
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestRatingExtraction:
    """Test _extract_rating() for various tag formats."""

    def _inject_mutagen(self, mod, mock_file):
        """Ensure MutagenFile exists on module so patch.object works."""
        if not hasattr(mod, 'MutagenFile'):
            mod.MutagenFile = Mock()
        return patch.object(mod, 'MutagenFile', mock_file)

    def _make_popm_audio(self, popm_rating):
        """Build a mock audio object with POPM tag."""
        mock_popm = MagicMock()
        mock_popm.rating = popm_rating
        mock_audio = MagicMock()
        mock_tags = MagicMock()
        mock_tags.keys.return_value = ['POPM:no@email']
        mock_tags.__getitem__ = Mock(return_value=mock_popm)
        mock_audio.tags = mock_tags
        return mock_audio

    def _make_flac_audio(self, tag_dict):
        """Build a mock audio object with Vorbis-style tags (dict-like)."""
        mock_audio = MagicMock()
        # Use a MagicMock that supports dict operations
        mock_tags = MagicMock()
        mock_tags.__contains__ = lambda self, key: key in tag_dict
        mock_tags.__getitem__ = lambda self, key: tag_dict[key]
        mock_tags.__bool__ = lambda self: bool(tag_dict)
        mock_audio.tags = mock_tags
        return mock_audio

    def test_no_mutagen_returns_none(self):
        mod = _import_localfiles()
        with patch.object(mod, 'MUTAGEN_AVAILABLE', False):
            assert mod._extract_rating('/test.mp3') is None

    def test_popm_rating_zero(self):
        """POPM rating 0 maps to 0."""
        mod = _import_localfiles()
        mock_audio = self._make_popm_audio(0)
        mock_mutagen = Mock(return_value=mock_audio)
        with patch.object(mod, 'MUTAGEN_AVAILABLE', True), \
             self._inject_mutagen(mod, mock_mutagen):
            result = mod._extract_rating('/test.mp3')
            assert result == 0

    def test_popm_rating_255_maps_to_5(self):
        """POPM rating 255 maps to 5."""
        mod = _import_localfiles()
        mock_audio = self._make_popm_audio(255)
        mock_mutagen = Mock(return_value=mock_audio)
        with patch.object(mod, 'MUTAGEN_AVAILABLE', True), \
             self._inject_mutagen(mod, mock_mutagen):
            result = mod._extract_rating('/test.mp3')
            assert result == 5

    def test_popm_rating_128_maps_to_3(self):
        """POPM rating 128 maps to 3."""
        mod = _import_localfiles()
        mock_audio = self._make_popm_audio(128)
        mock_mutagen = Mock(return_value=mock_audio)
        with patch.object(mod, 'MUTAGEN_AVAILABLE', True), \
             self._inject_mutagen(mod, mock_mutagen):
            result = mod._extract_rating('/test.mp3')
            assert result == 3

    def test_flac_fmps_rating_0_5(self):
        """FLAC FMPS_RATING 0.5 maps to round(0.5*5)=3."""
        mod = _import_localfiles()
        mock_audio = self._make_flac_audio({'FMPS_RATING': ['0.5']})
        mock_mutagen = Mock(return_value=mock_audio)
        with patch.object(mod, 'MUTAGEN_AVAILABLE', True), \
             self._inject_mutagen(mod, mock_mutagen):
            result = mod._extract_rating('/test.flac')
            # 0.5 * 5 = 2.5, round() = 2 (banker's rounding)
            assert result == 2

    def test_flac_rating_direct_scale(self):
        """FLAC RATING tag with direct 0-5 value."""
        mod = _import_localfiles()
        mock_audio = self._make_flac_audio({'RATING': ['4']})
        mock_mutagen = Mock(return_value=mock_audio)
        with patch.object(mod, 'MUTAGEN_AVAILABLE', True), \
             self._inject_mutagen(mod, mock_mutagen):
            result = mod._extract_rating('/test.flac')
            assert result == 4

    def test_mutagen_file_none_returns_none(self):
        """MutagenFile returning None gives None rating."""
        mod = _import_localfiles()
        mock_mutagen = Mock(return_value=None)
        with patch.object(mod, 'MUTAGEN_AVAILABLE', True), \
             self._inject_mutagen(mod, mock_mutagen):
            assert mod._extract_rating('/test.mp3') is None

    def test_exception_returns_none(self):
        """Exceptions during rating extraction return None."""
        mod = _import_localfiles()
        mock_mutagen = Mock(side_effect=Exception('error'))
        with patch.object(mod, 'MUTAGEN_AVAILABLE', True), \
             self._inject_mutagen(mod, mock_mutagen):
            assert mod._extract_rating('/test.mp3') is None


# ---------------------------------------------------------------------------
# M3U Playlist Management
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestM3UPlaylistManagement:
    """Test M3U playlist create/list/delete operations."""

    def test_get_all_playlists_no_dir(self):
        """Missing playlist directory returns empty list."""
        mod = _import_localfiles()
        with patch.object(mod, 'get_config', return_value={'playlist_directory': '/nonexistent'}), \
             patch('os.path.isdir', return_value=False):
            result = mod.get_all_playlists()
            assert result == []

    def test_get_all_playlists_finds_m3u(self):
        """Lists .m3u and .m3u8 files."""
        mod = _import_localfiles()
        with patch.object(mod, 'get_config', return_value={'playlist_directory': '/playlists'}), \
             patch('os.path.isdir', return_value=True), \
             patch('os.listdir', return_value=['rock.m3u', 'jazz.m3u8', 'notes.txt']):
            result = mod.get_all_playlists()
            names = [p['Name'] for p in result]
            assert 'rock' in names
            assert 'jazz' in names
            assert len(result) == 2

    def test_get_playlist_by_name(self):
        """Find a playlist by exact name."""
        mod = _import_localfiles()
        with patch.object(mod, 'get_all_playlists', return_value=[
            {'Id': 'rock.m3u', 'Name': 'rock', 'Path': '/p/rock.m3u'},
            {'Id': 'jazz.m3u', 'Name': 'jazz', 'Path': '/p/jazz.m3u'},
        ]):
            result = mod.get_playlist_by_name('jazz')
            assert result is not None
            assert result['Name'] == 'jazz'

    def test_get_playlist_by_name_not_found(self):
        """Non-existent playlist returns None."""
        mod = _import_localfiles()
        with patch.object(mod, 'get_all_playlists', return_value=[]):
            assert mod.get_playlist_by_name('nonexistent') is None

    def test_delete_playlist_success(self):
        """Deleting an existing playlist returns True."""
        mod = _import_localfiles()
        with patch.object(mod, 'get_config', return_value={'playlist_directory': '/playlists'}), \
             patch('os.path.exists', return_value=True), \
             patch('os.remove') as mock_rm:
            result = mod.delete_playlist('rock.m3u')
            assert result is True
            mock_rm.assert_called_once()

    def test_delete_playlist_not_found(self):
        """Deleting a nonexistent playlist returns False."""
        mod = _import_localfiles()
        with patch.object(mod, 'get_config', return_value={'playlist_directory': '/playlists'}), \
             patch('os.path.exists', return_value=False):
            result = mod.delete_playlist('missing.m3u')
            assert result is False


# ---------------------------------------------------------------------------
# Connection Testing
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestConnectionTesting:
    """Test test_connection()."""

    def test_missing_directory(self):
        mod = _import_localfiles()
        with patch('os.path.exists', return_value=False):
            ok, msg = mod.test_connection({'music_directory': '/nonexistent'})
            assert not ok
            assert 'does not exist' in msg

    def test_not_a_directory(self):
        mod = _import_localfiles()
        with patch('os.path.exists', return_value=True), \
             patch('os.path.isdir', return_value=False):
            ok, msg = mod.test_connection({'music_directory': '/music/file.txt'})
            assert not ok
            assert 'not a directory' in msg

    def test_not_readable(self):
        mod = _import_localfiles()
        with patch('os.path.exists', return_value=True), \
             patch('os.path.isdir', return_value=True), \
             patch('os.access', return_value=False):
            ok, msg = mod.test_connection({'music_directory': '/music'})
            assert not ok
            assert 'not readable' in msg

    def test_no_audio_files(self):
        mod = _import_localfiles()
        with patch('os.path.exists', return_value=True), \
             patch('os.path.isdir', return_value=True), \
             patch('os.access', return_value=True), \
             patch('os.walk', return_value=[('/music', [], ['readme.txt'])]):
            ok, msg = mod.test_connection({'music_directory': '/music'})
            assert not ok
            assert 'No audio files' in msg

    def test_success_with_audio_files(self):
        mod = _import_localfiles()
        with patch('os.path.exists', return_value=True), \
             patch('os.path.isdir', return_value=True), \
             patch('os.access', return_value=True), \
             patch('os.walk', return_value=[('/music', [], ['song.mp3', 'track.flac'])]):
            ok, msg = mod.test_connection({'music_directory': '/music'})
            assert ok
            assert 'Found audio files' in msg


# ---------------------------------------------------------------------------
# Directory Scanning
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestDirectoryScanning:
    """Test get_all_songs() directory scanning."""

    def test_nonexistent_dir_returns_empty(self):
        mod = _import_localfiles()
        with patch.object(mod, 'get_config', return_value={
            'music_directory': '/nonexistent',
            'supported_formats': ['.mp3'],
            'scan_subdirectories': True
        }), patch('os.path.isdir', return_value=False):
            result = mod.get_all_songs()
            assert result == []

    def test_recursive_scan(self):
        """Recursive scan finds files in subdirectories."""
        mod = _import_localfiles()
        walk_data = [
            ('/music', ['Artist'], ['root.mp3']),
            ('/music/Artist', [], ['song.flac']),
        ]
        with patch.object(mod, 'get_config', return_value={
            'music_directory': '/music',
            'supported_formats': ['.mp3', '.flac'],
            'scan_subdirectories': True
        }), patch('os.path.isdir', return_value=True), \
             patch('os.walk', return_value=walk_data), \
             patch.object(mod, '_format_song', side_effect=lambda fp, bp: {
                 'Id': 'hash', 'Name': os.path.basename(fp), 'Path': fp
             }):
            result = mod.get_all_songs()
            assert len(result) == 2

    def test_flat_scan(self):
        """Non-recursive scan only finds files in the root."""
        mod = _import_localfiles()
        with patch.object(mod, 'get_config', return_value={
            'music_directory': '/music',
            'supported_formats': ['.mp3'],
            'scan_subdirectories': False
        }), patch('os.path.isdir', return_value=True), \
             patch('os.listdir', return_value=['song.mp3', 'notes.txt', 'track.mp3']), \
             patch('os.path.isfile', return_value=True), \
             patch.object(mod, '_format_song', side_effect=lambda fp, bp: {
                 'Id': 'hash', 'Name': os.path.basename(fp), 'Path': fp
             }):
            result = mod.get_all_songs()
            assert len(result) == 2  # Only .mp3 files

    def test_unsupported_format_skipped(self):
        """Files with unsupported extensions are skipped."""
        mod = _import_localfiles()
        walk_data = [('/music', [], ['song.mp3', 'image.jpg', 'doc.pdf'])]
        with patch.object(mod, 'get_config', return_value={
            'music_directory': '/music',
            'supported_formats': ['.mp3'],
            'scan_subdirectories': True
        }), patch('os.path.isdir', return_value=True), \
             patch('os.walk', return_value=walk_data), \
             patch.object(mod, '_format_song', side_effect=lambda fp, bp: {
                 'Id': 'hash', 'Name': os.path.basename(fp), 'Path': fp
             }):
            result = mod.get_all_songs()
            assert len(result) == 1


# ---------------------------------------------------------------------------
# Download Track
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestDownloadTrack:
    """Test download_track() (copy to temp dir)."""

    def test_missing_source_returns_none(self):
        mod = _import_localfiles()
        result = mod.download_track('/tmp', {'Path': '/nonexistent/file.mp3'})
        assert result is None

    def test_no_path_returns_none(self):
        mod = _import_localfiles()
        result = mod.download_track('/tmp', {'Id': '123'})
        assert result is None

    def test_successful_copy(self):
        mod = _import_localfiles()
        with patch('os.path.exists', return_value=True), \
             patch('shutil.copy2') as mock_copy:
            # First exists check is for source, second is for dest collision
            with patch('os.path.exists', side_effect=[True, False]):
                result = mod.download_track('/tmp', {
                    'Path': '/music/Artist/song.mp3',
                    'Name': 'song'
                })
            assert result is not None


# ---------------------------------------------------------------------------
# Provider Info
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestProviderInfo:
    """Test provider info metadata."""

    def test_provider_type(self):
        mod = _import_localfiles()
        info = mod.get_provider_info()
        assert info['type'] == 'localfiles'

    def test_no_play_history(self):
        mod = _import_localfiles()
        info = mod.get_provider_info()
        assert info['supports_play_history'] is False

    def test_config_fields_include_music_directory(self):
        mod = _import_localfiles()
        info = mod.get_provider_info()
        field_names = [f['name'] for f in info['config_fields']]
        assert 'music_directory' in field_names

    def test_top_played_returns_empty(self):
        mod = _import_localfiles()
        assert mod.get_top_played_songs(10) == []

    def test_last_played_returns_none(self):
        mod = _import_localfiles()
        assert mod.get_last_played_time('id123') is None
