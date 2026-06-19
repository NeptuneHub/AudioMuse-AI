from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_playlist_curator_uses_renamed_mediaserver_package_imports():
    source = (REPO_ROOT / "app_playlist_curator.py").read_text(encoding="utf-8")

    legacy_imports = [
        "tasks.mediaserver_jellyfin",
        "tasks.mediaserver_emby",
        "tasks.mediaserver_navidrome",
        "tasks.mediaserver_lyrion",
    ]

    for legacy_import in legacy_imports:
        assert legacy_import not in source

    assert "tasks.mediaserver.jellyfin" in source
    assert "tasks.mediaserver.emby" in source
    assert "tasks.mediaserver.navidrome" in source
    assert "tasks.mediaserver.lyrion" in source


def test_ivf_manager_no_longer_references_removed_voyager_index_global():
    source = (REPO_ROOT / "tasks" / "ivf_manager.py").read_text(encoding="utf-8")

    assert "voyager_index" not in source
