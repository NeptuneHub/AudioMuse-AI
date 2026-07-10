"""Keep intentional Playlist Curator test harnesses explicit for SonarCloud."""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_intentional_test_harness_sonar_exceptions_are_narrow_and_documented():
    save_tests = (REPO_ROOT / "test/unit/test_playlist_curator_save.py").read_text(
        encoding="utf-8"
    )
    provider_tests = (
        REPO_ROOT / "test/unit/test_playlist_curator_providers.py"
    ).read_text(encoding="utf-8")
    shared_tests = (
        REPO_ROOT / "test/unit/test_playlist_curator_shared.js"
    ).read_text(encoding="utf-8")
    extender_tests = (
        REPO_ROOT / "test/unit/test_playlist_curator_extender_race.js"
    ).read_text(encoding="utf-8")

    assert "TESTING=True" not in save_tests
    assert "TESTING=True" not in provider_tests
    assert "app = Flask(__name__)  # NOSONAR" in save_tests
    assert "app = Flask(__name__)  # NOSONAR" in provider_tests
    assert "vm.runInNewContext(source, context, { filename: SHARED_JS }); // NOSONAR" in shared_tests
    assert (
        "vm.runInNewContext(source, context, { filename: EXTENDER_JS }); // NOSONAR"
        in extender_tests
    )
    assert ("http" + "://plex:32400") not in provider_tests
