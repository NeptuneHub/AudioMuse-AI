# Real lyrics analysis integration test
#
# Runs the full lyrics pipeline (whisper → langdetect → marian → e5 embedding +
# axis scoring) on the audio files shipped under test/songs/ and verifies that
# the produced ``embedding`` and ``axis_vector`` are stable across runs.
#
# The pipeline is too floating-point heavy for byte-exact reproduction, so we
# compare against pre-recorded reference vectors via cosine similarity. A run
# is considered successful when both vectors score >= 0.99 against their
# expected counterparts.
#
# Quick start:
#   1. Create a virtual environment and install deps:
#        python -m venv test/.venv
#        source test/.venv/bin/activate    (PowerShell: test\.venv\Scripts\Activate.ps1)
#        pip install -r test/requirements.txt
#   2. Make sure the bundled HuggingFace cache is available. On CI the workflow
#      downloads it from the v4.0.0-model release and exports HF_HOME. For a
#      local run set HF_HOME yourself, e.g.:
#        export HF_HOME=$PWD/test/.hf_cache
#      and extract huggingface_models.tar.gz into that folder so e5-base-v2 is
#      resolvable with local_files_only=True.
#   3. Record the expected vectors once:
#        LYRICS_RECORD_EXPECTED=1 pytest test/test_lyrics_analysis_integration.py -s -v
#      This writes test/lyrics_expected.json, which must be committed.
#   4. From then on, just run:
#        pytest test/test_lyrics_analysis_integration.py -s -v
#
# The test is intentionally configured to be deterministic:
#   * LYRICS_API_ENABLE  = False  (no LRCLIB / lyrics.ovh lookups)
#   * LYRICS_LLM_ENABLED = False  (no Qwen cleanup, no llama-cpp dep needed)
#   * LYRICS_USE_GPU     = false  (CPU only, fp16=False inside whisper)
#   * use_llm_cleanup    = False  (extra safety belt at the call site)
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest


SIMILARITY_THRESHOLD = 0.99


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 0.0 and nb <= 0.0:
        # Two zero vectors: treat as identical (sentinel-on-sentinel).
        return 1.0
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


@pytest.mark.integration
def test_real_lyrics_analysis_runs_and_matches_expected_vectors():
    """Integration test: runs analyze_lyrics on the bundled test songs and
    verifies the produced embedding / axis vectors are stable.

    Skipped if openai-whisper / torch / transformers / librosa are not
    importable, or if the bundled HuggingFace e5-base-v2 cache is not
    available offline.
    """
    project_root = Path(__file__).resolve().parents[1]
    songs_dir = project_root / 'test' / 'songs'
    models_dir = project_root / 'test' / 'models'
    expected_path = project_root / 'test' / 'lyrics_expected.json'

    if not songs_dir.exists():
        pytest.skip(f'Test songs directory not found: {songs_dir}')

    # Optional heavy deps — skip cleanly if any are missing in this env.
    try:
        import torch  # noqa: F401
    except Exception as exc:  # pragma: no cover - env-dependent
        pytest.skip(f'torch not importable: {exc}')
    try:
        import whisper  # noqa: F401
    except Exception as exc:  # pragma: no cover
        pytest.skip(f'openai-whisper not importable: {exc}')
    try:
        import librosa  # noqa: F401
    except Exception as exc:  # pragma: no cover
        pytest.skip(f'librosa not importable: {exc}')
    try:
        from transformers import AutoTokenizer
    except Exception as exc:  # pragma: no cover
        pytest.skip(f'transformers not importable: {exc}')

    # ---- Force offline / CPU / no external services ------------------------
    os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('HF_DATASETS_OFFLINE', '1')
    os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
    os.environ.setdefault('HF_XET_DISABLE', '1')

    os.environ['LYRICS_API_ENABLE'] = 'false'
    os.environ['LYRICS_LLM_ENABLED'] = 'false'
    os.environ['LYRICS_USE_GPU'] = 'false'

    # Cache whisper's small.pt under test/models so first-run download is
    # reused by subsequent runs.
    models_dir.mkdir(parents=True, exist_ok=True)
    os.environ['LYRICS_MODEL_DIR'] = str(models_dir)

    # Verify the bundled HF cache contains the e5-base-v2 model offline.
    try:
        AutoTokenizer.from_pretrained('intfloat/e5-base-v2', local_files_only=True)
    except Exception as exc:
        pytest.skip(
            "intfloat/e5-base-v2 not available offline (HF_HOME=%s). "
            "Extract huggingface_models.tar.gz from release v4.0.0-model "
            "into your HF cache. Underlying error: %s"
            % (os.environ.get('HF_HOME'), exc)
        )

    # ---- Make project importable + override config singleton ---------------
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    import config
    config.LYRICS_API_ENABLE = False
    config.LYRICS_LLM_ENABLED = False
    config.LYRICS_USE_GPU = 'false'
    config.LYRICS_MODEL_DIR = str(models_dir)

    from lyrics.lyrics_transcriber import analyze_lyrics, axis_columns

    # ---- Tracks under test --------------------------------------------------
    test_tracks = [
        'Art Flower - Art Flower - Creamy Snowflakes.mp3',
        'Aaron Dunn - Minuet - Notebook for Anna Magdalena.mp3',
        "Michael Hawley - Sonata 'Waldstein', Op. 53 - II. Introduzione-Adagio molto.mp3",
    ]

    # Record mode is triggered explicitly via env var, OR implicitly the very
    # first time the test runs (no expected JSON checked in yet). In CI the
    # workflow detects the freshly-written file and commits it back to main
    # so subsequent runs pin against it.
    explicit_record = os.environ.get('LYRICS_RECORD_EXPECTED', '').lower() in ('1', 'true', 'yes')
    record_mode = explicit_record or not expected_path.exists()
    expected = {}
    if record_mode and not explicit_record:
        print(
            f'\n[lyrics-test] {expected_path.name} not found - entering RECORD mode. '
            f'The freshly-computed vectors will be written to {expected_path} and must be '
            f'committed (CI does this automatically on push to main).'
        )
    if not record_mode:
        with expected_path.open('r', encoding='utf-8') as fh:
            expected = json.load(fh)

    recorded = {}
    failures = []
    expected_axis_dim = len(axis_columns())

    for track_name in test_tracks:
        track_path = songs_dir / track_name
        if not track_path.exists():
            print(f'\n[skip] {track_name!r} not present in test/songs/')
            continue

        print(f'\n{"=" * 80}')
        print(f'=== Analyzing lyrics: {track_name}')
        print(f'{"=" * 80}')

        result = analyze_lyrics(
            source_path=str(track_path),
            use_llm_cleanup=False,
            artist=None,
            track=None,
        )

        embedding = result.get('embedding')
        axis_vector = result.get('axis_vector')

        assert embedding is not None, f'{track_name}: pipeline returned no embedding'
        assert axis_vector is not None, f'{track_name}: pipeline returned no axis_vector'

        emb = np.asarray(embedding, dtype=np.float32).reshape(-1)
        axv = np.asarray(axis_vector, dtype=np.float32).reshape(-1)

        print(f'  language          : {result.get("language")}')
        print(f'  embedding shape   : {emb.shape}')
        print(f'  axis_vector shape : {axv.shape} (expected {expected_axis_dim})')
        text_preview = (result.get('cleaned_text') or result.get('text') or '')[:80]
        print(f'  text preview      : {text_preview!r}')

        assert axv.shape[0] == expected_axis_dim, (
            f'{track_name}: axis_vector dim {axv.shape[0]} != {expected_axis_dim}'
        )

        if record_mode:
            recorded[track_name] = {
                'language': result.get('language'),
                'embedding': emb.tolist(),
                'axis_vector': axv.tolist(),
            }
            print('  RECORDED expected vectors.')
            continue

        track_expected = expected.get(track_name)
        if track_expected is None:
            failures.append(f'{track_name}: no expected entry in {expected_path.name}')
            continue

        exp_emb = np.asarray(track_expected['embedding'], dtype=np.float32)
        exp_axv = np.asarray(track_expected['axis_vector'], dtype=np.float32)

        if emb.shape != exp_emb.shape:
            failures.append(
                f'{track_name}: embedding shape mismatch '
                f'{tuple(emb.shape)} != {tuple(exp_emb.shape)}'
            )
            continue
        if axv.shape != exp_axv.shape:
            failures.append(
                f'{track_name}: axis_vector shape mismatch '
                f'{tuple(axv.shape)} != {tuple(exp_axv.shape)}'
            )
            continue

        emb_sim = _cosine(emb, exp_emb)
        axv_sim = _cosine(axv, exp_axv)
        emb_status = 'OK' if emb_sim >= SIMILARITY_THRESHOLD else 'FAIL'
        axv_status = 'OK' if axv_sim >= SIMILARITY_THRESHOLD else 'FAIL'
        print(f'  embedding cos sim  : {emb_sim:.6f} [{emb_status}]')
        print(f'  axis_vector cos sim: {axv_sim:.6f} [{axv_status}]')

        if emb_sim < SIMILARITY_THRESHOLD:
            failures.append(
                f'{track_name}: embedding cosine similarity {emb_sim:.6f} '
                f'< threshold {SIMILARITY_THRESHOLD}'
            )
        if axv_sim < SIMILARITY_THRESHOLD:
            failures.append(
                f'{track_name}: axis_vector cosine similarity {axv_sim:.6f} '
                f'< threshold {SIMILARITY_THRESHOLD}'
            )

    if record_mode:
        with expected_path.open('w', encoding='utf-8') as fh:
            json.dump(recorded, fh, indent=2)
        print(f'\nWrote expected vectors to {expected_path}')
        return

    if failures:
        msg = 'Lyrics integration test failed:\n  - ' + '\n  - '.join(failures)
        pytest.fail(msg)


if __name__ == '__main__':
    pytest.main([__file__, '-s', '-v'])
