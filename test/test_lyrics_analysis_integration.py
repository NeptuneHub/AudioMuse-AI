# Real lyrics analysis integration test
#
# Goal: verify that the deterministic part of the lyrics pipeline (e5-base-v2
# text embedding + per-axis softmax scoring) is stable across code changes.
#
# Strategy: skip Whisper entirely. We monkey-patch ``fetch_remote_lyrics`` to
# return a fixed, hand-written ~200-word English lyric, so the pipeline takes
# the API path:
#     STEP 0 (API hit) -> STEP 3 (langdetect=en) -> STEP 4 (no translation)
#     -> STEP 5 (LLM disabled) -> STEP 6 (e5 embedding + axis vector)
# Because the input text is identical on every run, the resulting
# ``embedding`` and ``axis_vector`` must be reproducible (cosine sim >= 0.99
# vs the recorded reference).
#
# The e5 model is loaded the same way the production code and Dockerfile do:
# from the *flat* directory ``${LYRICS_MODEL_DIR}/e5-base-v2`` extracted from
# the ``lyrics_model.tar.gz`` GitHub release artifact, NOT from a HuggingFace
# Hub cache.
#
# First-run behaviour: if ``test/lyrics_expected.json`` is missing, the test
# enters RECORD mode automatically, writes the file and passes. The CI
# workflow (only on push to main) then commits the file back so subsequent
# runs pin against it.
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest


SIMILARITY_THRESHOLD = 0.99


# Hand-written, deterministic, ~200-word English "lyrics" used as the fake
# API payload. Purely synthetic so we don't ship copyrighted material.
FAKE_LYRICS_BY_TRACK = {
    'love_song': (
        "I walk along the river when the morning light is gold\n"
        "I think about the stories that your tender heart has told\n"
        "Every step beside you feels like coming home\n"
        "Every laugh we share is a small forever of our own\n"
        "\n"
        "Hold me close and never let the quiet slip away\n"
        "Hold me close, my dear, and I will hold you every day\n"
        "We are dancing in the kitchen with the radio so low\n"
        "Singing songs about the people we used to know\n"
        "\n"
        "When the winter comes and steals the colour from the trees\n"
        "We will build a little fire and share a cup of tea\n"
        "When the summer rolls around and paints the sky in blue\n"
        "Every sunny afternoon I will spend with you\n"
        "\n"
        "So tell me all your worries, tell me all your dreams\n"
        "Tell me where the road of every wandering heart leads\n"
        "I am here to listen and I am here to stay\n"
        "Hold me close, my dear, and we will find our way\n"
        "\n"
        "Years will keep on turning and the candles will burn down\n"
        "But the love we built together is the steadiest sound\n"
        "Hold me close, my dear, until the morning shines anew\n"
        "Every quiet promise of my song belongs to you\n"
    ),
    'angry_song': (
        "You lied to me again and now the silence fills the room\n"
        "Every word you ever said was nothing but perfume\n"
        "I am tired of your shadows, I am tired of your games\n"
        "I am burning every letter that you ever signed your name\n"
        "\n"
        "Get out, get out, I cannot stand the way you smile\n"
        "Get out, get out, you have not been honest in a while\n"
        "I am breaking down the door that I once built for you\n"
        "I am tearing up the pictures and the promises too\n"
        "\n"
        "You can take your golden rings and your hollow little crown\n"
        "You can drag your phony halo to some other lonely town\n"
        "I will scream until the windows of this empty house all crack\n"
        "I will not say I am sorry and I will not take it back\n"
        "\n"
        "All the years I gave you turned to ashes on the floor\n"
        "All the trust I had is rotting in a locked and rusted drawer\n"
        "I am wide awake and angry and I see you for the first time\n"
        "Every cruel little secret that you whispered was a crime\n"
        "\n"
        "So get out, get out, slam the door and never call\n"
        "I am stronger than the silence and I am taller than your wall\n"
        "I am taller than your wall\n"
    ),
    'calm_song': (
        "Soft piano in the morning, gentle rain against the glass\n"
        "Empty kitchen, empty hallway, watch the quiet hours pass\n"
        "Nothing said and nothing needed, only colour, only sound\n"
        "Steam is rising from a teacup as the world keeps turning round\n"
        "\n"
        "Birds outside are humming patterns older than the road\n"
        "Patterns drifting through the curtain in a slow and silver code\n"
        "I am sitting at the window with a notebook on my knee\n"
        "Drawing lines that never finish, drawing rivers, drawing seas\n"
        "\n"
        "Through the doorway runs a cat that does not have a name\n"
        "Through the chimney rolls a cloud that never looks the same\n"
        "On the table sits a candle that has never once been lit\n"
        "On the carpet sleeps a shadow that prefers to simply sit\n"
        "\n"
        "Hours move like honey, slow and warm and amber bright\n"
        "Afternoon will turn to evening, evening turn to gentle night\n"
        "I will close the wooden shutters, I will fold the linen sheet\n"
        "I will listen to the floorboards as they sing beneath my feet\n"
        "\n"
        "Soft piano in the evening, gentle rain against the door\n"
        "Empty hallway, empty kitchen, dreaming on the wooden floor\n"
        "Nothing said and nothing needed, only colour, only sound\n"
        "Steam is rising from a teacup as the world keeps turning round\n"
    ),
}


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 0.0 and nb <= 0.0:
        return 1.0
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


@pytest.mark.integration
def test_real_lyrics_analysis_runs_and_matches_expected_vectors(monkeypatch):
    """Runs analyze_lyrics with a fake API hit (deterministic English text)
    and checks the e5 embedding + axis vector against pre-recorded values
    via cosine similarity (threshold = 0.99).
    """
    project_root = Path(__file__).resolve().parents[1]
    models_dir = project_root / 'test' / 'models'
    expected_path = project_root / 'test' / 'lyrics_expected.json'

    # Heavy deps required by the e5 path. Skip cleanly if missing.
    try:
        import torch  # noqa: F401
    except Exception as exc:  # pragma: no cover
        pytest.skip(f'torch not importable: {exc}')
    try:
        from transformers import AutoTokenizer  # noqa: F401
    except Exception as exc:  # pragma: no cover
        pytest.skip(f'transformers not importable: {exc}')

    # The flat e5 directory must exist (extracted from lyrics_model.tar.gz).
    e5_dir = models_dir / 'e5-base-v2'
    if not (e5_dir / 'config.json').exists():
        pytest.skip(
            f'e5-base-v2 model directory not found at {e5_dir}. '
            f'In CI the workflow extracts lyrics_model.tar.gz from release '
            f'v4.0.0-model into test/models/. For a local run download and '
            f'extract it manually.'
        )

    # ---- Force offline / CPU / no LLM / no real HTTP API ------------------
    os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('HF_DATASETS_OFFLINE', '1')
    os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
    os.environ.setdefault('HF_XET_DISABLE', '1')

    os.environ['LYRICS_API_ENABLE'] = 'true'   # keep API path enabled
    os.environ['LYRICS_LLM_ENABLED'] = 'false'  # no Qwen cleanup
    os.environ['LYRICS_USE_GPU'] = 'false'

    os.environ['LYRICS_MODEL_DIR'] = str(models_dir)
    os.environ['LYRICS_DEFAULT_TOPIC_EMBEDDING_CACHE_DIR'] = str(e5_dir)

    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    import config
    config.LYRICS_API_ENABLE = True
    config.LYRICS_LLM_ENABLED = False
    config.LYRICS_USE_GPU = 'false'
    config.LYRICS_MODEL_DIR = str(models_dir)
    config.LYRICS_DEFAULT_TOPIC_EMBEDDING_CACHE_DIR = str(e5_dir)

    from lyrics import lyrics_transcriber
    from lyrics.lyrics_transcriber import analyze_lyrics, axis_columns

    # ---- Fake the external API: STEP 0 returns our hand-written text -----
    current = {'name': None}

    def _fake_fetch_remote_lyrics(artist, track, total_budget=10.0):
        text = FAKE_LYRICS_BY_TRACK.get(current['name'])
        assert text is not None, f'No fake lyrics registered for {current["name"]!r}'
        return text

    monkeypatch.setattr(lyrics_transcriber, 'fetch_remote_lyrics',
                        _fake_fetch_remote_lyrics)

    # Defensive: passthrough translator so a stray langdetect misclassification
    # can never trigger a Marian download in offline CI.
    monkeypatch.setattr(lyrics_transcriber, '_translate_to_english',
                        lambda text, src_lang: text)

    # ---- Recording / replay logic -----------------------------------------
    explicit_record = os.environ.get('LYRICS_RECORD_EXPECTED', '').lower() in ('1', 'true', 'yes')
    record_mode = explicit_record or not expected_path.exists()
    expected = {}
    if record_mode and not explicit_record:
        print(f'\n[lyrics-test] {expected_path.name} not found - entering RECORD mode.')
    if not record_mode:
        with expected_path.open('r', encoding='utf-8') as fh:
            expected = json.load(fh)

    recorded = {}
    failures = []
    expected_axis_dim = len(axis_columns())

    for track_name in FAKE_LYRICS_BY_TRACK:
        current['name'] = track_name
        print(f'\n{"=" * 80}')
        print(f'=== Analyzing fake lyrics: {track_name}')
        print(f'{"=" * 80}')

        # No source_path / audio: pipeline goes through the API branch only.
        result = analyze_lyrics(
            audio=None,
            sr=None,
            source_path=None,
            use_llm_cleanup=False,
            artist='AudioMuseTest',
            track=track_name,
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
