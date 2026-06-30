import os
import sys
from pathlib import Path
import pytest
import numpy as np


def _ensure_stubs():
    # no-op hook: integration run uses real modules, no stubbing needed
    pass


@pytest.mark.integration
def test_clap_analysis_runs_and_shows_output():
    expected_similarities = {
        'Art Flower - Art Flower - Creamy Snowflakes.mp3': {
            'rock': 0.328612,
            'classic piano song': 0.056155,
            'electro': 0.171881,
            'acoustic': 0.142844,
        },
        'Aaron Dunn - Minuet - Notebook for Anna Magdalena.mp3': {
            'rock': 0.121994,
            'classic piano song': 0.405505,
            'electro': 0.080324,
            'acoustic': 0.274541,
        },
        "Michael Hawley - Sonata 'Waldstein', Op. 53 - II. Introduzione-Adagio molto.mp3": {
            'rock': 0.123132,
            'classic piano song': 0.445012,
            'electro': 0.052204,
            'acoustic': 0.116278,
        },
    }
    project_root = Path(__file__).resolve().parents[2]
    models_dir = project_root / 'test' / 'models'
    clap_audio_model = models_dir / 'model_epoch_36.onnx'
    clap_text_model = models_dir / 'clap_text_model.onnx'

    if not clap_audio_model.exists():
        pytest.skip(f"DCLAP audio model not present in test/models: {clap_audio_model}")

    if not clap_text_model.exists():
        pytest.skip(f"CLAP text model not present in test/models: {clap_text_model}")

    try:
        import onnxruntime as ort  # noqa: F401
    except Exception as e:
        pytest.skip(f"onnxruntime not importable: {e}")

    try:
        import librosa  # noqa: F401
    except Exception as e:
        pytest.skip(f"librosa not importable: {e}")

    os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('HF_DATASETS_OFFLINE', '1')

    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    _ensure_stubs()

    from transformers import AutoTokenizer

    AutoTokenizer.from_pretrained('roberta-base', local_files_only=True)

    import config

    config.CLAP_AUDIO_MODEL_PATH = str(clap_audio_model)
    config.CLAP_TEXT_MODEL_PATH = str(clap_text_model)
    config.CLAP_ENABLED = True

    from tasks.clap_analyzer import analyze_audio_file, get_text_embedding

    test_queries = ["rock", "classic piano song", "electro", "acoustic"]

    test_tracks = [
        'Art Flower - Art Flower - Creamy Snowflakes.mp3',
        'Aaron Dunn - Minuet - Notebook for Anna Magdalena.mp3',
        "Michael Hawley - Sonata 'Waldstein', Op. 53 - II. Introduzione-Adagio molto.mp3",
    ]

    for track_name in test_tracks:
        track_path = project_root / 'test' / 'songs' / track_name

        if not track_path.exists():
            print(f'\n{track_name} not present in test/songs/; skipping.')
            continue

        print(f'\n{"=" * 80}')
        print(f'=== Analyzing with CLAP: {track_name} ===')
        print(f'{"=" * 80}')

        try:
            embedding, duration, num_segments = analyze_audio_file(str(track_path))

            assert embedding is not None, f'{track_name}: CLAP returned None'
            assert isinstance(embedding, np.ndarray), f'{track_name}: embedding not numpy array'
            assert embedding.ndim == 1, (
                f'{track_name}: expected 1-D embedding, got {embedding.ndim}-D'
            )

            emb_dim = embedding.shape[0]
            print(f'\nAudio duration: {duration:.2f} seconds')
            print(f'Number of segments: {num_segments}')
            print(f'Embedding dimension: {emb_dim}')

            print(f'\n{"=" * 80}')
            print('Text Query Similarities:')
            print(f'{"=" * 80}')

            expected_sims = expected_similarities.get(track_name, {})
            all_passed = True

            for query in test_queries:
                text_embedding = None
                last_exc = None
                for attempt in range(2):
                    try:
                        text_embedding = get_text_embedding(query)
                        last_exc = None
                        break
                    except Exception as e:
                        last_exc = e
                        if attempt == 0:
                            continue
                if last_exc is not None:
                    pytest.fail(f"CLAP text model unavailable after retry: {last_exc}")

                if text_embedding is None:
                    print(f'  {query:25s} - Failed to compute text embedding')
                    pytest.fail(
                        f'{track_name}: Failed to compute text embedding for query "{query}"'
                    )
                    continue

                cosine_sim = np.dot(embedding, text_embedding)

                expected_sim = expected_sims.get(query)
                if expected_sim is not None:
                    diff = abs(cosine_sim - expected_sim)
                    tolerance = 0.001
                    passed = diff <= tolerance
                    status = "OK" if passed else "X"

                    print(
                        f'  {query:25s} - Cosine Similarity: {cosine_sim:.6f} (expected: {expected_sim:.6f}) {status}'
                    )

                    if not passed:
                        all_passed = False
                        print(f'    ERROR: Difference {diff:.6f} exceeds tolerance {tolerance}')
                else:
                    print(
                        f'  {query:25s} - Cosine Similarity: {cosine_sim:.6f} (no expected value)'
                    )

            if not all_passed:
                pytest.fail(
                    f'{track_name}: One or more cosine similarities differ from expected values'
                )

            print(f'\n{track_name}: OK CLAP analysis completed successfully')

        except Exception as e:
            print(f'\n{track_name}: X CLAP analysis failed with error:')
            print(f'  {type(e).__name__}: {e}')
            import traceback

            traceback.print_exc()
            raise


if __name__ == '__main__':
    pytest.main([__file__, '-s', '-v'])
