# CLAP analysis integration test
# 1. Create a virtual environment:
#      python3 -m venv test/.venv
#
# 2. Activate the virtual environment:
#      source test/.venv/bin/activate
#
# 3. Install requirements:
#      pip install -r test/requirements.txt
# 4. Run this script:
#      pytest test/test_clap_analysis_integration.py -s -q
#
# Note: Test audio files should be in test/songs/
#       CLAP ONNX model should be in test/models/
import sys
import types
from pathlib import Path
import json
import pytest
import numpy as np


def _ensure_stubs():
    """Insert minimal runtime stubs for optional heavy packages (currently none needed)."""
    # No stubs needed - transformers is installed in test requirements
    pass


@pytest.mark.integration
def test_clap_analysis_runs_and_shows_output():
    """Integration test: runs CLAP analysis on test tracks and validates results.
    
    This test analyzes three test tracks with the CLAP model:
    1. Art Flower - Art Flower - Creamy Snowflakes.mp3
    2. Aaron Dunn - Minuet - Notebook for Anna Magdalena.mp3
    3. Michael Hawley - Sonata 'Waldstein', Op. 53 - II. Introduzione-Adagio molto.mp3

    Validates that cosine similarities match expected values (tolerance: 0.001).
    """
    
    # Expected cosine similarities for each track and query
    # These are the reference values that must be maintained
    expected_similarities = {
        'Art Flower - Art Flower - Creamy Snowflakes.mp3': {
            'rock': 0.333237,
            'classic piano song': 0.043963,
            'electro': 0.153350,
            'acoustic': 0.129170
        },
        'Aaron Dunn - Minuet - Notebook for Anna Magdalena.mp3': {
            'rock': 0.113808,
            'classic piano song': 0.389167,
            'electro': 0.078039,
            'acoustic': 0.246442
        },
        "Michael Hawley - Sonata 'Waldstein', Op. 53 - II. Introduzione-Adagio molto.mp3": {
            'rock': 0.095823,
            'classic piano song': 0.349463,
            'electro': 0.042161,
            'acoustic': 0.066694
        }
    }
    project_root = Path(__file__).resolve().parents[1]
    models_dir = project_root / 'test' / 'models'
    clap_model = models_dir / 'clap_model.onnx'
    
    if not clap_model.exists():
        pytest.skip(f"CLAP model not present in test/models: {clap_model}")

    # Ensure onnxruntime is available
    try:
        import onnxruntime as ort  # noqa: F401
    except Exception as e:
        pytest.skip(f"onnxruntime not importable: {e}")

    # Ensure librosa is available
    try:
        import librosa  # noqa: F401
    except Exception as e:
        pytest.skip(f"librosa not importable: {e}")

    # Ensure the project is importable and provide stubs
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    _ensure_stubs()

    # Override config before importing CLAP analyzer
    import config
    config.CLAP_MODEL_PATH = str(clap_model)
    config.CLAP_ENABLED = True

    # Import CLAP analyzer
    from tasks.clap_analyzer import analyze_audio_file, get_text_embedding

    # Define test queries
    test_queries = [
        "rock",
        "classic piano song",
        "electro",
        "acoustic"
    ]

    # Define test tracks
    test_tracks = [
        'Art Flower - Art Flower - Creamy Snowflakes.mp3',
        'Aaron Dunn - Minuet - Notebook for Anna Magdalena.mp3',
        "Michael Hawley - Sonata 'Waldstein', Op. 53 - II. Introduzione-Adagio molto.mp3"
    ]

    for track_name in test_tracks:
        track_path = project_root / 'test' / 'songs' / track_name
        
        if not track_path.exists():
            print(f'\n{track_name} not present in test/songs/; skipping.')
            continue
        
        print(f'\n{"="*80}')
        print(f'=== Analyzing with CLAP: {track_name} ===')
        print(f'{"="*80}')
        
        try:
            # Run CLAP analysis (returns tuple: embedding, duration, num_segments)
            embedding, duration, num_segments = analyze_audio_file(str(track_path))
            
            # Validate embedding
            assert embedding is not None, f'{track_name}: CLAP returned None'
            assert isinstance(embedding, np.ndarray), f'{track_name}: embedding not numpy array'
            assert embedding.ndim == 1, f'{track_name}: expected 1-D embedding, got {embedding.ndim}-D'
            
            emb_dim = embedding.shape[0]
            print(f'\nAudio duration: {duration:.2f} seconds')
            print(f'Number of segments: {num_segments}')
            print(f'Embedding dimension: {emb_dim}')
            
            # Test text queries and compute cosine similarities
            print(f'\n{"="*80}')
            print(f'Text Query Similarities:')
            print(f'{"="*80}')
            
            expected_sims = expected_similarities.get(track_name, {})
            all_passed = True
            
            for query in test_queries:
                text_embedding = get_text_embedding(query)
                
                if text_embedding is None:
                    print(f'  {query:25s} - Failed to compute text embedding')
                    pytest.fail(f'{track_name}: Failed to compute text embedding for query "{query}"')
                    continue
                
                # Compute cosine similarity (dot product of normalized vectors)
                cosine_sim = np.dot(embedding, text_embedding)
                
                # Check against expected value
                expected_sim = expected_sims.get(query)
                if expected_sim is not None:
                    diff = abs(cosine_sim - expected_sim)
                    tolerance = 0.001
                    passed = diff <= tolerance
                    status = "✓" if passed else "✗"
                    
                    print(f'  {query:25s} - Cosine Similarity: {cosine_sim:.6f} (expected: {expected_sim:.6f}) {status}')
                    
                    if not passed:
                        all_passed = False
                        print(f'    ERROR: Difference {diff:.6f} exceeds tolerance {tolerance}')
                else:
                    print(f'  {query:25s} - Cosine Similarity: {cosine_sim:.6f} (no expected value)')
            
            # Fail test if any similarities don't match
            if not all_passed:
                pytest.fail(f'{track_name}: One or more cosine similarities differ from expected values')
            
            print(f'\n{track_name}: ✓ CLAP analysis completed successfully')
            
        except Exception as e:
            print(f'\n{track_name}: ✗ CLAP analysis failed with error:')
            print(f'  {type(e).__name__}: {e}')
            import traceback
            traceback.print_exc()
            raise


if __name__ == '__main__':
    # Allow running directly with: python test/test_clap_analysis_integration.py
    pytest.main([__file__, '-s', '-v'])
