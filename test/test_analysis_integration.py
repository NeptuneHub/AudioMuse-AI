# Real song analysis integration test
# 1. Create a virtual environment:
#      python3 -m venv test/.venv
#
# 2. Activate the virtual environment:
#      source test/.venv/bin/activate
#
# 3. Install requirements:
#      pip install -r test/requirements.txt
# 4. Run this script:
#      pytest test/test_analysis_integration.py -s -q
#
# Note: Test audio files should be in test/songs/
#       ONNX models should be in test/models/
import sys
import types
from pathlib import Path
import importlib
import json
import pytest


def _ensure_stubs():
    """Insert minimal runtime stubs for optional heavy packages so importing
    `tasks.analysis` won't fail during the integration test.
    """
    # google.genai stub (new API)
    if 'google.genai' not in sys.modules:
        genai = types.ModuleType('google.genai')
        
        class _Client:
            def __init__(self, api_key=None, **kwargs):
                self.api_key = api_key
                self.models = self
            
            def generate_content(self, model=None, contents=None, config=None, **kwargs):
                """Mock generate_content for new google-genai API"""
                return types.SimpleNamespace(text='[stubbed google.genai]')
        
        genai.Client = _Client
        genai.types = types.SimpleNamespace(
            GenerateContentConfig=lambda *a, **k: None,
            HttpOptions=lambda *a, **k: None
        )
        sys.modules['google.genai'] = genai

    # mistralai stub
    if 'mistralai' not in sys.modules:
        mistral_mod = types.ModuleType('mistralai')
        class Mistral:
            def __init__(self, api_key=None):
                class Chat:
                    def complete(self, *a, **k):
                        message = types.SimpleNamespace(content='[stubbed mistral]')
                        choice = types.SimpleNamespace(message=message)
                        return types.SimpleNamespace(choices=[choice])
                self.chat = Chat()
        mistral_mod.Mistral = Mistral
        sys.modules['mistralai'] = mistral_mod

    # voyager stub
    if 'voyager' not in sys.modules:
        voyager_mod = types.ModuleType('voyager')
        voyager_mod.Space = types.SimpleNamespace(Cosine=0, Euclidean=1, InnerProduct=2)
        class RecallError(Exception):
            pass
        voyager_mod.RecallError = RecallError
        class _Index:
            def __init__(self, *a, **k):
                self.ef = 0
            @staticmethod
            def load(stream):
                return _Index()
            def save(self, path):
                with open(path, 'wb'):
                    pass
            def add_items(self, arr, ids=None):
                return None
            def get_vector(self, idx):
                import numpy as _np
                return _np.zeros(128, dtype=_np.float32)
            def query(self, vec, k=10):
                return ([], [])
            def __len__(self):
                return 0
        voyager_mod.Index = _Index
        sys.modules['voyager'] = voyager_mod


def _validate_analysis_result(result, expected, track_name, tol=1e-3):
    """Helper to validate analysis results match expected values."""
    # Scalar checks
    assert abs(float(result.get('tempo', 0)) - expected['tempo']) <= tol, \
        f"{track_name}: tempo mismatch: {result.get('tempo')} != {expected['tempo']}"
    assert result.get('key') == expected['key'], \
        f"{track_name}: key mismatch: {result.get('key')} != {expected['key']}"
    assert result.get('scale') == expected['scale'], \
        f"{track_name}: scale mismatch: {result.get('scale')} != {expected['scale']}"

    # Other scalar features
    for k in ['energy', 'danceable', 'aggressive', 'happy', 'party', 'relaxed', 'sad']:
        assert k in result, f"{track_name}: missing feature: {k}"
        assert abs(float(result[k]) - expected[k]) <= tol, \
            f"{track_name}: feature {k} mismatch: {result[k]} != {expected[k]}"

    # Moods: compare each expected mood
    got_moods = result.get('moods', {})
    for mood, exp_val in expected['moods'].items():
        assert mood in got_moods, f"{track_name}: missing mood: {mood}"
        got_val = float(got_moods[mood])
        assert abs(got_val - exp_val) <= tol, \
            f"{track_name}: mood {mood} mismatch: {got_val} != {exp_val}"


@pytest.mark.integration
def test_real_analysis_runs_and_returns_expected_shape():
    """Integration test: runs analyze_track with ONNX models in test/models.
    
    Validates exact analysis results for three test tracks:
    1. Art Flower - Art Flower - Creamy Snowflakes.mp3
    2. Aaron Dunn - Minuet - Notebook for Anna Magdalena.mp3
    3. Michael Hawley - Sonata 'Waldstein', Op. 53 - II. Introduzione-Adagio molto.mp3

    This test is skipped if models are not present or onnxruntime is not
    importable in the environment. It injects lightweight stubs for optional
    AI/voyager libraries so module import succeeds.
    """
    project_root = Path(__file__).resolve().parents[1]
    models_dir = project_root / 'test' / 'models'
    required = [
        'msd-musicnn-1.onnx', 'msd-msd-musicnn-1.onnx', 'danceability-msd-musicnn-1.onnx',
        'mood_aggressive-msd-musicnn-1.onnx', 'mood_happy-msd-musicnn-1.onnx',
        'mood_party-msd-musicnn-1.onnx', 'mood_relaxed-msd-musicnn-1.onnx', 'mood_sad-msd-musicnn-1.onnx'
    ]
    missing = [p for p in required if not (models_dir / p).exists()]
    if missing:
        pytest.skip(f"Real ONNX models not present in test/models (missing: {missing})")

    # Ensure onnxruntime is available
    try:
        import onnxruntime as ort  # noqa: F401
    except Exception as e:
        pytest.skip(f"onnxruntime not importable in this environment: {e}")

    # Ensure the project is importable and provide stubs before import
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    _ensure_stubs()

    # Now import tasks.analysis and run the real analysis on the repo MP3
    import importlib
    analysis = importlib.import_module('tasks.analysis')
    importlib.reload(analysis)

    model_paths = {
        'embedding': str(models_dir / 'msd-musicnn-1.onnx'),
        'prediction': str(models_dir / 'msd-msd-musicnn-1.onnx'),
        'danceable': str(models_dir / 'danceability-msd-musicnn-1.onnx'),
        'aggressive': str(models_dir / 'mood_aggressive-msd-musicnn-1.onnx'),
        'happy': str(models_dir / 'mood_happy-msd-musicnn-1.onnx'),
        'party': str(models_dir / 'mood_party-msd-musicnn-1.onnx'),
        'relaxed': str(models_dir / 'mood_relaxed-msd-musicnn-1.onnx'),
        'sad': str(models_dir / 'mood_sad-msd-musicnn-1.onnx'),
    }

    # Print existence info (helpful when running the test)
    print('Models exist:')
    for k, p in model_paths.items():
        print(f"  {k}: {Path(p).exists()} -> {p}")

    # Define test tracks with their expected values
    test_tracks = [
        {
            'path': project_root / 'test' / 'songs' / 'Art Flower - Art Flower - Creamy Snowflakes.mp3',
            'name': 'Art Flower - Art Flower - Creamy Snowflakes.mp3',
            'expected': {
                "tempo": 75.0,
                "key": "G",
                "scale": "minor",
                "moods": {
                    "rock": 0.5461238026618958,
                    "pop": 0.520070493221283,
                    "alternative": 0.5178204774856567,
                    "indie": 0.516494870185852,
                    "electronic": 0.5195158123970032,
                    "female vocalists": 0.5169416666030884,
                    "dance": 0.5022153258323669,
                    "00s": 0.5027338862419128,
                    "alternative rock": 0.5071790814399719,
                    "jazz": 0.5160742998123169,
                    "beautiful": 0.507721483707428,
                    "metal": 0.5058281421661377,
                    "chillout": 0.5148181319236755,
                    "male vocalists": 0.5025433301925659,
                    "classic rock": 0.509633481502533,
                    "soul": 0.5050523281097412,
                    "indie rock": 0.5048931241035461,
                    "Mellow": 0.5051383376121521,
                    "electronica": 0.5060535073280334,
                    "80s": 0.5089928507804871,
                    "folk": 0.5082688331604004,
                    "90s": 0.505206823348999,
                    "chill": 0.5040011405944824,
                    "instrumental": 0.5402882099151611,
                    "punk": 0.5017344951629639,
                    "oldies": 0.5039896965026855,
                    "blues": 0.5043821930885315,
                    "hard rock": 0.5042359828948975,
                    "ambient": 0.5251195430755615,
                    "acoustic": 0.5029069185256958,
                    "experimental": 0.5074111223220825,
                    "female vocalist": 0.5019606947898865,
                    "guitar": 0.5052618980407715,
                    "Hip-Hop": 0.5007167458534241,
                    "70s": 0.5039721131324768,
                    "party": 0.5003747940063477,
                    "country": 0.5127668976783752,
                    "easy listening": 0.505574107170105,
                    "sexy": 0.5008877515792847,
                    "catchy": 0.5005943775177002,
                    "funk": 0.5014483332633972,
                    "electro": 0.5010225176811218,
                    "heavy metal": 0.5027199983596802,
                    "Progressive rock": 0.5230364799499512,
                    "60s": 0.5027734637260437,
                    "rnb": 0.500874400138855,
                    "indie pop": 0.5025118589401245,
                    "sad": 0.5023919939994812,
                    "House": 0.5007247924804688,
                    "happy": 0.5004531145095825
                },
                "energy": 0.11941074579954147,
                "danceable": 0.09910931438207626,
                "aggressive": 0.021448878571391106,
                "happy": 0.06873109191656113,
                "party": 0.045930881053209305,
                "relaxed": 0.9612494111061096,
                "sad": 0.8027413487434387,
            }
        },
        {
            'path': project_root / 'test' / 'songs' / "Aaron Dunn - Minuet - Notebook for Anna Magdalena.mp3",
            'name': 'Aaron Dunn - Minuet - Notebook for Anna Magdalena.mp3',
            'expected': {
                "tempo": 125.0,
                "key": "E",
                "scale": "minor",
                "moods": {
                    "rock": 0.5274592638015747,
                    "pop": 0.5133988261222839,
                    "alternative": 0.5120611190795898,
                    "indie": 0.5155648589134216,
                    "electronic": 0.5100196599960327,
                    "female vocalists": 0.515810489654541,
                    "dance": 0.5016465187072754,
                    "00s": 0.5020167827606201,
                    "alternative rock": 0.5035533905029297,
                    "jazz": 0.5606123805046082,
                    "beautiful": 0.5064935684204102,
                    "metal": 0.5029664039611816,
                    "chillout": 0.5055583119392395,
                    "male vocalists": 0.5023089647293091,
                    "classic rock": 0.5080583095550537,
                    "soul": 0.5059599280357361,
                    "indie rock": 0.5037187337875366,
                    "Mellow": 0.5063819885253906,
                    "electronica": 0.5033227801322937,
                    "80s": 0.5062050223350525,
                    "folk": 0.5289740562438965,
                    "90s": 0.5027812123298645,
                    "chill": 0.502841591835022,
                    "instrumental": 0.5431712865829468,
                    "punk": 0.5024588704109192,
                    "oldies": 0.504041850566864,
                    "blues": 0.5085024833679199,
                    "hard rock": 0.5023196935653687,
                    "ambient": 0.5106694102287292,
                    "acoustic": 0.5108007192611694,
                    "experimental": 0.5092200040817261,
                    "female vocalist": 0.5024746060371399,
                    "guitar": 0.5087099671363831,
                    "Hip-Hop": 0.500925600528717,
                    "70s": 0.5061967372894287,
                    "party": 0.5003656148910522,
                    "country": 0.5047071576118469,
                    "easy listening": 0.5051171183586121,
                    "sexy": 0.5006621479988098,
                    "catchy": 0.5005181431770325,
                    "funk": 0.502180814743042,
                    "electro": 0.5016681551933289,
                    "heavy metal": 0.50184166431427,
                    "Progressive rock": 0.5151917338371277,
                    "60s": 0.5032697319984436,
                    "rnb": 0.5013742446899414,
                    "indie pop": 0.5026872158050537,
                    "sad": 0.5027293562889099,
                    "House": 0.5008860230445862,
                    "happy": 0.5007311105728149
                },
                "energy": 0.006939841900020838,
                "danceable": 0.017882201820611954,
                "aggressive": 0.000734207103960216,
                "happy": 0.006147411186248064,
                "party": 0.00013406496145762503,
                "relaxed": 0.9952480792999268,
                "sad": 0.980929970741272
            }
        },
        {
            'path': project_root / 'test' / 'songs' / "Michael Hawley - Sonata 'Waldstein', Op. 53 - II. Introduzione-Adagio molto.mp3",
            'name': "Michael Hawley - Sonata 'Waldstein', Op. 53 - II. Introduzione-Adagio molto.mp3",
            'expected': {
                "tempo": 104.16666666666667,
                "key": "A#",
                "scale": "minor",
                "moods": {
                    "rock": 0.546576976776123,
                    "pop": 0.5153129696846008,
                    "alternative": 0.5214908123016357,
                    "indie": 0.5249019861221313,
                    "electronic": 0.5161430239677429,
                    "female vocalists": 0.5169497728347778,
                    "dance": 0.5027590990066528,
                    "00s": 0.5031359195709229,
                    "alternative rock": 0.5089637041091919,
                    "jazz": 0.5305668711662292,
                    "beautiful": 0.5074053406715393,
                    "metal": 0.5052326321601868,
                    "chillout": 0.5064440369606018,
                    "male vocalists": 0.5029566884040833,
                    "classic rock": 0.5111653208732605,
                    "soul": 0.5066967606544495,
                    "indie rock": 0.5085793137550354,
                    "Mellow": 0.5061439275741577,
                    "electronica": 0.5050395727157593,
                    "80s": 0.5062342882156372,
                    "folk": 0.5180213451385498,
                    "90s": 0.5045623183250427,
                    "chill": 0.5029629468917847,
                    "instrumental": 0.5336787700653076,
                    "punk": 0.5095312595367432,
                    "oldies": 0.5031309127807617,
                    "blues": 0.5071375370025635,
                    "hard rock": 0.5053707957267761,
                    "ambient": 0.5182665586471558,
                    "acoustic": 0.5080854296684265,
                    "experimental": 0.5142408013343811,
                    "female vocalist": 0.5023537874221802,
                    "guitar": 0.5049938559532166,
                    "Hip-Hop": 0.5032581686973572,
                    "70s": 0.5055437684059143,
                    "party": 0.5004801750183105,
                    "country": 0.5038270950317383,
                    "easy listening": 0.5037663578987122,
                    "sexy": 0.5008610486984253,
                    "catchy": 0.5006652474403381,
                    "funk": 0.5019800066947937,
                    "electro": 0.502349317073822,
                    "heavy metal": 0.5034934282302856,
                    "Progressive rock": 0.5157937407493591,
                    "60s": 0.5035749077796936,
                    "rnb": 0.5019146800041199,
                    "indie pop": 0.5033155679702759,
                    "sad": 0.5038461089134216,
                    "House": 0.5015008449554443,
                    "happy": 0.5005825161933899
                },
                "energy": 0.01083404291421175,
                "danceable": 0.07516419887542725,
                "aggressive": 0.07692281156778336,
                "happy": 0.015692999586462975,
                "party": 0.0005335173336789012,
                "relaxed": 0.9905794858932495,
                "sad": 0.9709755182266235
            }
        }
    ]

    tol = 1e-3
    
    # Test each track
    for track_info in test_tracks:
        track_path = track_info['path']
        track_name = track_info['name']
        expected = track_info['expected']
        
        if not track_path.exists():
            print(f'\n{track_name} not present in test/; skipping.')
            continue
        
        print(f'\n=== Analyzing: {track_name} ===')
        
        # Run analysis
        result, embedding = analysis.analyze_track(str(track_path), analysis.MOOD_LABELS, model_paths)
        
        # Print result for visibility
        try:
            print(f'\n{track_name} analysis result:')
            print(json.dumps(result, indent=2))
        except Exception:
            print(f'\n{track_name} analysis result (repr):')
            print(repr(result))
        
        # Validate results
        assert result is not None, f'{track_name}: analyze_track returned None for analysis_result'
        assert isinstance(result, dict), f'{track_name}: result is not a dict'
        assert 'moods' in result and isinstance(result['moods'], dict), f'{track_name}: moods missing or invalid'
        
        # Validate embedding
        assert embedding is not None, f'{track_name}: analyze_track returned None for embedding'
        assert hasattr(embedding, 'shape') and embedding.ndim == 1, \
            f'{track_name}: expected 1-D embedding, got ndim={getattr(embedding, "ndim", None)}'
        emb_dim = int(embedding.shape[0])
        print(f'{track_name}: embedding dimension = {emb_dim}')
        assert emb_dim > 0, f'{track_name}: Unexpected embedding dimension: {emb_dim}'
        
        # Validate exact values
        _validate_analysis_result(result, expected, track_name, tol)
        print(f'{track_name}: ✓ All validations passed')
