"""Small lyrics analysis orchestrator.

Models are assumed to be already present inside the container; this module never
downloads anything. It exposes a single high level entry point ``analyze_lyrics``
plus the cached model loaders used by the worker bootstrap.

Pipeline (each step emits a ``STEP X start`` and ``STEP X end`` log line):

    STEP 1  load / clip audio (max 4 minutes)
    STEP 2  whisper transcription
    STEP 3  language detection
    STEP 4  optional translation to English (MarianMT)
    STEP 5  qwen cleanup over 50-word chunks
    STEP 6  e5 embedding + axis scoring
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import signal
import tempfile
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# Disable the HuggingFace xet/XetHub Rust transport before any HF library is
# imported. The xet log path ($HF_HOME/xet/logs = /app/.cache/huggingface/xet/logs)
# is read-only in the container and causes noisy "Permission denied" errors.
# HF_HOME itself stays at /app/.cache/huggingface so all bundled models resolve normally.
os.environ['HF_HUB_DISABLE_XET'] = '1'
os.environ['HF_XET_DISABLE'] = '1'

import numpy as np

try:
    import soundfile as sf
except ImportError:  # pragma: no cover
    sf = None

try:
    import librosa
except ImportError:  # pragma: no cover
    librosa = None

try:
    import whisper
except ImportError:  # pragma: no cover
    whisper = None

try:
    from llama_cpp import Llama
except ImportError:  # pragma: no cover
    Llama = None

try:
    from langdetect import detect_langs, DetectorFactory
except ImportError:  # pragma: no cover
    detect_langs = None
    DetectorFactory = None

try:
    from transformers import AutoModel, AutoModelForSeq2SeqLM, AutoTokenizer
except ImportError:  # pragma: no cover
    AutoModel = None
    AutoModelForSeq2SeqLM = None
    AutoTokenizer = None

try:
    import torch
except Exception:  # pragma: no cover - CUDA-build torch can raise OSError on dlopen
    torch = None

try:
    from silero_vad import load_silero_vad, get_speech_timestamps
except Exception:  # pragma: no cover - silero pulls torchaudio which can fail to dlopen libcudart
    load_silero_vad = None
    get_speech_timestamps = None

if DetectorFactory is not None:
    DetectorFactory.seed = 0

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_SAMPLE_RATE = 16000
MAX_AUDIO_SECONDS = 240.0          # never feed whisper more than 4 minutes
MAX_WORDS_PER_CHUNK = 50           # llm cleanup chunk size
MIN_WORDS_FOR_CLEANUP = 50         # below this, skip cleanup (matches stand-alone behavior)
MIN_WORDS_FOR_EMBEDDING = 50       # below this, treat song as having no usable lyrics

MUSIC_ANALYSIS_AXES = {
    "AXIS_1_SETTING": {
        "description": "The primary physical or environmental container of the song.",
        "labels": {
            "URBAN": "Cities, skyscrapers, streets, neon, traffic, and industrial zones.",
            "WILDERNESS": "Nature in its raw state: forests, mountains, oceans, and deserts.",
            "INTERIOR": "Enclosed private or public spaces: rooms, bars, hallways, or houses.",
            "TRANSIT": "Active movement: cars, trains, planes, or walking the open road.",
            "EXTRATERRESTRIAL": "Outer space, planetary bodies, and the cosmic void.",
            "SURREAL_ABSTRACT": "Non-physical realms, dreams, or places that defy physics.",
        },
    },
    "AXIS_2_SOCIAL_DYNAMIC": {
        "description": "The target or partner of the narrator's communication.",
        "labels": {
            "SOLITARY": "Introspective monologue; the narrator is alone with their thoughts.",
            "ROMANTIC": "Interaction with a lover, crush, or ex-partner.",
            "KINSHIP": "Family structures: parents, children, siblings, or ancestors.",
            "COLLECTIVE": "A crowd, a friend group, 'the youth', or society as a whole.",
            "ADVERSARIAL": "A rival, an enemy, 'the system', or an oppressor.",
            "DIVINE": "A higher power, God, spirits, or the universe itself.",
        },
    },
    "AXIS_3_EMOTIONAL_VALENCE": {
        "description": "The psychological tone (Nostalgia = Retrospective + Melancholic).",
        "labels": {
            "RADIANT": "Joy, euphoria, celebration, and high-energy optimism.",
            "MELANCHOLIC": "Sadness, grief, longing, and quiet despair.",
            "VOLATILE": "Anger, frustration, chaos, and intense restlessness.",
            "VULNERABLE": "Fear, anxiety, paranoia, and the feeling of being exposed.",
            "SERENE": "Acceptance, peace, calmness, and emotional stillness.",
            "NUMB": "Boredom, apathy, emptiness, and emotional detachment.",
        },
    },
    "AXIS_4_NARRATIVE_TEMPORALITY": {
        "description": "The 'When' and 'How' of the lyrical structure.",
        "labels": {
            "RETROSPECTIVE": "Memory-based; looking back at what has passed.",
            "CHRONICLE": "The 'now'; a linear description of events as they happen.",
            "EXISTENTIAL": "Philosophical pondering on concepts like time, life, or death.",
            "STORYTELLING": "Narrating the life or actions of a third-party character/fable.",
            "DIRECT_PLEA": "A targeted message or letter to a 'you' with an immediate goal.",
        },
    },
    "AXIS_5_THEMATIC_WEIGHT": {
        "description": "The gravity and intent behind the lyrical content.",
        "labels": {
            "TRIVIAL": "Lighthearted, casual, and focused on style, fun, or the moment.",
            "MORTAL": "Deeply serious, focused on legacy, life's end, and human struggle.",
            "POLITICAL": "Observation of power, justice, war, and societal mechanics.",
            "SENSORIAL": "Focus on physical indulgence: drinking, dancing, and pleasure.",
        },
    },
}


# ---------------------------------------------------------------------------
# Threading
# ---------------------------------------------------------------------------

def get_lyrics_threads() -> int:
    """Number of CPU threads for Whisper / MarianMT / Qwen inside this process.

    Uses ``os.cpu_count() // 2`` with a floor of two.
    """
    cpus = os.cpu_count() or 2
    return max(2, cpus // 2)


def _apply_thread_env(num_threads: int) -> None:
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = str(num_threads)
    if torch is not None:
        try:
            torch.set_num_threads(num_threads)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Cached model loaders
# ---------------------------------------------------------------------------

_whisper_model = None
_whisper_model_name: Optional[str] = None
_llama_model = None
_llama_model_path: Optional[str] = None
_embedding_tokenizer = None
_embedding_model = None
_embedding_model_name: Optional[str] = None
_axis_label_map: Optional[Dict] = None
_axis_embeddings: Optional[Dict] = None
_marian_cache: Dict[str, Tuple[object, object]] = {}
_lyrics_device_cache: Optional[str] = None
# Serialize model loads so concurrent callers (e.g. axis warm-up + a user
# search request hitting the API at the same time) cannot observe a
# half-initialized BERT model whose parameters are still on the ``meta``
# device. ``transformers`` with ``accelerate`` installed initializes weights
# lazily on ``meta`` and only materializes them at the end of
# ``from_pretrained``; without a lock another thread can grab the partially
# constructed module from the global cache.
_embedding_load_lock = threading.Lock()


def _resolve_lyrics_device() -> str:
    """Return 'cuda' or 'cpu' based on LYRICS_USE_GPU and torch availability."""
    global _lyrics_device_cache
    if _lyrics_device_cache is not None:
        return _lyrics_device_cache
    try:
        from config import LYRICS_USE_GPU
        pref = str(LYRICS_USE_GPU).lower()
    except Exception:
        pref = 'auto'
    if pref == 'false' or torch is None:
        _lyrics_device_cache = 'cpu'
        return _lyrics_device_cache
    try:
        cuda_ok = bool(torch.cuda.is_available())
    except Exception:
        cuda_ok = False
    if pref == 'true':
        _lyrics_device_cache = 'cuda' if cuda_ok else 'cpu'
        if not cuda_ok:
            logger.warning('LYRICS_USE_GPU=true but CUDA is not available; using CPU.')
    else:  # auto
        _lyrics_device_cache = 'cuda' if cuda_ok else 'cpu'
    logger.info('Lyrics compute device resolved: %s', _lyrics_device_cache)
    return _lyrics_device_cache


def load_whisper_model(model_name: str = 'small', device: Optional[str] = None,
                      num_threads: Optional[int] = None):
    global _whisper_model, _whisper_model_name
    if whisper is None:
        raise RuntimeError('openai-whisper is not installed.')

    threads = num_threads or get_lyrics_threads()
    _apply_thread_env(threads)

    resolved_device = device or _resolve_lyrics_device()

    if (_whisper_model is not None
            and _whisper_model_name == model_name
            and getattr(_whisper_model, '_lyrics_device', None) == resolved_device):
        return _whisper_model

    try:
        from config import LYRICS_MODEL_DIR
    except Exception:
        LYRICS_MODEL_DIR = '/app/model'

    local_pt = os.path.join(LYRICS_MODEL_DIR, f'{model_name}.pt')
    target = local_pt if os.path.isfile(local_pt) else model_name
    logger.info('Loading Whisper model %r (device=%s, threads=%s) from %s',
                model_name, resolved_device, threads, target)
    _whisper_model = whisper.load_model(target, device=resolved_device, download_root=LYRICS_MODEL_DIR)
    try:
        setattr(_whisper_model, '_lyrics_device', resolved_device)
    except Exception:
        pass
    _whisper_model_name = model_name
    logger.info('Whisper model %r ready on %s', model_name, resolved_device)
    return _whisper_model


def load_llama_model(model_path: Optional[str] = None,
                     num_threads: Optional[int] = None):
    global _llama_model, _llama_model_path
    if Llama is None:
        raise RuntimeError('llama-cpp-python is not installed.')

    if model_path is None:
        try:
            from config import LYRICS_LLM_MODEL_PATH
            model_path = LYRICS_LLM_MODEL_PATH
        except Exception as exc:
            raise RuntimeError('LYRICS_LLM_MODEL_PATH is not configured.') from exc

    if not os.path.exists(model_path):
        raise RuntimeError(f'LLaMA model file not found: {model_path}')

    threads = num_threads or get_lyrics_threads()
    _apply_thread_env(threads)

    if _llama_model is not None and _llama_model_path == model_path:
        return _llama_model

    # Qwen / llama-cpp is intentionally pinned to CPU to avoid regressions on
    # images that ship the default CPU-only llama-cpp-python wheel.
    logger.info('Loading LLaMA model %s (threads=%s, n_gpu_layers=0)',
                model_path, threads)
    _llama_model = Llama(model_path=model_path, n_threads=threads,
                         n_gpu_layers=0, verbose=False)
    _llama_model_path = model_path
    logger.info('LLaMA model ready')
    return _llama_model


def load_topic_embedding_model(model_name: Optional[str] = None):
    """Load the e5 embedding tokenizer + model from the local container cache."""
    global _embedding_tokenizer, _embedding_model, _embedding_model_name
    if AutoTokenizer is None or AutoModel is None:
        raise RuntimeError('transformers is required for embeddings.')

    if model_name is None:
        try:
            from config import LYRICS_DEFAULT_TOPIC_EMBEDDING_MODEL
            model_name = LYRICS_DEFAULT_TOPIC_EMBEDDING_MODEL
        except Exception:
            model_name = 'intfloat/e5-base-v2'

    # Fast path: already loaded.
    if (_embedding_tokenizer is not None
            and _embedding_model is not None
            and _embedding_model_name == model_name):
        return _embedding_tokenizer, _embedding_model

    with _embedding_load_lock:
        # Re-check after acquiring the lock: another thread may have finished
        # the load while we were waiting.
        if (_embedding_tokenizer is not None
                and _embedding_model is not None
                and _embedding_model_name == model_name):
            return _embedding_tokenizer, _embedding_model

        try:
            from config import LYRICS_DEFAULT_TOPIC_EMBEDDING_CACHE_DIR
            cache_dir = LYRICS_DEFAULT_TOPIC_EMBEDDING_CACHE_DIR
        except Exception:
            cache_dir = '/app/model/e5-base-v2'

        target = cache_dir if os.path.isdir(cache_dir) else model_name
        logger.info('Loading embedding model %s (offline)', target)
        # e5 is bundled inside /app/model/e5-base-v2; force offline so a
        # misconfigured network can never trigger a download for this required model.
        # ``low_cpu_mem_usage=False`` disables ``accelerate``'s meta-device
        # lazy init that previously caused ``Tensor on device cpu is not on
        # the expected device meta!`` errors when the search endpoint was
        # called before the model was fully materialized.
        tokenizer = AutoTokenizer.from_pretrained(str(target), local_files_only=True)
        model = AutoModel.from_pretrained(
            str(target), local_files_only=True, low_cpu_mem_usage=False)
        # Defensive: explicitly move to CPU and eval mode. ``to('cpu')`` is a
        # no-op when weights are already there, but it materializes any
        # parameter still on ``meta`` (older transformers fallback).
        try:
            if torch is not None:
                model = model.to('cpu')
            model.eval()
        except Exception as exc:
            logger.warning('Could not finalize embedding model state: %s', exc)
        _embedding_tokenizer = tokenizer
        _embedding_model = model
        _embedding_model_name = model_name
        logger.info('Embedding model ready')
        return _embedding_tokenizer, _embedding_model


def _get_axis_embeddings():
    global _axis_label_map, _axis_embeddings
    if _axis_label_map is not None and _axis_embeddings is not None:
        return _axis_label_map, _axis_embeddings

    tokenizer, model = load_topic_embedding_model()
    label_map: Dict[str, List[Tuple[str, str]]] = {}
    embeddings: Dict[str, np.ndarray] = {}
    for axis_name, axis_meta in MUSIC_ANALYSIS_AXES.items():
        labels = list(axis_meta.get('labels', {}).items())
        label_map[axis_name] = labels
        vectors = []
        for _, description in labels:
            vec = _embed_text(description, tokenizer, model)
            if vec is not None:
                vectors.append(vec)
        embeddings[axis_name] = (np.stack(vectors)
                                 if vectors else np.zeros((0, 0), dtype=np.float32))
    _axis_label_map = label_map
    _axis_embeddings = embeddings
    return _axis_label_map, _axis_embeddings


def embed_query_text(text: str) -> Optional[np.ndarray]:
    """Embed a free-form user query with the same e5-base-v2 model used at analysis time.

    Returns a normalized float32 vector of shape (LYRICS_EMBEDDING_DIMENSION,)
    suitable for nearest-neighbor search against the lyrics voyager index.
    Returns ``None`` if the model is not yet ready or the embedding pass
    fails for any reason; callers should treat that as "index/cache not
    ready, retry later" rather than as a fatal error.
    """
    if not text or not text.strip():
        return None
    try:
        tokenizer, model = load_topic_embedding_model()
    except Exception as exc:
        logger.warning('Embedding model not ready (%s); returning no query vector',
                       exc)
        return None
    try:
        vec = _embed_text(text.strip(), tokenizer, model)
    except Exception as exc:
        logger.warning('Embedding pass failed for query %r: %s', text, exc)
        return None
    if vec is None:
        return None
    return vec.astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def _load_audio_from_path(path: str, sr: int = DEFAULT_SAMPLE_RATE) -> Tuple[np.ndarray, int]:
    if sf is not None:
        data, sample_rate = sf.read(path, dtype='float32')
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        if sample_rate != sr:
            if librosa is None:
                raise RuntimeError('librosa is required to resample audio.')
            data = librosa.resample(data, orig_sr=sample_rate, target_sr=sr)
            sample_rate = sr
        return data.astype(np.float32), sample_rate
    if librosa is not None:
        data, sample_rate = librosa.load(path, sr=sr, mono=True)
        return data.astype(np.float32), sample_rate
    raise RuntimeError('Install soundfile or librosa to load audio.')


def _clip_audio(audio: np.ndarray, sr: int,
                max_seconds: float = MAX_AUDIO_SECONDS) -> Tuple[np.ndarray, float]:
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    duration = len(audio) / sr if sr else 0.0
    if duration <= max_seconds:
        return audio.astype(np.float32, copy=False), duration
    end_sample = int(round(max_seconds * sr))
    return audio[:end_sample].astype(np.float32, copy=False), max_seconds


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def _normalize_text(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r'\s+([?.!,;:])', r'\1', cleaned)
    cleaned = re.sub(r'\s*\n\s*', '\n', cleaned)
    cleaned = re.sub(r'(^|[.!?]\s+)(i)\b', lambda m: m.group(1) + 'I', cleaned)
    cleaned = re.sub(r'\s{2,}', ' ', cleaned)
    return cleaned


def _split_into_word_chunks(text: str, max_words: int = MAX_WORDS_PER_CHUNK) -> List[str]:
    text = text.strip()
    if not text:
        return []
    words = text.split()
    if len(words) <= max_words:
        return [text]
    return [' '.join(words[i:i + max_words]) for i in range(0, len(words), max_words)]


# ---------------------------------------------------------------------------
# External lyrics APIs (LRCLIB, Vagalume)
# ---------------------------------------------------------------------------

_LRC_METADATA_RE = re.compile(r'^\s*\[(?:ar|ti|al|au|by|la|length|offset|re|ve):[^\]]*\]\s*$', re.IGNORECASE)
# Section markers like (Chorus), [Verse 2], {Bridge}, "Pre-Chorus:", "Outro -", etc.
_SECTION_HEADER_RE = re.compile(
    r'^\s*[\(\[\{]?\s*'
    r'(?:pre[\s-]?chorus|chorus|verse|bridge|intro|outro|hook|refrain|interlude|'
    r'breakdown|drop|coda|prelude|reprise|post[\s-]?chorus|solo|instrumental)'
    r'(?:\s*[\divxlcIVXLC0-9]+)?'
    r'\s*[\)\]\}]?\s*[:\-]?\s*$',
    re.IGNORECASE,
)
_CONTROL_CHAR_RE = re.compile(r'[\x00-\x08\x0b-\x1f\x7f]')
# Emoji, pictographs, symbols, dingbats, arrows, box drawing, regional indicators, etc.
_NON_TEXT_UNICODE_RE = re.compile(
    "["
    "\U0001F300-\U0001FAFF"  # misc symbols & pictographs, emoticons, transport, supplemental
    "\U0001F600-\U0001F64F"  # emoticons (subset of above, kept for clarity)
    "\U0001F680-\U0001F6FF"  # transport & map
    "\U0001F700-\U0001F77F"  # alchemical
    "\U0001F780-\U0001F7FF"  # geometric extended
    "\U0001F800-\U0001F8FF"  # supplemental arrows-C
    "\U0001F900-\U0001F9FF"  # supplemental symbols & pictographs
    "\U0001FA00-\U0001FA6F"  # chess symbols
    "\U0001FA70-\U0001FAFF"  # symbols & pictographs extended-A
    "\U0001E000-\U0001E02F"  # glagolitic supplement
    "\U0001F000-\U0001F02F"  # mahjong
    "\U0001F0A0-\U0001F0FF"  # playing cards
    "\u2600-\u26FF"          # misc symbols (☀ sun, ☕ coffee, etc.)
    "\u2700-\u27BF"          # dingbats
    "\u2300-\u23FF"          # technical (⏰ alarm, ⏳ hourglass)
    "\u2190-\u21FF"          # arrows
    "\u2500-\u257F"          # box drawing
    "\u2580-\u259F"          # block elements
    "\u25A0-\u25FF"          # geometric shapes
    "\U0001F1E6-\U0001F1FF"  # regional indicator (flags)
    "\u200D\uFE0F\uFE0E"     # ZWJ + variation selectors
    "]",
    flags=re.UNICODE,
)


def _sanitize_lyrics_text(text: str, max_words: int = 300) -> str:
    """Defensive cleanup for lyrics text from any source (API or whisper).

    - Strips control characters, BOMs, zero-width chars.
    - Drops emoji, dingbats, geometric/box symbols, regional indicators, ZWJ.
    - Removes obvious HTML/script tags and LRC ID3 metadata lines.
    - Collapses runs of blank lines.
    - Truncates the whole text to ``max_words`` words (default 300).
    The output is plain text only; no HTML/markup or pictographic characters.
    """
    if not text:
        return ''
    text = text.replace('\ufeff', '').replace('\u200b', '').replace('\u200c', '')
    text = _CONTROL_CHAR_RE.sub('', text)
    text = _NON_TEXT_UNICODE_RE.sub('', text)
    # If a provider accidentally returned HTML, strip tags conservatively.
    text = re.sub(r'<\s*(script|style)[^>]*>.*?<\s*/\s*\1\s*>', '', text,
                  flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r'<[^<>]{1,200}>', '', text)
    out_lines: List[str] = []
    blank_run = 0
    for line in text.splitlines():
        line = line.rstrip()
        if _LRC_METADATA_RE.match(line):
            continue
        if _SECTION_HEADER_RE.match(line):
            continue
        if not line.strip():
            blank_run += 1
            if blank_run <= 1:
                out_lines.append('')
            continue
        blank_run = 0
        out_lines.append(line)
    cleaned = '\n'.join(out_lines).strip()
    words = cleaned.split()
    if len(words) > max_words:
        cleaned = ' '.join(words[:max_words])
    return cleaned


# Backwards-compatible alias used by the API helpers.
_sanitize_api_lyrics = _sanitize_lyrics_text


_LRC_TIMESTAMP_RE = re.compile(r'\[\d+:\d+(?:[.,:]\d+)?\]')


def _strip_lrc_timestamps(text: str) -> str:
    """Strip leading ``[mm:ss.xx]`` timestamps from synced LRC lyrics."""
    lines = []
    for line in text.splitlines():
        cleaned = _LRC_TIMESTAMP_RE.sub('', line).strip()
        if cleaned:
            lines.append(cleaned)
    return '\n'.join(lines)


def _resolve_nested_field(obj: dict, field_path: str) -> Optional[str]:
    """Walk a dot-separated field path into a JSON object.

    e.g. field_path='trackInfo.lyrics' resolves obj['trackInfo']['lyrics'].
    Returns the string value, or None if any key is missing.
    """
    parts = field_path.split('.')
    cur = obj
    for part in parts:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return str(cur).strip() if cur is not None and str(cur).strip() else None


def _fetch_from_configured_api(
    slot: int,
    artist: str,
    track: str,
    timeout: float,
) -> Optional[str]:
    """Call a user-configured lyrics API slot (1 or 2).

    Reads URL template + field config from config module at call time so
    changes saved via the setup wizard take effect without restarting.
    Returns plain-text lyrics or None.
    """
    import urllib.parse
    import urllib.request
    try:
        import config as _cfg
        url_template   = getattr(_cfg, f'LYRICS_API_{slot}_URL_TEMPLATE',  '').strip()
        artist_param   = getattr(_cfg, f'LYRICS_API_{slot}_ARTIST_PARAM',  'artist').strip()
        title_param    = getattr(_cfg, f'LYRICS_API_{slot}_TITLE_PARAM',   'title').strip()
        lyrics_field   = getattr(_cfg, f'LYRICS_API_{slot}_LYRICS_FIELD',  'lyrics').strip()
        apikey_param   = getattr(_cfg, f'LYRICS_API_{slot}_APIKEY_PARAM',  '').strip()
        apikey_value   = getattr(_cfg, f'LYRICS_API_{slot}_APIKEY_VALUE',  '').strip()
    except Exception:
        return None

    if not url_template or not artist_param or not title_param or not lyrics_field:
        return None

    # Build query string
    params: dict = {
        artist_param: artist,
        title_param:  track,
    }
    if apikey_param and apikey_value:
        params[apikey_param] = apikey_value

    # Replace {artist_param}/{title_param} placeholders if present in URL
    # or append as query string — support both styles.
    if '{artist}' in url_template or '{title}' in url_template:
        # Template-style URL: user wrote {artist} and {title} directly
        url = url_template.format(
            artist=urllib.parse.quote(artist, safe=''),
            title=urllib.parse.quote(track, safe=''),
        )
        if apikey_param and apikey_value:
            sep = '&' if '?' in url else '?'
            url += sep + urllib.parse.urlencode({apikey_param: apikey_value})
    else:
        # Param-style URL: append all params as query string
        sep = '&' if '?' in url_template else '?'
        url = url_template + sep + urllib.parse.urlencode(params)

    try:
        req = urllib.request.Request(
            url,
            headers={'Accept': 'application/json'},
        )
        import socket
        ctx = None
        try:
            import ssl
            ctx = ssl.create_default_context()
        except Exception:
            pass
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
            raw = resp.read(512 * 1024).decode('utf-8', errors='replace')
    except Exception as exc:
        logger.debug('Lyrics API slot %s HTTP error: %s', slot, exc)
        return None

    try:
        import json as _json
        data = _json.loads(raw)
    except Exception:
        return None

    return _resolve_nested_field(data, lyrics_field)


def fetch_remote_lyrics(artist: Optional[str], track: Optional[str],
                        total_budget: Optional[float] = None) -> Optional[str]:
    """Try user-configured API slots 1 and 2. Return plain-text lyrics or None.

    ``total_budget`` defaults to the sum of both configured per-slot timeouts
    (from config), so the overall deadline automatically reflects what the user
    set in the setup wizard. Falls back to Whisper if no lyrics are found.
    """
    if total_budget is None:
        try:
            import config as _cfg
            total_budget = (
                float(getattr(_cfg, 'LYRICS_API_1_TIMEOUT', 5.0) or 5.0) +
                float(getattr(_cfg, 'LYRICS_API_2_TIMEOUT', 5.0) or 5.0)
            )
        except Exception:
            total_budget = 10.0
    import time
    artist = (artist or '').strip()
    track = (track or '').strip()
    if not artist or not track:
        return None
    deadline = time.monotonic() + total_budget
    for slot in (1, 2):
        remaining = deadline - time.monotonic()
        if remaining <= 0.5:
            logger.info('Lyrics API budget exhausted before slot %s', slot)
            break
        try:
            import config as _cfg
            configured_timeout = float(getattr(_cfg, f'LYRICS_API_{slot}_TIMEOUT', 5.0) or 5.0)
        except Exception:
            configured_timeout = 5.0
        per_slot_timeout = min(configured_timeout, remaining)
        try:
            text = _fetch_from_configured_api(slot, artist, track, per_slot_timeout)
        except Exception as exc:
            logger.warning('Lyrics API slot %s failed for %r/%r: %s', slot, artist, track, exc)
            continue
        if text:
            sanitized = _sanitize_api_lyrics(text)
            if not sanitized:
                logger.warning('Lyrics API slot %s returned content but sanitizer dropped it for %r/%r',
                               slot, artist, track)
                continue
            logger.info('Lyrics API slot %s returned %s words for %r/%r',
                        slot, len(sanitized.split()), artist, track)
            return sanitized
    return None


# ---------------------------------------------------------------------------
# Voice activity detection (silero) — keep only voiced regions before whisper
# ---------------------------------------------------------------------------

_vad_model = None
_VAD_CACHE_DIR = os.path.join(tempfile.gettempdir(), 'audiomuse-vad-cache')


def _get_vad_model():
    global _vad_model
    if _vad_model is None and load_silero_vad is not None:
        # Ensure our private /tmp cache dir exists
        try:
            os.makedirs(_VAD_CACHE_DIR, exist_ok=True)
        except Exception as exc:
            logger.warning('VAD: could not create cache dir %s: %s', _VAD_CACHE_DIR, exc)
        # Disable XetHub CDN (same fix as Marian translator) before any download
        os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
        os.environ.setdefault('HF_XET_DISABLE', '1')
        # Point torch.hub to our writable /tmp dir
        _prev_hub_dir = torch.hub.get_dir() if torch is not None else None
        if torch is not None:
            torch.hub.set_dir(_VAD_CACHE_DIR)
        logger.info('VAD: loading silero from %s', _VAD_CACHE_DIR)
        try:
            _vad_model = load_silero_vad()
        except Exception as exc:
            logger.warning('VAD model load failed (%s); cleaning up %s', exc, _VAD_CACHE_DIR)
            try:
                shutil.rmtree(_VAD_CACHE_DIR, ignore_errors=True)  # shutil imported at module level
                logger.info('VAD: deleted stale cache dir %s', _VAD_CACHE_DIR)
            except Exception as cleanup_exc:
                logger.warning('VAD cache cleanup failed: %s', cleanup_exc)
            logger.warning('VAD disabled for this process — cache cleaned, will retry on next worker start')
        finally:
            # Restore the original torch.hub dir
            if torch is not None and _prev_hub_dir is not None:
                torch.hub.set_dir(_prev_hub_dir)
    return _vad_model


def _apply_vad(audio: np.ndarray, sr: int) -> np.ndarray:
    """Return a concatenation of voiced regions; fall back to ``audio`` on any issue."""
    if sr != 16000 or torch is None or get_speech_timestamps is None:
        return audio
    model = _get_vad_model()
    if model is None:
        return audio
    try:
        tensor = torch.from_numpy(audio.astype(np.float32))
        ts = get_speech_timestamps(tensor, model, sampling_rate=sr)
    except Exception as exc:
        logger.warning('VAD failed: %s; using raw audio', exc)
        return audio
    if not ts:
        return audio
    voiced = np.concatenate([audio[t['start']:t['end']] for t in ts])
    if len(voiced) < sr * 5:  # less than 5s of voice -> trust the original
        return audio
    return voiced


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------

def _transcribe(audio: np.ndarray, sr: int, model,
                language: Optional[str] = None) -> Dict[str, object]:
    if len(audio) == 0:
        return {'text': '', 'language': language, 'duration': 0.0}
    if sf is None:
        raise RuntimeError('soundfile is required to feed Whisper.')
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        tmp_path = tmp.name
    try:
        sf.write(tmp_path, audio, sr, subtype='PCM_16')
        result = model.transcribe(tmp_path, language=language, fp16=False)
        return {
            'text': result.get('text', '').strip(),
            'language': result.get('language', language),
            'duration': len(audio) / sr,
        }
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Language detection + translation
# ---------------------------------------------------------------------------

def _detect_language(text: str) -> Tuple[str, float]:
    if not text or not text.strip() or detect_langs is None:
        return 'en', 0.0
    try:
        candidates = detect_langs(text.replace('\n', ' '))
    except Exception:
        return 'en', 0.0
    if not candidates:
        return 'en', 0.0
    best = candidates[0]
    return best.lang, float(best.prob)


_MARIAN_CACHE_DIR: Optional[str] = None


def _get_marian_cache_dir() -> str:
    """Return a process-private writable directory for Marian downloads.

    We deliberately do NOT use the shared ``HF_HOME`` cache (where the bundled
    e5 / RoBERTa / MuLan / BERT / BART models live read-only) so a stale lock
    file or restrictive perms there cannot block the translator. The path is
    configured via ``LYRICS_MARIAN_CACHE_DIR`` (config.py / env var).
    """
    global _MARIAN_CACHE_DIR
    if _MARIAN_CACHE_DIR is not None:
        return _MARIAN_CACHE_DIR
    try:
        from config import LYRICS_MARIAN_CACHE_DIR
        base = LYRICS_MARIAN_CACHE_DIR
    except Exception:
        base = os.environ.get('LYRICS_MARIAN_CACHE_DIR') \
            or os.path.join(tempfile.gettempdir(), 'audiomuse-marian-cache')
    try:
        os.makedirs(base, exist_ok=True)
    except Exception as exc:
        logger.warning('Could not create Marian cache dir %s: %s', base, exc)
        base = tempfile.mkdtemp(prefix='audiomuse-marian-')
    _MARIAN_CACHE_DIR = base
    # Redirect HF Xet client logs to a writable subdir of the same cache.
    # Default ($HF_HOME/xet/logs == /app/.cache/huggingface/xet/logs) is
    # read-only in our container and triggers "Permission denied" errors.
    xet_log_dir = os.path.join(_MARIAN_CACHE_DIR, 'xet-logs')
    try:
        os.makedirs(xet_log_dir, exist_ok=True)
        os.environ.setdefault('HF_XET_LOG_DIR', xet_log_dir)
    except Exception as exc:
        logger.warning('Could not create xet log dir %s: %s', xet_log_dir, exc)
    # The Hugging Face Hub xet client (Rust) has been observed to fail with
    # `HTTP 416 Range Not Satisfiable` against cas-server.xethub.hf.co when
    # downloading Marian translation models, and to crash with `Permission
    # denied` while writing logs to a read-only $HF_HOME/xet/logs. Disable the
    # xet transport so transformers falls back to plain HTTPS downloads.
    os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
    os.environ.setdefault('HF_XET_DISABLE', '1')
    logger.info('Marian translator cache dir: %s', _MARIAN_CACHE_DIR)
    return _MARIAN_CACHE_DIR


def _get_marian(source_lang: str):
    source_lang = source_lang.lower()
    if source_lang == 'en':
        return None, None
    if AutoModelForSeq2SeqLM is None or AutoTokenizer is None:
        return None, None
    cached = _marian_cache.get(source_lang)
    if cached is not None:
        return cached

    try:
        from config import LYRICS_DEFAULT_MARIAN_PREFIX
    except Exception:
        LYRICS_DEFAULT_MARIAN_PREFIX = 'Helsinki-NLP/opus-mt-{}-en'

    model_name = LYRICS_DEFAULT_MARIAN_PREFIX.format(source_lang)

    # Marian language pairs are downloaded on demand the first time a new source
    # language is seen. They land in a process-private temp dir so they cannot
    # collide with the read-only bundled HF cache. The bundled models (e5,
    # RoBERTa, MuLan, ...) use explicit local_files_only=True elsewhere so they
    # cannot accidentally hit the network.
    cache_dir = _get_marian_cache_dir()
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=cache_dir)
    except Exception as exc:
        logger.warning('Could not load Marian model %s: %s; cleaning up cache dir %s', model_name, exc, cache_dir)
        try:
            shutil.rmtree(cache_dir, ignore_errors=True)
            logger.info('Marian: deleted stale cache dir %s', cache_dir)
        except Exception as cleanup_exc:
            logger.warning('Marian cache cleanup failed: %s', cleanup_exc)
        return None, None
    _marian_cache[source_lang] = (tokenizer, model)
    return tokenizer, model


def _translate_to_english(text: str, source_lang: str) -> str:
    """Translate ``text`` to English. Return ``''`` on any failure.

    We never fall back to the original (non-English) text: downstream we
    embed with an English-tuned model and score English axis descriptions,
    so leaking a foreign-language transcription would poison the vector
    space. An empty string makes the caller treat the track as having no
    usable lyrics, which then triggers the instrumental sentinel fallback.
    """
    if not text or source_lang.lower() == 'en':
        return text
    try:
        tokenizer, model = _get_marian(source_lang)
    except Exception as exc:
        logger.warning('Marian load failed for %s: %s; dropping lyrics',
                       source_lang, exc)
        return ''
    if tokenizer is None or model is None:
        logger.warning('No Marian model available for %s; dropping lyrics',
                       source_lang)
        return ''
    pieces: List[str] = []
    for chunk in _split_into_word_chunks(text):
        try:
            inputs = tokenizer(chunk, truncation=True, padding=True,
                               return_tensors='pt', max_length=512)
            outputs = model.generate(**inputs, max_length=512)
            translated = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            piece = translated[0].strip() if translated else ''
            if not piece:
                logger.warning('Marian returned empty translation for chunk; '
                               'dropping lyrics')
                return ''
            pieces.append(piece)
        except Exception as exc:
            logger.warning('Marian translation chunk failed (%s); dropping lyrics',
                           exc)
            return ''
    return ' '.join(pieces)


# ---------------------------------------------------------------------------
# Cleanup with Qwen
# ---------------------------------------------------------------------------

def _llama_clean_chunk(model, chunk: str, max_tokens: int, temperature: float) -> str:
    prompt = (
        "You are a song transcription cleanup assistant.\n"
        "This text is a Whisper transcription output.\n"
        "Your job is to fix only obvious transcription mistakes and minor formatting issues.\n"
        "Do not invent new lyrics, do not add new content, and do not change the meaning.\n"
        "Preserve the original phrasing and sentence structure unless an obvious error must be fixed.\n"
        "Output only the cleaned lyrics text with no extra labels.\n\n"
        "Raw transcription:\n"
        f"{chunk}\n\n"
        "Cleaned lyrics text:\n"
    )
    response = model.create_completion(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=0.75,
        presence_penalty=0.5,
        repeat_penalty=1.15,
        echo=False,
        stop=["\n\n", "\nOutput only", "\nDo not include any metadata"],
    )
    text_out = ''
    if isinstance(response, dict):
        choices = response.get('choices') or []
        if choices:
            text_out = choices[0].get('text', '')
    return text_out or chunk


def _clean_with_llama(text: str, model, max_tokens: int = 256,
                      temperature: float = 0.2) -> str:
    if not text or not text.strip():
        return ''
    chunks = _split_into_word_chunks(text)
    cleaned: List[str] = []
    for index, chunk in enumerate(chunks, start=1):
        try:
            text_out = _llama_clean_chunk(model, chunk, max_tokens, temperature)
            cleaned.append(_normalize_text(text_out))
            continue
        except Exception as exc:
            logger.warning('LLaMA cleanup chunk %s/%s failed: %s; retrying with smaller chunks',
                           index, len(chunks), exc)
        # Retry: split the failing chunk into 30-word sub-chunks with a tighter token cap.
        for sub in _split_into_word_chunks(chunk, 30):
            try:
                cleaned.append(_normalize_text(_llama_clean_chunk(model, sub, 128, temperature)))
            except Exception as exc2:
                logger.warning('LLaMA cleanup retry failed: %s; using raw sub-chunk', exc2)
                cleaned.append(_normalize_text(sub))
    return '\n\n'.join(cleaned).strip()


# ---------------------------------------------------------------------------
# Embedding + axis scoring
# ---------------------------------------------------------------------------

def _embed_text(text: str, tokenizer, model) -> Optional[np.ndarray]:
    if torch is None:
        raise RuntimeError('torch is required to compute embeddings.')
    if not text or not text.strip():
        return None
    encoded = tokenizer(text, truncation=True, padding='max_length',
                        max_length=128, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**encoded)
    last_hidden = outputs.last_hidden_state
    mask = encoded['attention_mask'].unsqueeze(-1).expand(last_hidden.size()).float()
    summed = (last_hidden * mask).sum(1)
    counts = mask.sum(1).clamp(min=1e-9)
    pooled = (summed / counts).squeeze(0)
    vector = pooled.cpu().numpy()
    norm = float(np.linalg.norm(vector))
    if norm > 0:
        vector = vector / norm
    return vector


def _softmax(values: np.ndarray, temperature: float) -> np.ndarray:
    if values.size == 0:
        return values
    temperature = temperature if temperature > 0 else 1.0
    scaled = values / temperature
    shifted = scaled - np.max(scaled)
    exp = np.exp(shifted)
    total = float(np.sum(exp))
    return exp / total if total > 0 else np.zeros_like(values)


def axis_columns() -> List[Tuple[str, str]]:
    """Canonical fixed order of (axis_name, label) pairs over MUSIC_ANALYSIS_AXES.

    The ``axis_vector`` stored in BYTEA is a float32 array in this exact order.
    """
    columns: List[Tuple[str, str]] = []
    for axis_name, axis_meta in MUSIC_ANALYSIS_AXES.items():
        for label in axis_meta.get('labels', {}).keys():
            columns.append((axis_name, label))
    return columns


def _score_axes(embedding: np.ndarray, temperature: float = 0.1) -> np.ndarray:
    """Score the embedding against every axis label and return a single fixed-order
    float32 vector (concatenated softmax probabilities per axis, in the order
    defined by ``axis_columns()``)."""
    label_map, axis_embeddings = _get_axis_embeddings()
    parts: List[np.ndarray] = []
    for axis_name, labels in label_map.items():
        matrix = axis_embeddings.get(axis_name)
        if matrix is None or matrix.size == 0:
            parts.append(np.zeros(len(labels), dtype=np.float32))
            continue
        sims = matrix.dot(embedding)
        probs = _softmax(sims, temperature).astype(np.float32, copy=False)
        # Pad/truncate to match the labels list length (defensive).
        if probs.shape[0] != len(labels):
            fixed = np.zeros(len(labels), dtype=np.float32)
            fixed[:min(probs.shape[0], len(labels))] = probs[:min(probs.shape[0], len(labels))]
            probs = fixed
        parts.append(probs)
    if not parts:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(parts).astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# Public orchestrator
# ---------------------------------------------------------------------------

def analyze_lyrics(audio: Optional[np.ndarray] = None,
                   sr: Optional[int] = None,
                   source_path: Optional[Union[str, Path]] = None,
                   use_llm_cleanup: bool = True,
                   artist: Optional[str] = None,
                   track: Optional[str] = None) -> Dict[str, object]:
    """Run the full lyrics pipeline.

    Either ``audio`` (mono float32 + ``sr``) or ``source_path`` must be supplied.
    When ``LYRICS_API_ENABLE`` is true and ``artist``+``track`` are provided,
    LRCLIB and lyrics.ovh are queried first; on a hit, STEPS 1, 1b and 2 are
    skipped and the API text is fed straight into translation/cleanup/embedding.
    Returns a dict with ``text``, ``cleaned_text``, ``language``, ``embedding``
    and ``axis_vector`` (float32 numpy array in canonical axis_columns() order).
    Raises if a required model/source is missing.
    """
    threads = get_lyrics_threads()
    _apply_thread_env(threads)

    used_seconds = 0.0
    raw_text = ''
    detected_lang = 'en'

    # ---- STEP 0 (API): try external lyrics services first ----
    try:
        from config import LYRICS_API_ENABLE
    except Exception:
        LYRICS_API_ENABLE = True
    logger.info('STEP 0 start: external lyrics API (enabled=%s, artist=%r, track=%r)',
                LYRICS_API_ENABLE, artist, track)
    if LYRICS_API_ENABLE and artist and track:
        api_text = fetch_remote_lyrics(artist, track)
        if api_text:
            raw_text = api_text
            logger.info('STEP 0 end: API HIT (%s chars / %s words) - skipping STEPS 1, 1b, 2',
                        len(raw_text), len(raw_text.split()))
            logger.info('STEP 0 raw API output: %s', raw_text)
        else:
            logger.info('STEP 0 end: API MISS - falling back to whisper')
    else:
        logger.info('STEP 0 end: API skipped (disabled or missing artist/track)')

    if not raw_text:
        # ---- STEP 1: audio ----
        logger.info('STEP 1 start: prepare audio (max %.1fs)', MAX_AUDIO_SECONDS)
        if audio is None or sr is None:
            if not source_path:
                raise ValueError('analyze_lyrics requires audio+sr, source_path, or artist+track for API lookup')
            if not os.path.exists(str(source_path)):
                raise FileNotFoundError(f'Audio source not found: {source_path}')
            audio, sr = _load_audio_from_path(str(source_path), sr=DEFAULT_SAMPLE_RATE)
        audio_clip, used_seconds = _clip_audio(audio, sr)
        logger.info('STEP 1 end: audio ready, used=%.2fs samples=%s sr=%s',
                    used_seconds, len(audio_clip), sr)

        # ---- STEP 1b: VAD pre-filter (keep only voiced regions) ----
        pre_vad_samples = len(audio_clip)
        audio_clip = _apply_vad(audio_clip, sr)
        if len(audio_clip) != pre_vad_samples:
            logger.info('VAD: %.2fs -> %.2fs voiced',
                        pre_vad_samples / sr, len(audio_clip) / sr)

        # ---- STEP 2: whisper transcription ----
        _WHISPER_TIMEOUT_S = 300  # 5 minutes
        logger.info('STEP 2 start: whisper transcription (threads=%s, timeout=%ss)', threads, _WHISPER_TIMEOUT_S)
        whisper_model = load_whisper_model(num_threads=threads)

        class _WhisperTimeout(Exception):
            pass

        def _alarm_handler(signum, frame):
            raise _WhisperTimeout()

        _old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
        signal.alarm(_WHISPER_TIMEOUT_S)
        try:
            transcription = _transcribe(audio_clip, sr, whisper_model)
        except _WhisperTimeout:
            logger.warning(
                'STEP 2 timeout: Whisper exceeded %ss — returning empty transcript',
                _WHISPER_TIMEOUT_S,
            )
            transcription = {'text': '', 'language': 'en', 'duration': len(audio_clip) / sr}
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, _old_handler)

        raw_text = _sanitize_lyrics_text((transcription.get('text') or '').strip())
        detected_lang = transcription.get('language') or 'en'
        logger.info('STEP 2 end: transcript length=%s chars / %s words',
                    len(raw_text), len(raw_text.split()))
        logger.info('STEP 2 raw whisper output: %s', raw_text or '<empty>')
        if len(raw_text.split()) < MIN_WORDS_FOR_CLEANUP:
            logger.info('STEP 2 word count below %s — skipping STEPS 3-5', MIN_WORDS_FOR_CLEANUP)
            raw_text = ''

    # ---- STEP 3: language detection ----
    logger.info('STEP 3 start: language detection')
    if raw_text:
        guess_lang, confidence = _detect_language(raw_text)
        if confidence >= 0.7:
            detected_lang = guess_lang
    else:
        detected_lang = 'en'
    logger.info('STEP 3 end: language=%s', detected_lang)

    # ---- STEP 4: translation ----
    logger.info('STEP 4 start: translation (source=%s)', detected_lang)
    if raw_text and detected_lang != 'en':
        try:
            text_for_cleanup = _translate_to_english(raw_text, detected_lang)
        except Exception as exc:
            # Translation must never crash the worker. If the Marian model
            # download or generation blows up (HF Hub 416/xet failures, OOM,
            # ...), drop the text so we don't embed a foreign-language
            # transcription with the English embedding model.
            logger.warning('Translation failed (%s); dropping lyrics', exc)
            text_for_cleanup = ''
    else:
        text_for_cleanup = raw_text
    logger.info('STEP 4 end: translated length=%s words', len(text_for_cleanup.split()))

    # ---- STEP 5: cleanup ----
    cleaned_text = ''
    word_count = len(text_for_cleanup.split())
    try:
        from config import LYRICS_LLM_ENABLED
    except Exception:
        LYRICS_LLM_ENABLED = True
    do_cleanup = (use_llm_cleanup and LYRICS_LLM_ENABLED
                  and word_count >= MIN_WORDS_FOR_CLEANUP)
    logger.info('STEP 5 start: qwen cleanup (enabled=%s, words=%s)',
                do_cleanup, word_count)
    if do_cleanup:
        try:
            llama = load_llama_model(num_threads=threads)
            cleaned_text = _clean_with_llama(text_for_cleanup, llama)
        except Exception as exc:
            logger.warning('LLaMA cleanup skipped: %s', exc)
            cleaned_text = ''
    final_text = cleaned_text or text_for_cleanup
    logger.info('STEP 5 end: final text length=%s words', len(final_text.split()))

    # ---- STEP 6: embedding + axes ----
    logger.info('STEP 6 start: embedding + axis scoring')
    embedding = None
    axis_vector: np.ndarray = np.zeros(0, dtype=np.float32)
    if len(final_text.split()) >= MIN_WORDS_FOR_EMBEDDING:
        tokenizer, model = load_topic_embedding_model()
        embedding = _embed_text(final_text, tokenizer, model)
        if embedding is not None:
            axis_vector = _score_axes(embedding)
    else:
        # Below threshold: treat the track as having no usable lyrics. The text
        # fields are blanked so callers never persist or display partial garbage.
        raw_text = ''
        text_for_cleanup = ''
        cleaned_text = ''
        final_text = ''

    # ---- STEP 6b: instrumental fallback ----
    # If no usable embedding was produced (no lyrics in the audio AND no API
    # hit, or fewer than MIN_WORDS_FOR_EMBEDDING words after cleanup), fall
    # back to deterministic sentinel vectors. This lets us:
    #   * persist a row so future analysis runs skip the track,
    #   * cluster all instrumental tracks together in vector search,
    #   * keep them safely far from real lyrical embeddings (the axis sentinel
    #     is uniformly negative, which a softmax axis_vector can never be).
    if embedding is None or getattr(embedding, 'size', 0) == 0:
        try:
            from config import (
                LYRICS_INSTRUMENTAL_EMBEDDING,
                LYRICS_INSTRUMENTAL_AXIS_FILL,
            )
            embedding = np.array(LYRICS_INSTRUMENTAL_EMBEDDING, dtype=np.float32, copy=True)
            axis_dim = len(axis_columns())
            axis_vector = np.full(axis_dim, LYRICS_INSTRUMENTAL_AXIS_FILL, dtype=np.float32)
            logger.info('STEP 6b: applied instrumental sentinel '
                        '(embedding_dim=%s, axis_dim=%s)',
                        embedding.shape[0], axis_vector.shape[0])
        except Exception as exc:
            logger.warning('Could not apply instrumental sentinel: %s', exc)

    logger.info('STEP 6 end: embedding=%s axis_vector_dim=%s',
                None if embedding is None else embedding.shape,
                int(axis_vector.shape[0]) if axis_vector is not None else 0)

    return {
        'text': raw_text,
        'translated_text': text_for_cleanup,
        'cleaned_text': cleaned_text,
        'final_text': final_text,
        'language': detected_lang,
        'used_seconds': used_seconds,
        'embedding': embedding,
        'axis_vector': axis_vector,
    }
