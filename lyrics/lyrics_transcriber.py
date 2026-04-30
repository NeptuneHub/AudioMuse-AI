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
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

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
    _embedding_tokenizer = AutoTokenizer.from_pretrained(str(target), local_files_only=True)
    _embedding_model = AutoModel.from_pretrained(str(target), local_files_only=True)
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
    """
    if not text or not text.strip():
        return None
    tokenizer, model = load_topic_embedding_model()
    vec = _embed_text(text.strip(), tokenizer, model)
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


def _fetch_from_lrclib(artist: str, track: str, timeout: float) -> Optional[str]:
    import requests
    r = requests.get(
        'https://lrclib.net/api/get',
        params={'artist_name': artist, 'track_name': track},
        timeout=timeout,
        headers={'User-Agent': 'AudioMuse-AI/1.0'},
    )
    if not r.ok:
        return None
    data = r.json() or {}
    plain = (data.get('plainLyrics') or '').strip()
    if plain:
        return plain
    synced = (data.get('syncedLyrics') or '').strip()
    if synced:
        return _strip_lrc_timestamps(synced) or None
    return None


def _fetch_from_lyrics_ovh(artist: str, track: str, timeout: float) -> Optional[str]:
    import requests
    from urllib.parse import quote
    url = f'https://api.lyrics.ovh/v1/{quote(artist)}/{quote(track)}'
    r = requests.get(url, timeout=timeout, headers={'User-Agent': 'AudioMuse-AI/1.0'})
    if not r.ok:
        return None
    data = r.json() or {}
    text = (data.get('lyrics') or '').strip()
    return text or None


def fetch_remote_lyrics(artist: Optional[str], track: Optional[str],
                        total_budget: float = 10.0) -> Optional[str]:
    """Try LRCLIB then lyrics.ovh. Return plain-text lyrics or ``None``.

    The combined wait across both providers is capped at ``total_budget`` seconds
    (default 10s). If we can't get lyrics in that window, fall back to whisper.
    """
    import time
    artist = (artist or '').strip()
    track = (track or '').strip()
    if not artist or not track:
        return None
    deadline = time.monotonic() + total_budget
    providers = (('LRCLIB', _fetch_from_lrclib),
                 ('lyrics.ovh', _fetch_from_lyrics_ovh))
    for provider_name, provider in providers:
        remaining = deadline - time.monotonic()
        if remaining <= 0.5:
            logger.info('Lyrics API budget exhausted before %s', provider_name)
            break
        per_provider_timeout = min(5.0, remaining)
        try:
            text = provider(artist, track, per_provider_timeout)
        except Exception as exc:
            logger.warning('%s lookup failed for %r/%r: %s',
                           provider_name, artist, track, exc)
            continue
        if text:
            sanitized = _sanitize_api_lyrics(text)
            if not sanitized:
                logger.warning('%s returned content but sanitizer dropped everything for %r/%r',
                               provider_name, artist, track)
                continue
            logger.info('%s returned lyrics for %r/%r (%s words)',
                        provider_name, artist, track, len(sanitized.split()))
            return sanitized
    return None


# ---------------------------------------------------------------------------
# Voice activity detection (silero) — keep only voiced regions before whisper
# ---------------------------------------------------------------------------

_vad_model = None


def _get_vad_model():
    global _vad_model
    if _vad_model is None and load_silero_vad is not None:
        _vad_model = load_silero_vad()
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
        logger.warning('Could not load Marian model %s: %s', model_name, exc)
        return None, None
    _marian_cache[source_lang] = (tokenizer, model)
    return tokenizer, model


def _translate_to_english(text: str, source_lang: str) -> str:
    if not text or source_lang.lower() == 'en':
        return text
    tokenizer, model = _get_marian(source_lang)
    if tokenizer is None or model is None:
        return text
    pieces: List[str] = []
    for chunk in _split_into_word_chunks(text):
        inputs = tokenizer(chunk, truncation=True, padding=True,
                           return_tensors='pt', max_length=512)
        outputs = model.generate(**inputs, max_length=512)
        translated = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        pieces.append(translated[0].strip() if translated else chunk)
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
        logger.info('STEP 2 start: whisper transcription (threads=%s)', threads)
        whisper_model = load_whisper_model(num_threads=threads)
        transcription = _transcribe(audio_clip, sr, whisper_model)
        raw_text = _sanitize_lyrics_text((transcription.get('text') or '').strip())
        detected_lang = transcription.get('language') or 'en'
        logger.info('STEP 2 end: transcript length=%s chars / %s words',
                    len(raw_text), len(raw_text.split()))
        logger.info('STEP 2 raw whisper output: %s', raw_text or '<empty>')

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
        text_for_cleanup = _translate_to_english(raw_text, detected_lang)
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
