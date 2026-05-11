"""Small lyrics analysis orchestrator.

Models are assumed to be already present inside the container; this module never
downloads anything. It exposes a single high level entry point ``analyze_lyrics``
plus the cached model loaders used by the worker bootstrap.

Pipeline (each step emits a ``STEP X start`` and ``STEP X end`` log line):

    STEP 1  load / clip audio (max 4 minutes)
    STEP 2  Qwen3-ASR transcription (ONNX CPU)
    STEP 3  language detection
    STEP 4  optional translation to English (MarianMT)
    STEP 5  e5 embedding + axis scoring
"""

from __future__ import annotations

import logging
import os
import re
import signal
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
    import torch
except Exception:  # pragma: no cover - CUDA-build torch can raise OSError on dlopen
    torch = None

try:
    # Torch-free silero VAD via raw onnxruntime; replaces the silero-vad
    # PyPI package (which transitively pulls torch + torchaudio).
    from .silero_onnx import get_speech_timestamps
except Exception:  # pragma: no cover - onnxruntime missing or model not yet downloaded
    get_speech_timestamps = None

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_SAMPLE_RATE = 16000
# Max audio length fed to the ASR. 4 minutes covers most songs end-to-end;
# longer tracks get clipped here so a single outlier doesn't dominate
# decode time. Tunable via env when needed (e.g. live recordings).
MAX_AUDIO_SECONDS = float(os.environ.get('LYRICS_MAX_AUDIO_SECONDS', '240'))

# Minimum CHARACTERS (not words) a transcript must have for the embedding
# step to run. Char-based so CJK / Thai / Lao lyrics — which have no
# whitespace between words — aren't all collapsed to "1 word" by
# str.split() and silently dropped as instrumental. Resolved through
# config.py so deployments can tune it via the LYRICS_MIN_CHARS_FOR_EMBEDDING
# env var.
try:
    from config import LYRICS_MIN_CHARS_FOR_EMBEDDING as _CFG_MIN_CHARS
    MIN_CHARS_FOR_EMBEDDING = int(_CFG_MIN_CHARS)
except Exception:
    MIN_CHARS_FOR_EMBEDDING = int(os.environ.get('LYRICS_MIN_CHARS_FOR_EMBEDDING', '250'))

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
    """Number of CPU threads for Qwen3-ASR / MarianMT inside this process.

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

_embedding_tokenizer = None
_embedding_model = None
_embedding_model_name: Optional[str] = None
_axis_label_map: Optional[Dict] = None
_axis_embeddings: Optional[Dict] = None
# Serialize model loads so concurrent callers (e.g. axis warm-up + a user
# search request hitting the API at the same time) cannot observe a
# half-initialized BERT model whose parameters are still on the ``meta``
# device. ``transformers`` with ``accelerate`` installed initializes weights
# lazily on ``meta`` and only materializes them at the end of
# ``from_pretrained``; without a lock another thread can grab the partially
# constructed module from the global cache.
_embedding_load_lock = threading.Lock()


def load_asr_model(num_threads: Optional[int] = None):
    """Load (or return cached) Qwen3-ASR ONNX pipeline (CPU, thread-capped)."""
    threads = num_threads or get_lyrics_threads()
    _apply_thread_env(threads)
    from .qwen_asr import load_asr_model as _load
    # qwen_asr internally divides this by 3 to size each ONNX session, since
    # four sessions share the worker's CPU pool.
    return _load(num_threads=threads)


def load_topic_embedding_model(model_name: Optional[str] = None):
    """Load the e5 embedding tokenizer + ONNX session.

    Returns ``(tokenizer, ort_session)`` — same return shape as the previous
    PyTorch-backed loader, so callers that destructure
    ``tokenizer, model = load_topic_embedding_model()`` keep working. The
    second element is now an ``onnxruntime.InferenceSession`` consumed by
    ``_embed_text`` below.

    The actual loading is delegated to :mod:`lyrics.e5_onnx`, which holds
    its own thread-safe singleton; the local globals here are kept only as
    defensive re-caches for callers that inspect the module state directly.
    """
    global _embedding_tokenizer, _embedding_model, _embedding_model_name
    from .e5_onnx import load_e5_model
    tokenizer, session = load_e5_model()
    _embedding_tokenizer = tokenizer
    _embedding_model = session
    _embedding_model_name = model_name or 'intfloat/e5-base-v2'
    return tokenizer, session


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

# ---------------------------------------------------------------------------
# External lyrics APIs (configured via LYRICS_API_*_URL_TEMPLATE env vars)
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
    """Defensive cleanup for lyrics text from any source (API or ASR).

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
        url_template   = str(getattr(_cfg, f'LYRICS_API_{slot}_URL_TEMPLATE',  '') or '').strip()
        artist_param   = str(getattr(_cfg, f'LYRICS_API_{slot}_ARTIST_PARAM',  'artist') or 'artist').strip()
        title_param    = str(getattr(_cfg, f'LYRICS_API_{slot}_TITLE_PARAM',   'title') or 'title').strip()
        lyrics_field   = str(getattr(_cfg, f'LYRICS_API_{slot}_LYRICS_FIELD',  'lyrics') or 'lyrics').strip()
        apikey_param   = str(getattr(_cfg, f'LYRICS_API_{slot}_APIKEY_PARAM',  '') or '').strip()
        apikey_value   = str(getattr(_cfg, f'LYRICS_API_{slot}_APIKEY_VALUE',  '') or '').strip()
    except Exception:
        return None

    if not url_template or not artist_param or not title_param or not lyrics_field:
        return None

    # SSRF guard: reject private/loopback/link-local destinations.
    # Validated on the template (hostname doesn't change after artist/title substitution).
    try:
        import ipaddress as _ipaddress
        import socket as _socket
        import urllib.parse as _up
        _parsed_tpl = _up.urlparse(url_template)
        _host = _parsed_tpl.hostname or ''
        _host_l = _host.strip().lower()
        if _host_l in ('localhost', '') or _host_l.endswith('.localhost') or _host_l.endswith('.local'):
            logger.warning('Lyrics API slot %s blocked: local hostname %r', slot, _host)
            return None
        _port = _parsed_tpl.port or (443 if _parsed_tpl.scheme == 'https' else 80)
        for _entry in _socket.getaddrinfo(_host, _port, type=_socket.SOCK_STREAM):
            _ip = _ipaddress.ip_address(_entry[4][0])
            if (_ip.is_private or _ip.is_loopback or _ip.is_link_local
                    or _ip.is_multicast or _ip.is_reserved or _ip.is_unspecified):
                logger.warning('Lyrics API slot %s blocked: %r resolves to non-public IP %s', slot, _host, _ip)
                return None
    except Exception as _ssrf_exc:
        logger.warning('Lyrics API slot %s SSRF check failed: %s', slot, _ssrf_exc)
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
            raw = resp.read(512 * 1024).decode(resp.info().get_content_charset('utf-8'), errors='replace')
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
    set in the setup wizard. Falls back to Qwen3-ASR if no lyrics are found.
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
            logger.info('Lyrics API slot %s returned %s chars for %r/%r',
                        slot, len(sanitized), artist, track)
            return sanitized
    return None


# ---------------------------------------------------------------------------
# Voice activity detection (silero, ONNX) — keep only voiced regions before ASR
# ---------------------------------------------------------------------------


def _apply_vad(audio: np.ndarray, sr: int) -> np.ndarray:
    """Return a concatenation of voiced regions; fall back to ``audio`` on any issue.

    Uses the torch-free silero ONNX shim in ``lyrics.silero_onnx``. The
    ONNX session is loaded lazily on first call and cached in that module.
    """
    if sr != 16000 or get_speech_timestamps is None:
        return audio
    try:
        # silero_onnx accepts the raw numpy array directly — no torch tensor wrap.
        ts = get_speech_timestamps(audio, sample_rate=sr, threshold=0.3)
    except Exception as exc:
        logger.warning('VAD failed: %s; using raw audio', exc)
        return audio
    if not ts:
        logger.info('VAD: no timestamps detected (Silero whiffed) — falling back to full audio')
        return audio
    from config import VAD_VOICE_RECOGNITION
    voiced = np.concatenate([audio[t['start']:t['end']] for t in ts])
    voiced_seconds = len(voiced) / sr
    if len(voiced) < sr * VAD_VOICE_RECOGNITION:
        logger.info('VAD: only %.2fs voiced (<%ss threshold) — treating as instrumental',
                    voiced_seconds, VAD_VOICE_RECOGNITION)
        return np.zeros(0, dtype=audio.dtype)
    logger.info('VAD: %.2fs voiced — keeping voiced segments', voiced_seconds)
    return voiced


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------

def _transcribe(audio: np.ndarray, sr: int,
                language: Optional[str] = None,
                num_threads: Optional[int] = None) -> Dict[str, object]:
    """Transcribe with the Qwen3-ASR ONNX pipeline.

    Returns ``{'text': str, 'language': str, 'duration': float}``.
    """
    if audio is None or len(audio) == 0:
        return {'text': '', 'language': language or '', 'duration': 0.0}
    from .qwen_asr import transcribe as _qwen_transcribe
    return _qwen_transcribe(audio, sr, language=language, num_threads=num_threads)


# ---------------------------------------------------------------------------
# Language detection + translation
# ---------------------------------------------------------------------------

def _translate_to_english(text: str, source_lang: str) -> str:
    """Translate ``text`` to English using the multilingual ONNX translator.

    Returns ``''`` on any failure. We never fall back to the original
    (non-English) text: downstream we embed with an English-tuned model
    and score English axis descriptions, so leaking a foreign-language
    transcription would poison the vector space. An empty string makes
    the caller treat the track as having no usable lyrics, which then
    triggers the instrumental sentinel fallback.

    Backed by a single ``Helsinki-NLP/opus-mt-mul-en`` ONNX bundle (loaded
    once per worker via :mod:`lyrics.translation_onnx`) instead of a
    per-language ``opus-mt-{src}-en`` model, so worker memory no longer
    grows linearly with the number of source languages encountered.
    """
    if not text or source_lang.lower() == 'en':
        return text
    from .translation_onnx import translate_to_english as _onnx_translate
    return _onnx_translate(text, source_lang=source_lang)


# ---------------------------------------------------------------------------
# Embedding + axis scoring
# ---------------------------------------------------------------------------

def _embed_text(text: str, tokenizer, model) -> Optional[np.ndarray]:
    """Embed ``text`` with the e5 ONNX session.

    Kept as a thin wrapper so call sites that still pass ``(tokenizer,
    model)`` from :func:`load_topic_embedding_model` keep working. ``model``
    is now an ``onnxruntime.InferenceSession``; the heavy lifting lives in
    :func:`lyrics.e5_onnx.embed_text`.
    """
    from .e5_onnx import embed_text as _onnx_embed
    return _onnx_embed(text, tokenizer=tokenizer, session=model)


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


def _make_instrumental_sentinel() -> Tuple[np.ndarray, np.ndarray]:
    """Build the deterministic instrumental ``(embedding, axis_vector)`` pair.

    Both vectors are unit-length so cosine similarity is well-defined,
    and both occupy regions of their respective spaces that real songs
    cannot reach (e5 sentinel sits on a basis axis the model almost
    never uses; axis sentinel is uniformly negative, which a softmax-
    derived axis_vector can never produce). See ``LYRICS_INSTRUMENTAL_*``
    in config.py for the full rationale.
    """
    from config import (
        LYRICS_INSTRUMENTAL_EMBEDDING,
        LYRICS_INSTRUMENTAL_AXIS_FILL,
    )
    embedding = np.array(LYRICS_INSTRUMENTAL_EMBEDDING, dtype=np.float32, copy=True)
    axis_dim = len(axis_columns())
    axis_vector = np.full(axis_dim, LYRICS_INSTRUMENTAL_AXIS_FILL, dtype=np.float32)
    return embedding, axis_vector


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
                   artist: Optional[str] = None,
                   track: Optional[str] = None,
                   track_id: Optional[str] = None,
                   top_moods: Optional[Dict[str, float]] = None) -> Dict[str, object]:
    """Run the full lyrics pipeline.

    Either ``audio`` (mono float32 + ``sr``) or ``source_path`` must be supplied.
    Pipeline order:
      STEP -1  media server embedded lyrics (Jellyfin/Emby/Navidrome/Lyrion, MUSICSERVER_LYRICS_TIMEOUT seconds)
      STEP  0  user-configured external APIs (slot 1 then slot 2)
      STEP  0b musicnn instrumental short-circuit (only if -1 and 0 missed)
      STEP  1  load / clip audio
      STEP  1b VAD pre-filter
      STEP  2  Qwen3-ASR transcription (ONNX CPU)
      STEP  3  language detection
      STEP  4  MarianMT translation to English
      STEP  5  e5 embedding + axis scoring

    ``top_moods`` is the MusicNN top-N moods dict (label → score) computed
    earlier in the audio pipeline. When provided and ``'instrumental'`` is
    one of the labels (case-insensitive), STEP 0b short-circuits the rest
    of the pipeline AND no upstream lyrics source (-1 or 0) returned text.
    Upstream lyrics always win over the audio classifier's instrumental
    tag because external-API / media-server lyrics are human-curated
    and authoritative — a heavily-instrumented song can be misclassified
    as "instrumental" by musicnn even when it has vocals.

    Returns a dict with ``text``, ``language``, ``embedding`` and
    ``axis_vector`` (float32 numpy array in canonical axis_columns()
    order). Raises if a required model/source is missing.
    """
    threads = get_lyrics_threads()
    _apply_thread_env(threads)

    used_seconds = 0.0
    raw_text = ''
    detected_lang = 'en'
    # Qwen-only signals. These are overwritten inside the STEP 2 block
    # when ASR actually runs. For STEP -1 / STEP 0 paths (media server,
    # external lyrics API) the text is upstream-trusted and the Qwen
    # confidence gates in STEP 3 must be no-ops — these defaults make
    # that happen without a separate origin flag:
    #   - asr_lang='en' → not in null-langs (skips Gate 3 drop) and
    #     marked English (skips Gate 2 non-English strict floor)
    #   - asr_avg_logprob=0.0 → above any confidence floor (skips Gate 1)
    asr_lang = 'en'
    asr_avg_logprob = 0.0

    # ---- STEP -1: media server embedded lyrics ----
    logger.info('STEP -1 start: media server lyrics (track_id=%r)', track_id)
    if track_id:
        try:
            import config as _cfg
            from tasks.mediaserver import get_lyrics as _ms_get_lyrics
            _ms_timeout = float(getattr(_cfg, 'MUSICSERVER_LYRICS_TIMEOUT', 2.5))
            ms_text = _ms_get_lyrics(track_id, timeout=_ms_timeout) if _ms_timeout > 0 else None
            if ms_text:
                sanitized = _sanitize_api_lyrics(ms_text)
                if sanitized:
                    raw_text = sanitized
                    logger.info('STEP -1 end: media server HIT (%s chars) - skipping STEPS 0, 1, 1b, 2',
                                len(raw_text))
                else:
                    logger.info('STEP -1 end: media server returned content but sanitizer dropped it')
            else:
                logger.info('STEP -1 end: media server MISS')
        except Exception as exc:
            logger.warning('STEP -1 failed: %s', exc)
    else:
        logger.info('STEP -1 end: skipped (no track_id)')

    # ---- STEP 0 (API): try external lyrics services first ----
    try:
        from config import LYRICS_API_ENABLE
    except Exception:
        LYRICS_API_ENABLE = True
    if not raw_text:
        logger.info('STEP 0 start: external lyrics API (enabled=%s, artist=%r, track=%r)',
                    LYRICS_API_ENABLE, artist, track)
        if LYRICS_API_ENABLE and artist and track:
            api_text = fetch_remote_lyrics(artist, track)
            if api_text:
                raw_text = api_text
                logger.info('STEP 0 end: API HIT (%s chars) - skipping STEPS 1, 1b, 2',
                            len(raw_text))
                logger.info('STEP 0 raw API output: %s', raw_text)
            else:
                logger.info('STEP 0 end: API MISS - falling back to Qwen3-ASR')
        else:
            logger.info('STEP 0 end: API skipped (disabled or missing artist/track)')
    else:
        logger.info('STEP 0 skipped: already have lyrics from media server')

    # ---- STEP 0b: musicnn instrumental short-circuit ----
    # When NO upstream lyrics source (media server, external API) found
    # text AND the audio classifier tagged this track as instrumental in
    # its top-N moods, skip every remaining model (Qwen3-ASR, MarianMT,
    # e5) and apply the sentinel directly. We deliberately run this
    # *after* STEP -1 / STEP 0 because the configured upstream sources
    # provide human-curated synced lyrics — if any of them returned text,
    # trust the upstream over musicnn's audio-based classification (a
    # song with heavy instrumentation can get tagged "instrumental"
    # even though it has vocals). The DB row gets the same vectors
    # STEP 5b would produce, so future analysis runs see the track as
    # "already analyzed" and skip it.
    if not raw_text and top_moods:
        normalized_moods = {str(k).strip().lower() for k in top_moods.keys() if k}
        if 'instrumental' in normalized_moods:
            embedding, axis_vector = _make_instrumental_sentinel()
            logger.info(
                "STEP 0b: musicnn flagged track as instrumental "
                "(top_moods=%r) and no upstream lyrics source returned "
                "text — skipping STEPS 1 through 5, applying sentinel "
                "directly (embedding_dim=%s, axis_dim=%s)",
                list(top_moods.keys()), embedding.shape[0], axis_vector.shape[0],
            )
            return {
                'text': '',
                'translated_text': '',
                'final_text': '',
                'language': '',
                'used_seconds': 0.0,
                'embedding': embedding,
                'axis_vector': axis_vector,
            }

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

        # ---- STEP 2: Qwen3-ASR transcription (ONNX CPU, thread-capped) ----
        _ASR_TIMEOUT_S = 300  # 5 minutes
        logger.info('STEP 2 start: Qwen3-ASR transcription (threads=%s, timeout=%ss)',
                    threads, _ASR_TIMEOUT_S)

        class _AsrTimeout(Exception):
            pass

        def _alarm_handler(signum, frame):
            raise _AsrTimeout()

        _old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
        signal.alarm(_ASR_TIMEOUT_S)
        try:
            transcription = _transcribe(audio_clip, sr, num_threads=threads)
        except _AsrTimeout:
            logger.warning(
                'STEP 2 timeout: Qwen3-ASR exceeded %ss — returning empty transcript',
                _ASR_TIMEOUT_S,
            )
            transcription = {'text': '', 'language': '', 'duration': len(audio_clip) / sr}
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, _old_handler)

        raw_text = _sanitize_lyrics_text((transcription.get('text') or '').strip())
        # ASR's audio-based language detection plus its own confidence
        # (avg_logprob; close to 0 = confident, very negative = uncertain).
        asr_lang = (transcription.get('language') or '').strip().lower()
        asr_avg_logprob = float(transcription.get('avg_logprob', float('-inf')))
        detected_lang = asr_lang or 'en'
        logger.info('STEP 2 end: transcript length=%s chars / '
                    'asr_lang=%r / avg_logprob=%.2f',
                    len(raw_text), asr_lang, asr_avg_logprob)
        logger.info('STEP 2 raw ASR output: %s', raw_text or '<empty>')
        if len(raw_text) < MIN_CHARS_FOR_EMBEDDING:
            logger.info('STEP 2 char count below %s (%s chars) — skipping STEPS 3-5',
                        MIN_CHARS_FOR_EMBEDDING, len(raw_text))
            raw_text = ''

    # ---- STEP 3: language decision + confidence gate ----
    # Trust Qwen3-ASR's audio-based detection. The language head is also
    # our proxy for transcription quality: when Qwen can confidently name
    # a language, the transcript itself is likely real; when it bails to
    # a null sentinel, the transcript is suspect and we drop. Drop the
    # transcription when Qwen's own avg_logprob is below
    # ASR_MIN_AVG_LOGPROB (default -1.0) — same signal Whisper uses
    # internally to flag hallucinated/uncertain output.
    ASR_MIN_AVG_LOGPROB = float(os.environ.get('LYRICS_ASR_MIN_AVG_LOGPROB', '-1.0'))
    # Stricter confidence floor when Qwen claims the audio is in a
    # specific non-English language: that's the failure mode where the
    # model confidently emits Chinese (or Spanish, or whatever) for a
    # song that's actually in Irish + Arabic. avg_logprob ≈ -0.7 looks
    # ``OK'' against the absolute floor above, but it's well below what
    # genuine confident output looks like (-0.3 to -0.5). The cost of
    # being wrong on non-English is doubled — we then spend translator
    # compute on the hallucination and persist garbage as the song's
    # "lyrics" — so we hold non-English to a higher bar.
    ASR_NON_ENGLISH_MIN_LOGPROB = float(os.environ.get(
        'LYRICS_ASR_NON_ENGLISH_MIN_LOGPROB', '-0.5'))
    # Sentinel languages Qwen emits when it can't confidently identify a
    # language. By themselves they don't mean "no speech" — they mean
    # "no usable language label". Combined with a confidence gate, we
    # only drop when BOTH signals point at garbage.
    _ASR_NULL_LANGS = {'', 'none', 'nolang', 'unknown', 'nospeech', 'noisy'}
    # Qwen3-ASR sometimes emits the spelled-out English name instead of the
    # ISO 639-1 code (e.g. ``language english<asr_text>...``). Treat anything
    # in here as English so STEP 4 skips the translator.
    _ASR_ENGLISH_LANGS = {'en', 'eng', 'english'}
    if raw_text and asr_avg_logprob < ASR_MIN_AVG_LOGPROB:
        logger.info('STEP 3: ASR avg_logprob %.2f < %.2f — dropping likely '
                    'hallucinated transcription, treating as instrumental',
                    asr_avg_logprob, ASR_MIN_AVG_LOGPROB)
        raw_text = ''
    if (raw_text and asr_lang
            and asr_lang not in _ASR_NULL_LANGS
            and asr_lang not in _ASR_ENGLISH_LANGS
            and asr_avg_logprob < ASR_NON_ENGLISH_MIN_LOGPROB):
        logger.info('STEP 3: ASR reported non-English language %r with '
                    'avg_logprob %.2f < %.2f — dropping likely hallucinated '
                    'transcription (translator would only amplify the '
                    'garbage), treating as instrumental',
                    asr_lang, asr_avg_logprob, ASR_NON_ENGLISH_MIN_LOGPROB)
        raw_text = ''
    if raw_text and asr_lang in _ASR_NULL_LANGS:
        # Qwen couldn't identify the language. We treat the language
        # head as a proxy for transcription quality: if Qwen can't
        # decide between ~95 languages it was trained on, whatever
        # text it produced is suspect at best. Drop and treat as
        # instrumental rather than guessing with a text-based
        # detector that's known to misclassify lyric snippets.
        logger.info('STEP 3: ASR reported no usable language (%r) — '
                    'treating as instrumental (Qwen language uncertainty '
                    'is a proxy for transcript uncertainty)',
                    asr_lang)
        raw_text = ''
    if raw_text:
        if asr_lang in _ASR_ENGLISH_LANGS:
            # Qwen sometimes returns the full word "english" instead of "en";
            # normalize so STEP 4 sees `detected_lang == 'en'` and skips the
            # translator entirely.
            detected_lang = 'en'
            logger.info('STEP 3: ASR-reported language %r → normalized to en '
                        '(translation skipped)', asr_lang)
        else:
            # Non-null, non-English — already passed the strict
            # confidence gate above, safe to use as-is.
            detected_lang = asr_lang
            logger.info('STEP 3: using ASR-reported language: %s', detected_lang)
    logger.info('STEP 3 end: language=%s, kept_text=%s', detected_lang, bool(raw_text))

    # ---- STEP 4: translation (only when source is not English) ----
    text_for_cleanup = raw_text
    if raw_text and detected_lang != 'en':
        logger.info('STEP 4 start: translation %s -> en (%s chars)',
                    detected_lang, len(raw_text))
        try:
            text_for_cleanup = _translate_to_english(raw_text, detected_lang)
            logger.info('STEP 4 end: translated to %s chars',
                        len(text_for_cleanup))
            if text_for_cleanup:
                logger.info('Translated to en: %s', text_for_cleanup)
        except Exception as exc:
            # Translation must never crash the worker. If the Marian model
            # download or generation blows up (HF Hub 416/xet failures, OOM,
            # ...), drop the text so we don't embed a foreign-language
            # transcription with the English embedding model.
            logger.warning('STEP 4 translation failed (%s); dropping lyrics', exc)
            text_for_cleanup = ''
    else:
        logger.info('STEP 4 skip: source already %s, no translation needed (%s chars)',
                    detected_lang, len(text_for_cleanup))

    # ---- STEP 5: embedding + axes ----
    final_text = text_for_cleanup
    logger.info('STEP 5 start: embedding + axis scoring (chars=%s)',
                len(final_text))
    embedding = None
    axis_vector: np.ndarray = np.zeros(0, dtype=np.float32)
    if len(final_text) >= MIN_CHARS_FOR_EMBEDDING:
        tokenizer, model = load_topic_embedding_model()
        embedding = _embed_text(final_text, tokenizer, model)
        if embedding is not None:
            axis_vector = _score_axes(embedding)
    else:
        # Below threshold: treat the track as having no usable lyrics. The text
        # fields are blanked so callers never persist or display partial garbage.
        raw_text = ''
        text_for_cleanup = ''
        final_text = ''

    # ---- STEP 5b: instrumental fallback ----
    # If no usable embedding was produced (no lyrics in the audio AND no API
    # hit, or fewer than MIN_CHARS_FOR_EMBEDDING chars after the pipeline), fall
    # back to deterministic sentinel vectors. This lets us:
    #   * persist a row so future analysis runs skip the track,
    #   * cluster all instrumental tracks together in vector search,
    #   * keep them safely far from real lyrical embeddings (the axis sentinel
    #     is uniformly negative, which a softmax axis_vector can never be).
    if embedding is None or getattr(embedding, 'size', 0) == 0:
        try:
            embedding, axis_vector = _make_instrumental_sentinel()
            logger.info('STEP 5b: applied instrumental sentinel '
                        '(embedding_dim=%s, axis_dim=%s)',
                        embedding.shape[0], axis_vector.shape[0])
        except Exception as exc:
            logger.warning('Could not apply instrumental sentinel: %s', exc)

    logger.info('STEP 5 end: embedding=%s axis_vector_dim=%s',
                None if embedding is None else embedding.shape,
                int(axis_vector.shape[0]) if axis_vector is not None else 0)

    return {
        'text': raw_text,
        'translated_text': text_for_cleanup,
        'final_text': final_text,
        'language': detected_lang,
        'used_seconds': used_seconds,
        'embedding': embedding,
        'axis_vector': axis_vector,
    }
