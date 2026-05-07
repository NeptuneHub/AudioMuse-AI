"""Small lyrics analysis orchestrator (torch-free ONNX backend).

Models are assumed to be already present inside the container; this module never
downloads anything. It exposes a single high level entry point ``analyze_lyrics``
plus the cached model loaders used by the worker bootstrap.

Pipeline (each step emits a ``STEP X start`` and ``STEP X end`` log line):

    STEP 1  load / clip audio (max 4 minutes)
    STEP 2  whisper transcription           — lyrics.whisper_onnx (ONNX Runtime)
    STEP 3  language detection              — langdetect (pure Python)
    STEP 4  optional translation to English — opus-mt-mul-en ONNX + numpy
                                              greedy decode (lyrics.translation_onnx)
    STEP 6  e5 embedding + axis scoring      — onnxruntime + transformers tokenizer

The legacy STEP 5 (Qwen LLM cleanup) is gone — raw whisper output is good
enough for axis scoring, and dropping it saves ~900 MB of model weights and
the llama-cpp-python wheel.

torch is no longer a runtime dependency: the bare ``import torch`` near the
top of this file is wrapped in ``try/except`` so the CPU image (which does not
ship torch) loads cleanly and the per-feature ``torch is not None`` guards
turn into no-ops.
"""

from __future__ import annotations

import logging
import os
import re
import signal
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

# Whisper transcription: torch-free via raw onnxruntime against the bundled
# /app/model/whisper-small-onnx/ artifacts (encoder/decoder ONNX + tokenizer).
# See lyrics/whisper_onnx.py for the mel + greedy decode pipeline; the legacy
# ``openai-whisper`` package (which pulls torch and torchaudio) is gone.
try:
    from . import whisper_onnx  # type: ignore
except Exception:  # pragma: no cover - keep transcription optional
    whisper_onnx = None

try:
    from langdetect import detect_langs, DetectorFactory
except ImportError:  # pragma: no cover
    detect_langs = None
    DetectorFactory = None

from .axes import MUSIC_ANALYSIS_AXES, axis_columns
from .embeddings import (
    _embed_text,
    _get_axis_embeddings,
    load_topic_embedding_model,
)

try:
    import torch
except Exception:  # pragma: no cover - torch is optional on the CPU image
    torch = None

# Silero VAD: torch-free path via raw onnxruntime against the bundled
# /app/model/silero_vad.onnx model file. The legacy ``silero-vad`` PyPI
# package (which pulls torch + torchaudio) is no longer required.
try:
    from .silero_onnx import get_speech_timestamps
except Exception:  # pragma: no cover - keep VAD optional
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

# ---------------------------------------------------------------------------

def get_lyrics_threads() -> int:
    """Number of CPU threads to use for the lyrics ONNX sessions in this process.

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


def load_whisper_model(model_name: str = 'small', device: Optional[str] = None,
                      num_threads: Optional[int] = None):
    """Eager-load the bundled whisper-small ONNX session.

    Returns the ``lyrics.whisper_onnx`` module itself; the caller treats it
    as opaque and only passes it back to ``_transcribe``. ``device`` and
    ``model_name`` are accepted for backwards compatibility — the ONNX path
    is CPU-only and the bundled model is always whisper-small.
    """
    global _whisper_model, _whisper_model_name
    if whisper_onnx is None:
        raise RuntimeError('lyrics.whisper_onnx module is unavailable.')

    threads = num_threads or get_lyrics_threads()
    _apply_thread_env(threads)

    if _whisper_model is not None:
        return _whisper_model

    try:
        from config import LYRICS_MODEL_DIR
    except Exception:
        LYRICS_MODEL_DIR = '/app/model'
    target_dir = os.path.join(LYRICS_MODEL_DIR, 'whisper-small-onnx')

    logger.info('Loading whisper-small ONNX from %s (threads=%s)', target_dir, threads)
    whisper_onnx._load(target_dir)  # warm the session up front
    _whisper_model = whisper_onnx
    _whisper_model_name = model_name
    logger.info('whisper-small ONNX ready')
    return _whisper_model


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
# Voice activity detection (silero, ONNX) — keep only voiced regions before whisper
# ---------------------------------------------------------------------------

def _apply_vad(audio: np.ndarray, sr: int) -> np.ndarray:
    """Return a concatenation of voiced regions; fall back to ``audio`` on any issue."""
    if sr != 16000 or get_speech_timestamps is None:
        return audio
    try:
        ts = get_speech_timestamps(audio.astype(np.float32, copy=False), sample_rate=sr)
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
    """Transcribe ``audio`` (float32) via the lyrics.whisper_onnx module.

    ``model`` must be the module returned by :func:`load_whisper_model`.
    The audio is expected to be 16 kHz mono. ``language=None`` triggers the
    cheap single-step language probe inside whisper_onnx.
    """
    if len(audio) == 0:
        return {'text': '', 'language': language, 'duration': 0.0}
    if sr != 16000:
        # Defensive: whisper expects 16 kHz. Resample with librosa if available.
        if librosa is None:
            raise RuntimeError('librosa is required to resample audio to 16 kHz.')
        audio = librosa.resample(audio.astype(np.float32, copy=False),
                                  orig_sr=sr, target_sr=16000)
        sr = 16000
    return model.transcribe(audio, language=language, task='transcribe')


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


def _translate_to_english(text: str, source_lang: str) -> str:
    """Translate ``text`` to English via the bundled ONNX opus-mt-mul-en model.

    A single ``Helsinki-NLP/opus-mt-mul-en`` ONNX export covers ~70 source
    languages and is loaded by ``lyrics.translation_onnx``. The torch-based
    per-language Marian path is gone: this function uses raw onnxruntime +
    a numpy greedy-decode loop.

    Returns ``''`` on any failure so the caller treats the track as
    instrumental, matching the previous Marian fallback behaviour.
    """
    if not text or source_lang.lower() == 'en':
        return text

    try:
        from .translation_onnx import translate_to_english as _onnx_translate
    except Exception as exc:
        logger.warning('Translator module unavailable (%s); dropping lyrics', exc)
        return ''

    pieces: List[str] = []
    for chunk in _split_into_word_chunks(text):
        try:
            piece = _onnx_translate(chunk)
        except Exception as exc:
            logger.warning('ONNX translation chunk failed (%s); dropping lyrics', exc)
            return ''
        if not piece:
            logger.warning('ONNX translation returned empty for chunk; dropping lyrics')
            return ''
        pieces.append(piece)
    return ' '.join(pieces)


# ---------------------------------------------------------------------------
# Embedding + axis scoring
# ---------------------------------------------------------------------------

def _softmax(values: np.ndarray, temperature: float) -> np.ndarray:
    if values.size == 0:
        return values
    temperature = temperature if temperature > 0 else 1.0
    scaled = values / temperature
    shifted = scaled - np.max(scaled)
    exp = np.exp(shifted)
    total = float(np.sum(exp))
    return exp / total if total > 0 else np.zeros_like(values)




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
                   use_llm_cleanup: bool = True,  # kept for backwards-compat; ignored
                   artist: Optional[str] = None,
                   track: Optional[str] = None,
                   track_id: Optional[str] = None) -> Dict[str, object]:
    """Run the full lyrics pipeline.

    Either ``audio`` (mono float32 + ``sr``) or ``source_path`` must be supplied.
    Pipeline order:
      STEP -1  media server embedded lyrics (Jellyfin/Emby/Navidrome/Lyrion, MUSICSERVER_LYRICS_TIMEOUT seconds)
      STEP  0  user-configured external APIs (slot 1 then slot 2)
      STEP  1  load / clip audio
      STEP  1b VAD pre-filter (silero ONNX)
      STEP  2  Whisper transcription (ONNX)
      STEP  3  language detection (langdetect)
      STEP  4  Translation to English (opus-mt-mul-en ONNX, if non-English)
      STEP  6  e5 embedding + axis scoring (ONNX)
    The legacy STEP 5 (Qwen LLM cleanup) was removed; ``use_llm_cleanup`` is
    accepted but ignored.
    Returns a dict with ``text``, ``cleaned_text``, ``language``, ``embedding``
    and ``axis_vector`` (float32 numpy array in canonical axis_columns() order).
    Raises if a required model/source is missing.
    """
    threads = get_lyrics_threads()
    _apply_thread_env(threads)

    used_seconds = 0.0
    raw_text = ''
    detected_lang = 'en'

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
                    logger.info('STEP -1 end: media server HIT (%s words) - skipping STEPS 0, 1, 1b, 2',
                                len(raw_text.split()))
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
                logger.info('STEP 0 end: API HIT (%s chars / %s words) - skipping STEPS 1, 1b, 2',
                            len(raw_text), len(raw_text.split()))
                logger.info('STEP 0 raw API output: %s', raw_text)
            else:
                logger.info('STEP 0 end: API MISS - falling back to whisper')
        else:
            logger.info('STEP 0 end: API skipped (disabled or missing artist/track)')
    else:
        logger.info('STEP 0 skipped: already have lyrics from media server')

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
            logger.info('STEP 2 word count below %s — skipping STEPS 3-4', MIN_WORDS_FOR_CLEANUP)
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

    # STEP 5 (Qwen LLM cleanup) removed — raw whisper output is good enough for
    # axis scoring and the GGUF model + llama-cpp-python dependency are gone.
    cleaned_text = ''
    final_text = text_for_cleanup

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
