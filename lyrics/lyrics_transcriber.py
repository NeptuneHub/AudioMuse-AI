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
except ImportError:  # pragma: no cover
    torch = None

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
MIN_WORDS_FOR_EMBEDDING = 4

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


def load_whisper_model(model_name: str = 'small', device: str = 'cpu',
                      num_threads: Optional[int] = None):
    global _whisper_model, _whisper_model_name
    if whisper is None:
        raise RuntimeError('openai-whisper is not installed.')

    threads = num_threads or get_lyrics_threads()
    _apply_thread_env(threads)

    if _whisper_model is not None and _whisper_model_name == model_name:
        return _whisper_model

    try:
        from config import LYRICS_MODEL_DIR
    except Exception:
        LYRICS_MODEL_DIR = '/app/model'

    local_pt = os.path.join(LYRICS_MODEL_DIR, f'{model_name}.pt')
    target = local_pt if os.path.isfile(local_pt) else model_name
    logger.info('Loading Whisper model %r (threads=%s) from %s', model_name, threads, target)
    _whisper_model = whisper.load_model(target, device=device, download_root=LYRICS_MODEL_DIR)
    _whisper_model_name = model_name
    logger.info('Whisper model %r ready', model_name)
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

    logger.info('Loading LLaMA model %s (threads=%s)', model_path, threads)
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
    logger.info('Loading embedding model %s', target)
    _embedding_tokenizer = AutoTokenizer.from_pretrained(str(target))
    _embedding_model = AutoModel.from_pretrained(str(target))
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
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
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

def _clean_with_llama(text: str, model, max_tokens: int = 256,
                      temperature: float = 0.2) -> str:
    if not text or not text.strip():
        return ''
    chunks = _split_into_word_chunks(text)
    cleaned: List[str] = []
    for index, chunk in enumerate(chunks, start=1):
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
        try:
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
        except Exception as exc:
            logger.warning('LLaMA cleanup chunk %s/%s failed: %s; using raw chunk',
                           index, len(chunks), exc)
            cleaned.append(_normalize_text(chunk))
            continue
        text_out = ''
        if isinstance(response, dict):
            choices = response.get('choices') or []
            if choices:
                text_out = choices[0].get('text', '')
        if not text_out:
            text_out = chunk
        cleaned.append(_normalize_text(text_out))
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


def _score_axes(embedding: np.ndarray,
                temperature: float = 0.1) -> Dict[str, List[Dict[str, object]]]:
    label_map, axis_embeddings = _get_axis_embeddings()
    out: Dict[str, List[Dict[str, object]]] = {}
    for axis_name, labels in label_map.items():
        matrix = axis_embeddings.get(axis_name)
        if matrix is None or matrix.size == 0:
            out[axis_name] = []
            continue
        sims = matrix.dot(embedding)
        probs = _softmax(sims, temperature)
        scored = [
            {'label': label, 'description': description, 'score': float(probs[idx])}
            for idx, (label, description) in enumerate(labels)
        ]
        scored.sort(key=lambda item: item['score'], reverse=True)
        out[axis_name] = scored
    return out


# ---------------------------------------------------------------------------
# Public orchestrator
# ---------------------------------------------------------------------------

def analyze_lyrics(audio: Optional[np.ndarray] = None,
                   sr: Optional[int] = None,
                   source_path: Optional[Union[str, Path]] = None,
                   use_llm_cleanup: bool = True) -> Dict[str, object]:
    """Run the full lyrics pipeline.

    Either ``audio`` (mono float32 + ``sr``) or ``source_path`` must be supplied.
    Returns a dict with ``text``, ``cleaned_text``, ``language``, ``embedding``
    and ``axis_scores``. Raises if a required model/source is missing.
    """
    threads = get_lyrics_threads()
    _apply_thread_env(threads)

    # ---- STEP 1: audio ----
    logger.info('STEP 1 start: prepare audio (max %.1fs)', MAX_AUDIO_SECONDS)
    if audio is None or sr is None:
        if not source_path:
            raise ValueError('analyze_lyrics requires either audio+sr or source_path')
        if not os.path.exists(str(source_path)):
            raise FileNotFoundError(f'Audio source not found: {source_path}')
        audio, sr = _load_audio_from_path(str(source_path), sr=DEFAULT_SAMPLE_RATE)
    audio_clip, used_seconds = _clip_audio(audio, sr)
    logger.info('STEP 1 end: audio ready, used=%.2fs samples=%s sr=%s',
                used_seconds, len(audio_clip), sr)

    # ---- STEP 2: whisper transcription ----
    logger.info('STEP 2 start: whisper transcription (threads=%s)', threads)
    whisper_model = load_whisper_model(num_threads=threads)
    transcription = _transcribe(audio_clip, sr, whisper_model)
    raw_text = (transcription.get('text') or '').strip()
    logger.info('STEP 2 end: transcript length=%s chars / %s words',
                len(raw_text), len(raw_text.split()))

    # ---- STEP 3: language detection ----
    logger.info('STEP 3 start: language detection')
    detected_lang = transcription.get('language') or 'en'
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
    axis_scores: Dict[str, List[Dict[str, object]]] = {}
    if len(final_text.split()) >= MIN_WORDS_FOR_EMBEDDING:
        tokenizer, model = load_topic_embedding_model()
        embedding = _embed_text(final_text, tokenizer, model)
        if embedding is not None:
            axis_scores = _score_axes(embedding)
    logger.info('STEP 6 end: embedding=%s axis_axes=%s',
                None if embedding is None else embedding.shape,
                len(axis_scores))

    return {
        'text': raw_text,
        'translated_text': text_for_cleanup,
        'cleaned_text': cleaned_text,
        'final_text': final_text,
        'language': detected_lang,
        'used_seconds': used_seconds,
        'embedding': embedding,
        'axis_scores': axis_scores,
    }
