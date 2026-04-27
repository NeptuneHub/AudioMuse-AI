import os
import re
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np

try:
    from transformers import AutoModel, AutoTokenizer, pipeline
except ImportError:  # pragma: no cover
    AutoModel = None
    AutoTokenizer = None
    pipeline = None

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
    import torch
except ImportError:  # pragma: no cover
    torch = None

if DetectorFactory is not None:
    DetectorFactory.seed = 0


def set_cpu_threading(num_threads: Optional[int] = None) -> None:
    if num_threads is None or num_threads <= 0:
        return

    os.environ.setdefault('OMP_NUM_THREADS', str(num_threads))
    os.environ.setdefault('MKL_NUM_THREADS', str(num_threads))
    os.environ.setdefault('VECLIB_MAXIMUM_THREADS', str(num_threads))
    os.environ.setdefault('OPENBLAS_NUM_THREADS', str(num_threads))
    os.environ.setdefault('NUMEXPR_NUM_THREADS', str(num_threads))

    if torch is not None:
        try:
            torch.set_num_threads(num_threads)
            torch.set_num_interop_threads(num_threads)
        except Exception:
            pass


_worker_model = None



SUPPORTED_AUDIO_EXTENSIONS = {
    '.wav', '.mp3', '.m4a', '.flac', '.ogg', '.opus', '.aac', '.aiff', '.aif', '.mp4'
}

DEFAULT_SAMPLE_RATE = 16000
DEFAULT_SEGMENT_DURATION = 60.0
DEFAULT_LLM_MAX_TOKENS = 256
DEFAULT_LLM_MAX_CHUNK_LENGTH = 640
DEFAULT_ROBERTA_EMBEDDING_MODEL = 'roberta-base'
MUSIC_ANALYSIS_AXES = {
    "AXIS_1_SETTING": {
        "description": "The primary physical or environmental container of the song.",
        "labels": {
            "URBAN": "Cities, skyscrapers, streets, neon, traffic, and industrial zones.",
            "WILDERNESS": "Nature in its raw state: forests, mountains, oceans, and deserts.",
            "INTERIOR": "Enclosed private or public spaces: rooms, bars, hallways, or houses.",
            "TRANSIT": "Active movement: cars, trains, planes, or walking the open road.",
            "EXTRATERRESTRIAL": "Outer space, planetary bodies, and the cosmic void.",
            "SURREAL_ABSTRACT": "Non-physical realms, dreams, or places that defy physics."
        }
    },
    "AXIS_2_SOCIAL_DYNAMIC": {
        "description": "The target or partner of the narrator's communication.",
        "labels": {
            "SOLITARY": "Introspective monologue; the narrator is alone with their thoughts.",
            "ROMANTIC": "Interaction with a lover, crush, or ex-partner.",
            "KINSHIP": "Family structures: parents, children, siblings, or ancestors.",
            "COLLECTIVE": "A crowd, a friend group, 'the youth', or society as a whole.",
            "ADVERSARIAL": "A rival, an enemy, 'the system', or an oppressor.",
            "DIVINE": "A higher power, God, spirits, or the universe itself."
        }
    },
    "AXIS_3_EMOTIONAL_VALENCE": {
        "description": "The psychological tone (Nostalgia = Retrospective + Melancholic).",
        "labels": {
            "RADIANT": "Joy, euphoria, celebration, and high-energy optimism.",
            "MELANCHOLIC": "Sadness, grief, longing, and quiet despair.",
            "VOLATILE": "Anger, frustration, chaos, and intense restlessness.",
            "VULNERABLE": "Fear, anxiety, paranoia, and the feeling of being exposed.",
            "SERENE": "Acceptance, peace, calmness, and emotional stillness.",
            "NUMB": "Boredom, apathy, emptiness, and emotional detachment."
        }
    },
    "AXIS_4_NARRATIVE_TEMPORALITY": {
        "description": "The 'When' and 'How' of the lyrical structure.",
        "labels": {
            "RETROSPECTIVE": "Memory-based; looking back at what has passed.",
            "CHRONICLE": "The 'now'; a linear description of events as they happen.",
            "EXISTENTIAL": "Philosophical pondering on concepts like time, life, or death.",
            "STORYTELLING": "Narrating the life or actions of a third-party character/fable.",
            "DIRECT_PLEA": "A targeted message or letter to a 'you' with an immediate goal."
        }
    },
    "AXIS_5_THEMATIC_WEIGHT": {
        "description": "The gravity and intent behind the lyrical content.",
        "labels": {
            "TRIVIAL": "Lighthearted, casual, and focused on style, fun, or the moment.",
            "MORTAL": "Deeply serious, focused on legacy, life's end, and human struggle.",
            "POLITICAL": "Observation of power, justice, war, and societal mechanics.",
            "SENSORIAL": "Focus on physical indulgence: drinking, dancing, and pleasure."
        }
    }
}


def load_audio(file_path: str, sr: Optional[int] = DEFAULT_SAMPLE_RATE) -> Tuple[np.ndarray, int]:
    """Load audio to a mono NumPy array at the target sample rate."""
    if sf is not None:
        data, sample_rate = sf.read(file_path, dtype='float32')
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        if sr is not None and sample_rate != sr:
            if librosa is None:
                raise RuntimeError("librosa is required to resample audio")
            data = librosa.resample(data, orig_sr=sample_rate, target_sr=sr)
            sample_rate = sr
        return data.astype(np.float32), sample_rate

    if librosa is not None:
        data, sample_rate = librosa.load(file_path, sr=sr, mono=True)
        return data.astype(np.float32), sample_rate

    raise RuntimeError("Missing audio backends: install soundfile or librosa")


def compute_energy_envelope(y: np.ndarray, sr: int, window_duration: float = 1.0, hop_duration: float = 0.25) -> Tuple[np.ndarray, float]:
    window_length = max(1, int(round(window_duration * sr)))
    hop_length = max(1, int(round(hop_duration * sr)))
    squared = np.square(y)
    kernel = np.ones(window_length, dtype=np.float32) / window_length
    energy = np.convolve(squared, kernel, mode='valid')[::hop_length]
    timestamps = np.arange(len(energy), dtype=np.float32) * hop_duration
    return energy, timestamps


def find_active_segment_start(y: np.ndarray, sr: int, segment_duration: float = DEFAULT_SEGMENT_DURATION) -> float:
    energy, timestamps = compute_energy_envelope(y, sr)
    if energy.size == 0:
        return 0.0

    max_energy = float(np.max(energy))
    median_energy = float(np.median(energy))
    threshold = max(max_energy * 0.08, median_energy * 1.5, 1e-8)

    active = energy >= threshold
    if not np.any(active):
        return 0.0

    frame_duration = timestamps[1] - timestamps[0] if len(timestamps) > 1 else segment_duration
    required_frames = int(np.ceil(segment_duration / frame_duration))
    if required_frames < 1:
        required_frames = 1

    energy_sums = np.convolve(energy, np.ones(required_frames, dtype=np.float32), mode='valid')
    file_duration = len(y) / sr
    max_start = max(0.0, file_duration - segment_duration)

    if max_start <= 0.0:
        return 0.0

    start_times = np.arange(len(energy_sums), dtype=np.float32) * frame_duration
    start_times = np.minimum(start_times, max_start)

    center_start = max_start / 2.0
    distance = np.abs(start_times - center_start)
    normalized_distance = distance / max(center_start, 1.0)
    center_penalty = np.minimum(1.0, normalized_distance ** 2)
    center_bias = 0.35

    scores = energy_sums * (1.0 - center_bias * center_penalty)
    best_frame_index = int(np.argmax(scores))
    start_time = float(start_times[best_frame_index])

    return start_time


def extract_segment(y: np.ndarray, sr: int, start_time: float, segment_duration: float = DEFAULT_SEGMENT_DURATION) -> np.ndarray:
    start_sample = int(round(start_time * sr))
    end_sample = start_sample + int(round(segment_duration * sr))
    return y[start_sample:end_sample]


DEFAULT_QWEN_ASR_MODEL = 'Qwen/Qwen3-ASR-0.6B'


def load_whisper_model(
    model_name: str = 'tiny',
    device: str = 'cpu',
    num_threads: Optional[int] = None,
):
    set_cpu_threading(num_threads)

    if whisper is None:
        raise RuntimeError('The whisper package is not installed. Install openai-whisper before running transcription.')

    return whisper.load_model(model_name, device=device)


def transcribe_audio_segment(
    audio: np.ndarray,
    sr: int,
    model,
    language: Optional[str] = None,
    asr_backend: str = 'whisper',
) -> Dict[str, object]:
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    if len(audio) == 0:
        return {
            'text': '',
            'language': language,
            'duration': 0.0,
        }

    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
        temp_path = tmp_file.name

    try:
        sf.write(temp_path, audio, sr, subtype='PCM_16')
        if asr_backend == 'whisper':
            result = model.transcribe(temp_path, language=language, fp16=False)
            text = result.get('text', '').strip()
            language_out = result.get('language', language)
        elif asr_backend == 'qwen':
            kwargs = {}
            if language:
                kwargs['language'] = language
            result = model(temp_path, **kwargs)
            if isinstance(result, dict):
                text = str(result.get('text', '')).strip()
                language_out = language
            else:
                text = str(result).strip()
                language_out = language
        else:
            raise RuntimeError(f'Unsupported ASR backend: {asr_backend}')

        return {
            'text': text,
            'language': language_out,
            'duration': len(audio) / sr,
        }
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass


def extract_title_and_artist(file_path: str) -> Tuple[str, str]:
    filename = Path(file_path).stem
    if ' - ' in filename:
        parts = [part.strip() for part in filename.split(' - ', 1)]
        if len(parts) == 2:
            return parts[1], parts[0]
    return filename, ''


def load_llama_cpp_model(model_path: str, num_threads: Optional[int] = None):
    if Llama is None:
        raise RuntimeError('The llama-cpp-python package is not installed. Install llama-cpp-python before running cleanup.')

    if not os.path.exists(model_path):
        raise RuntimeError(f'LLM model file not found: {model_path}')

    kwargs = {
        'model_path': model_path,
        'n_threads': num_threads or 1,
        'n_gpu_layers': 0,
        'verbose': False,
    }
    return Llama(**kwargs)


def detect_language(text: str, model=None) -> Tuple[str, float]:
    if not text or not text.strip():
        return 'en', 0.0

    if detect_langs is None:
        raise RuntimeError('Install langdetect to use language detection.')

    try:
        candidates = detect_langs(text.replace('\n', ' '))
    except Exception:
        return 'en', 0.0

    if not candidates:
        return 'en', 0.0

    best = candidates[0]
    return best.lang, float(best.prob)


_MARIAN_MODEL_CACHE: Dict[str, Tuple[Optional[object], Optional[object]]] = {}


def split_text_into_word_chunks(text: str, max_words: int = 50) -> List[str]:
    text = text.strip()
    if not text:
        return []

    words = text.split()
    if len(words) <= max_words:
        return [text]

    return [' '.join(words[i:i + max_words]) for i in range(0, len(words), max_words)]


def load_roberta_embedding_model(model_name: str = DEFAULT_ROBERTA_EMBEDDING_MODEL):
    if AutoTokenizer is None or AutoModel is None:
        raise RuntimeError('The transformers package is not installed. Install transformers to compute embeddings.')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return tokenizer, model


DEFAULT_TOPIC_EMBEDDING_MODEL = 'intfloat/e5-base-v2'

def load_topic_embedding_model(model_name: str = DEFAULT_TOPIC_EMBEDDING_MODEL):
    return load_roberta_embedding_model(model_name)

def compute_topic_label_embeddings(labels: List[str], tokenizer, model) -> np.ndarray:
    embeddings: List[np.ndarray] = []
    for label in labels:
        embedding = embed_text_with_roberta(label, tokenizer, model)
        if embedding is not None:
            embeddings.append(embedding)
    if not embeddings:
        return np.zeros((0, 0), dtype=np.float32)
    return np.stack(embeddings)

def embed_text_with_roberta(text: str, tokenizer, model) -> Optional[np.ndarray]:
    if tokenizer is None or model is None:
        raise RuntimeError('Roberta embedding model is not loaded.')
    if not text or not text.strip():
        return None
    encoded = tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=128,
        return_tensors='pt',
    )
    with torch.no_grad():
        outputs = model(**encoded)
    last_hidden = outputs.last_hidden_state
    attention_mask = encoded['attention_mask'].unsqueeze(-1).expand(last_hidden.size()).float()
    summed = (last_hidden * attention_mask).sum(1)
    counts = attention_mask.sum(1).clamp(min=1e-9)
    pooled = (summed / counts).squeeze(0)
    vector = pooled.cpu().numpy()
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm
    return vector


def compute_axis_label_embeddings(axes, tokenizer, model):
    axis_label_map = {}
    axis_embeddings = {}
    for axis_name, axis_meta in axes.items():
        labels = list(axis_meta.get('labels', {}).items())
        descriptions = [description for _, description in labels]
        axis_label_map[axis_name] = labels
        axis_embeddings[axis_name] = compute_topic_label_embeddings(descriptions, tokenizer, model)
    return axis_label_map, axis_embeddings


def softmax_scores(values: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    if values.size == 0:
        return values
    temperature = temperature if temperature > 0 else 1.0
    scaled = values / temperature
    shifted = scaled - np.max(scaled)
    exp_values = np.exp(shifted)
    total = float(np.sum(exp_values))
    return exp_values / total if total > 0 else np.zeros_like(values)


def score_text_against_axes(text: str, tokenizer, model, axis_label_map, axis_embeddings, temperature: float = 0.1):
    song_embedding = embed_text_with_roberta(text, tokenizer, model)
    if song_embedding is None:
        return {}
    axis_scores = {}
    for axis_name, labels in axis_label_map.items():
        embedding_matrix = axis_embeddings.get(axis_name)
        if embedding_matrix is None or embedding_matrix.size == 0:
            axis_scores[axis_name] = []
            continue
        similarities = embedding_matrix.dot(song_embedding)
        probabilities = softmax_scores(similarities, temperature=temperature)
        label_scores = [
            {'label': label, 'description': description, 'score': float(probabilities[idx])}
            for idx, (label, description) in enumerate(labels)
        ]
        label_scores.sort(key=lambda item: item['score'], reverse=True)
        axis_scores[axis_name] = label_scores
    return axis_scores


def split_transcription_into_chunks(
    transcription: str,
    max_chars: int = DEFAULT_LLM_MAX_CHUNK_LENGTH,
    max_words: int = 50,
) -> List[str]:
    transcription = transcription.strip()
    if not transcription:
        return []

    words = transcription.split()
    if len(words) <= max_words and len(transcription) <= max_chars:
        return [transcription]

    chunks: List[str] = []
    start = 0
    while start < len(words):
        end = min(start + max_words, len(words))
        chunk = ' '.join(words[start:end]).strip()
        if not chunk:
            break
        chunks.append(chunk)
        start = end
    return chunks


def normalize_cleaned_text(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r'\s+([?.!,;:])', r'\1', cleaned)
    cleaned = re.sub(r'\s*\n\s*', '\n', cleaned)
    cleaned = re.sub(r'(^|[.!?]\s+)(i)\b', lambda m: m.group(1) + 'I', cleaned)
    cleaned = re.sub(r'\s{2,}', ' ', cleaned)
    return cleaned


def similarity_score(text_a: str, text_b: str) -> float:
    a_tokens = [token.lower() for token in re.findall(r"[A-Za-z']+", text_a)]
    b_tokens = [token.lower() for token in re.findall(r"[A-Za-z']+", text_b)]
    if not a_tokens or not b_tokens:
        return 0.0
    common = set(a_tokens) & set(b_tokens)
    return len(common) / max(len(set(a_tokens)), len(set(b_tokens)))


def clean_transcription_with_llama(
    title: str,
    artist: str,
    transcription: str,
    model,
    translate_to_english: bool = True,
    max_tokens: int = DEFAULT_LLM_MAX_TOKENS,
    temperature: float = 0.2,
) -> str:
    transcription = transcription.strip()
    if not transcription:
        return ''

    chunks = split_transcription_into_chunks(transcription)
    cleaned_chunks: List[str] = []

    for index, chunk in enumerate(chunks, start=1):
        prompt = (
            "You are a song transcription cleanup assistant.\n"
            "This text is a Whisper transcription output.\n"
            "Your job is to fix only obvious transcription mistakes and minor formatting issues.\n"
            "Do not invent any new lyrics, lines, words, or ideas.\n"
            "Do not rephrase the song, do not add new content, and do not change the meaning.\n"
            "If a phrase already reads like a valid lyric, keep it exactly as written unless there is a clear transcription error.\n"
            "Correct only the words that appear to be mistaken, misspelled, or clearly generated incorrectly.\n"
            "Preserve the original phrasing, line breaks, and sentence structure unless an obvious error must be fixed.\n"
            "Do not repeat any content, and do not merge or split lines arbitrarily.\n"
            "If you are not sure, leave the original text unchanged.\n"
            "Do not include any metadata, title, artist name, or file information in the output.\n"
            "Output ONLY the cleaned lyrics text, with no explanation, no analysis, and no extra labels.\n\n"
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
            stop=[
                "\n\n",
                "\nDo not invent new lyrics",
                "\nOutput ONLY",
                "\nDo not include any metadata",
                "\nCorrect obvious transcription errors",
            ],
        )
        choices = response.get('choices') if isinstance(response, dict) else None
        if choices and isinstance(choices, list) and len(choices) > 0:
            text = choices[0].get('text', '')
        else:
            text = str(response)

        markers = ['Cleaned lyrics text:', 'Cleaned transcription:', 'Cleaned:', 'Output:']
        for marker in markers:
            if marker in text:
                text = text.rsplit(marker, 1)[-1]
                break

        cleaned_chunk = normalize_cleaned_text(text)
        cleaned_chunk = '\n'.join(
            line for line in cleaned_chunk.splitlines()
            if line.strip() and not re.match(
                r'^(Output:|Cleaned transcription is:|Cleaned:|The song is titled|The song is in a language|This transcription was|This song is|This transcription is|The title:|The artist:|Artist:|Raw transcription:|If the transcription is not in English|Do not change or add any other information|I am ready to start the cleanup process|I have transcribed it into this state|This is the raw transcription|This is the cleaned lyrics text|I don\'t have enough context|It is now up to you|This transcription was done|Do not change the number of lines|remove repeated artifacts|normalize punctuation|Do not include any metadata|Output ONLY|Correct obvious transcription errors|Do not invent new lyrics|Do not include any metadata, title, artist name, or file information)',
                line.strip(),
                re.IGNORECASE,
            )
        ).strip()

        if not cleaned_chunk or len(cleaned_chunk.split()) < 4:
            cleaned_chunk = normalize_cleaned_text(chunk)
        else:
            sim = similarity_score(chunk, cleaned_chunk)
            if sim < 0.70:
                retry_prompt = prompt.replace(
                    "Raw transcription:\n",
                    "Look, this cleaned output is too different from the raw transcription. "
                    "Please be more precise and keep the output as close to the original text as possible.\n\n"
                    "Raw transcription:\n",
                )
                retry_response = model.create_completion(
                    retry_prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=0.75,
                    presence_penalty=0.5,
                    repeat_penalty=1.15,
                    echo=False,
                    stop=[
                        "\n\n",
                        "\nDo not invent new lyrics",
                        "\nOutput ONLY",
                        "\nDo not include any metadata",
                        "\nCorrect obvious transcription errors",
                    ],
                )
                retry_choices = retry_response.get('choices') if isinstance(retry_response, dict) else None
                if retry_choices and isinstance(retry_choices, list) and len(retry_choices) > 0:
                    retry_text = retry_choices[0].get('text', '')
                else:
                    retry_text = str(retry_response)

                for marker in markers:
                    if marker in retry_text:
                        retry_text = retry_text.rsplit(marker, 1)[-1]
                        break

                retry_chunk = normalize_cleaned_text(retry_text)
                retry_chunk = '\n'.join(
                    line for line in retry_chunk.splitlines()
                    if line.strip() and not re.match(
                        r'^(Output:|Cleaned transcription is:|Cleaned:|The song is titled|The song is in a language|This transcription was|This song is|This transcription is|The title:|The artist:|Artist:|Raw transcription:|If the transcription is not in English|Do not change or add any other information|I am ready to start the cleanup process|I have transcribed it into this state|This is the raw transcription|This is the cleaned lyrics text|I don\'t have enough context|It is now up to you|This transcription was done|Do not change the number of lines|remove repeated artifacts|normalize punctuation|Do not include any metadata|Output ONLY|Correct obvious transcription errors|Do not invent new lyrics|Do not include any metadata, title, artist name, or file information)',
                        line.strip(),
                        re.IGNORECASE,
                    )
                ).strip()

                retry_sim = similarity_score(chunk, retry_chunk)
                if retry_chunk and len(retry_chunk.split()) >= 4 and retry_sim >= sim:
                    cleaned_chunk = retry_chunk
                else:
                    cleaned_chunk = normalize_cleaned_text(chunk)

        cleaned_chunks.append(cleaned_chunk)

    return '\n\n'.join(cleaned_chunks).strip()

def transcribe_file_segment(
    audio_source: Union[str, Path, np.ndarray],
    model,
    sr: Optional[int] = None,
    model_name: str = 'tiny',
    device: str = 'cpu',
    segment_duration: float = DEFAULT_SEGMENT_DURATION,
    full_song: bool = False,
    num_threads: Optional[int] = None,
    language: Optional[str] = None,
    llama_model=None,
    source_path: Optional[Union[str, Path]] = None,
    asr_backend: str = 'whisper',
) -> Dict[str, object]:
    started = time.perf_counter()
    if isinstance(audio_source, np.ndarray):
        if sr is None:
            raise ValueError('Sample rate must be provided when passing audio arrays.')
        y = audio_source
        if y.ndim > 1:
            y = np.mean(y, axis=1)
    else:
        y, sr = load_audio(str(audio_source), sr=DEFAULT_SAMPLE_RATE)

    if full_song:
        start_time = 0.0
        segment = y
    else:
        start_time = find_active_segment_start(y, sr, segment_duration=segment_duration)
        segment = extract_segment(y, sr, start_time, segment_duration=segment_duration)

    whisper_start = time.perf_counter()
    transcription = transcribe_audio_segment(segment, sr, model, language=language, asr_backend=asr_backend)
    asr_seconds = time.perf_counter() - whisper_start
    text = transcription['text']
    segment_duration = transcription['duration']

    cleaned_text = None
    llama_seconds = None
    if llama_model is not None:
        title, artist = extract_title_and_artist(str(source_path or audio_source))
        llama_start = time.perf_counter()
        cleaned_text = clean_transcription_with_llama(title, artist, text, llama_model)
        llama_seconds = time.perf_counter() - llama_start

    total_seconds = time.perf_counter() - started
    return {
        'file_path': str(source_path or audio_source),
        'start_time': start_time,
        'segment_duration': segment_duration,
        'text': text,
        'cleaned_text': cleaned_text,
        'language': language,
        'model_name': model_name,
        'device': device,
        'asr_backend': asr_backend,
        'whisper_seconds': asr_seconds,
        'asr_seconds': asr_seconds,
        'llama_seconds': llama_seconds,
        'total_seconds': total_seconds,
        'full_song': full_song,
    }


