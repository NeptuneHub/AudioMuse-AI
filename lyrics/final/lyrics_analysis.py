import os
import re
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np

from config import (
    LYRICS_DEFAULT_SAMPLE_RATE,
    LYRICS_DEFAULT_SEGMENT_DURATION,
    LYRICS_DEFAULT_TOPIC_EMBEDDING_CACHE_DIR,
    LYRICS_DEFAULT_TOPIC_EMBEDDING_MODEL,
    LYRICS_DEFAULT_MARIAN_PREFIX,
    LYRICS_LLM_MODEL_PATH,
    LYRICS_LLM_MODEL_URL,
    LYRICS_MAX_SONGS_TO_ANALYZE,
    LYRICS_MODEL_DIR,
    LYRICS_SONGS_DIR,
    LYRICS_SUPPORTED_AUDIO_EXTENSIONS,
    LYRICS_WHISPER_MODEL,
)

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

if DetectorFactory is not None:
    DetectorFactory.seed = 0


# Current cleanup model: Qwen 2.5 1.5B instruct q4_k_m

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

QUERY_DEFINITIONS = [
    {
        'name': 'Atmospheric Search',
        'weights': {'TRANSIT': 1.0, 'SOLITARY': 1.0, 'EXISTENTIAL': 1.0},
    },
    {
        'name': 'High-Energy Search',
        'weights': {'RADIANT': 1.0, 'SENSORIAL': 1.0, 'URBAN': 1.0},
    },
    {
        'name': 'Moody Search',
        'weights': {'MELANCHOLIC': 1.0, 'INTERIOR': 1.0, 'SOLITARY': 1.0},
    },
    {
        'name': 'Abstract Search',
        'weights': {'SURREAL_ABSTRACT': 1.0, 'EXISTENTIAL': 1.0},
    },
    {
        'name': 'Antagonistic Search',
        'weights': {'ADVERSARIAL': 1.0, 'VOLATILE': 1.0, 'POLITICAL': 1.0},
    },
    {
        'name': 'Romantic Transit Search',
        'weights': {'TRANSIT': 1.0, 'ROMANTIC': 1.0, 'VULNERABLE': 1.0, 'MORTAL': 1.0},
    },
]


def set_cpu_threading(num_threads: Optional[int] = None) -> None:
    if num_threads is None or num_threads <= 0:
        return
    os.environ.setdefault('OMP_NUM_THREADS', str(num_threads))
    os.environ.setdefault('MKL_NUM_THREADS', str(num_threads))
    os.environ.setdefault('VECLIB_MAXIMUM_THREADS', str(num_threads))
    os.environ.setdefault('OPENBLAS_NUM_THREADS', str(num_threads))
    os.environ.setdefault('NUMEXPR_NUM_THREADS', str(num_threads))


def find_audio_files(root: Path) -> List[Path]:
    tracks = []
    for ext in LYRICS_SUPPORTED_AUDIO_EXTENSIONS:
        tracks.extend(sorted(root.rglob(f'*{ext}')))
    return sorted(tracks)


def load_audio(file_path: str, sr: int = LYRICS_DEFAULT_SAMPLE_RATE) -> Tuple[np.ndarray, int]:
    if sf is not None:
        data, sample_rate = sf.read(file_path, dtype='float32')
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        if sr is not None and sample_rate != sr:
            if librosa is None:
                raise RuntimeError('librosa is required to resample audio')
            data = librosa.resample(data, orig_sr=sample_rate, target_sr=sr)
            sample_rate = sr
        return data.astype(np.float32), sample_rate
    if librosa is not None:
        data, sample_rate = librosa.load(file_path, sr=sr, mono=True)
        return data.astype(np.float32), sample_rate
    raise RuntimeError('Missing audio backends: install soundfile or librosa')


def compute_energy_envelope(y: np.ndarray, sr: int, window_duration: float = 1.0, hop_duration: float = 0.25) -> Tuple[np.ndarray, float]:
    window_length = max(1, int(round(window_duration * sr)))
    hop_length = max(1, int(round(hop_duration * sr)))
    squared = np.square(y)
    kernel = np.ones(window_length, dtype=np.float32) / window_length
    energy = np.convolve(squared, kernel, mode='valid')[::hop_length]
    timestamps = np.arange(len(energy), dtype=np.float32) * hop_duration
    return energy, timestamps


def find_active_segment_start(y: np.ndarray, sr: int, segment_duration: float = LYRICS_DEFAULT_SEGMENT_DURATION) -> float:
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
    required_frames = max(required_frames, 1)
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
    return float(start_times[best_frame_index])


def extract_segment(y: np.ndarray, sr: int, start_time: float, segment_duration: float = LYRICS_DEFAULT_SEGMENT_DURATION) -> np.ndarray:
    start_sample = int(round(start_time * sr))
    end_sample = start_sample + int(round(segment_duration * sr))
    return y[start_sample:end_sample]


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    req = Request(url, headers={'User-Agent': 'AudioMuse-Lyrics-Downloader/1.0'})
    try:
        with urlopen(req) as response, open(destination, 'wb') as out_file:
            total = int(response.getheader('Content-Length', '0') or 0)
            downloaded = 0
            chunk_size = 64 * 1024
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                out_file.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded * 100 // total
                    print(f'  downloaded {pct}% ({downloaded}/{total} bytes)', end='\r')
            if total:
                print(f'  downloaded 100% ({downloaded}/{total} bytes)')
    except HTTPError as e:
        raise RuntimeError(
            f"Failed to download model from {url}: HTTP {e.code} {e.reason}. "
            "The default Hugging Face URL may not be valid or may require authentication. "
            "Download the gguf manually and place it in the model directory."
        ) from e
    except URLError as e:
        raise RuntimeError(
            f"Failed to download model from {url}: {e.reason}. "
            "Check your network connection or download the gguf manually."
        ) from e


def ensure_llm_model_exists() -> None:
    if LYRICS_LLM_MODEL_PATH.exists():
        return
    print(f'LLM cleanup model not found at {LYRICS_LLM_MODEL_PATH}. Downloading Qwen model to local model directory...')
    download_file(LYRICS_LLM_MODEL_URL, LYRICS_LLM_MODEL_PATH)
    print(f'LLM cleanup model downloaded to {LYRICS_LLM_MODEL_PATH}')


def load_whisper_model(model_name: str = LYRICS_WHISPER_MODEL, device: str = 'cpu', num_threads: Optional[int] = None):
    set_cpu_threading(num_threads)
    if whisper is None:
        raise RuntimeError('The whisper package is not installed. Install openai-whisper.')

    model_path = model_name
    if Path(str(model_name)).exists():
        model_path_obj = Path(str(model_name))
        if model_path_obj.suffix.lower() == '.gguf':
            raise RuntimeError(
                'openai-whisper does not support local .gguf models. '
                'Use the alias "small" or provide a supported Whisper checkpoint file.'
            )
        model_path = str(model_path_obj)

    return whisper.load_model(model_path, device=device, download_root=str(LYRICS_MODEL_DIR))


def transcribe_audio_segment(audio: np.ndarray, sr: int, model, language: Optional[str] = None) -> Dict[str, object]:
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if len(audio) == 0:
        return {'text': '', 'language': language, 'duration': 0.0}
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
        temp_path = tmp_file.name
    try:
        sf.write(temp_path, audio, sr, subtype='PCM_16')
        result = model.transcribe(temp_path, language=language, fp16=False)
        return {'text': result.get('text', '').strip(), 'language': result.get('language', language), 'duration': len(audio) / sr}
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
        raise RuntimeError('The llama-cpp-python package is not installed. Install llama-cpp-python.')
    if not os.path.exists(model_path):
        raise RuntimeError(f'LLM model file not found: {model_path}')
    return Llama(model_path=model_path, n_threads=num_threads or 1, n_gpu_layers=0, verbose=False)


def detect_language(text: str) -> Tuple[str, float]:
    if not text or not text.strip():
        return 'en', 0.0
    if detect_langs is None:
        raise RuntimeError('langdetect is required to detect text language.')
    try:
        candidates = detect_langs(text.replace('\n', ' '))
    except Exception:
        return 'en', 0.0
    if not candidates:
        return 'en', 0.0
    best = candidates[0]
    return best.lang, float(best.prob)


def get_marian_model_for_language(source_lang: str):
    source_lang = source_lang.lower()
    if source_lang == 'en':
        return None, None
    if AutoModelForSeq2SeqLM is None or AutoTokenizer is None:
        raise RuntimeError('transformers is required to load MarianMT models.')
    model_name = LYRICS_DEFAULT_MARIAN_PREFIX.format(source_lang)
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    except Exception:
        return None, None
    return tokenizer, model


def ensure_topic_embedding_model_cached(model_name: str = LYRICS_DEFAULT_TOPIC_EMBEDDING_MODEL) -> Path:
    if AutoTokenizer is None or AutoModel is None:
        raise RuntimeError('transformers is required to cache the topic embedding model.')
    local_path = LYRICS_DEFAULT_TOPIC_EMBEDDING_CACHE_DIR
    if local_path.exists():
        return local_path
    print(f'Caching topic embedding model {model_name} to {local_path}')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    local_path.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(local_path)
    model.save_pretrained(local_path)
    return local_path


def split_text_into_word_chunks(text: str, max_words: int = 50) -> List[str]:
    text = text.strip()
    if not text:
        return []
    words = text.split()
    if len(words) <= max_words:
        return [text]
    return [' '.join(words[i:i + max_words]) for i in range(0, len(words), max_words)]


def translate_text_to_english(text: str, source_lang: str) -> str:
    if not text or source_lang.lower() == 'en':
        return text
    tokenizer, model = get_marian_model_for_language(source_lang)
    if tokenizer is None or model is None:
        return text
    chunks = split_text_into_word_chunks(text, max_words=50)
    translations: List[str] = []
    for chunk in chunks:
        inputs = tokenizer(chunk, truncation=True, padding=True, return_tensors='pt', max_length=512)
        outputs = model.generate(**inputs, max_length=512)
        translated = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        translations.append(translated[0].strip() if translated else chunk)
    return ' '.join(translations)


def normalize_cleaned_text(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r'\s+([?.!,;:])', r'\1', cleaned)
    cleaned = re.sub(r'\s*\n\s*', '\n', cleaned)
    cleaned = re.sub(r'(^|[.!?]\s+)(i)\b', lambda m: m.group(1) + 'I', cleaned)
    cleaned = re.sub(r'\s{2,}', ' ', cleaned)
    return cleaned


def clean_text_with_llama(text: str, model, max_tokens: int = 256, temperature: float = 0.2) -> str:
    if not text or not text.strip():
        return ''
    chunks = split_text_into_word_chunks(text, max_words=50)
    cleaned_chunks: List[str] = []
    for chunk in chunks:
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
                stop=["\n\n", "\nDo not invent new lyrics", "\nOutput only", "\nDo not include any metadata"],
            )
        except ValueError as exc:
            err = str(exc)
            if 'Requested tokens' in err or 'exceed context window' in err or 'exceeds context window' in err:
                fallback_chunks = split_text_into_word_chunks(chunk, max_words=20)
                try:
                    for sub_chunk in fallback_chunks:
                        prompt = (
                            "Clean up this song transcription.\n"
                            "Fix obvious transcription mistakes only.\n"
                            "Do not invent lyrics or change the meaning.\n"
                            "Keep the original phrasing as much as possible.\n\n"
                            "Input:\n"
                            f"{sub_chunk}\n\n"
                            "Output:\n"
                        )
                        response = model.create_completion(
                            prompt,
                            max_tokens=80,
                            temperature=temperature,
                            top_p=0.75,
                            presence_penalty=0.5,
                            repeat_penalty=1.15,
                            echo=False,
                            stop=["\n\n", "\nDo not invent lyrics", "\nOutput:", "\nDo not include any metadata"],
                        )
                        if isinstance(response, dict):
                            choices = response.get('choices', [])
                            if choices:
                                text = choices[0].get('text', '')
                            else:
                                text = str(response)
                        else:
                            text = str(response)
                        cleaned_chunk = normalize_cleaned_text(text)
                        cleaned_chunks.append(cleaned_chunk)
                    continue
                except Exception:
                    return ''
            return ''
        except Exception:
            return ''
        if isinstance(response, dict):
            choices = response.get('choices', [])
            if choices:
                text = choices[0].get('text', '')
            else:
                text = str(response)
        else:
            text = str(response)
        cleaned_chunk = normalize_cleaned_text(text)
        cleaned_chunks.append(cleaned_chunk)
    return '\n\n'.join(cleaned_chunks).strip()


def load_roberta_embedding_model(model_name: str = LYRICS_DEFAULT_TOPIC_EMBEDDING_MODEL):
    if AutoTokenizer is None or AutoModel is None:
        raise RuntimeError('transformers is required for embedding computation.')
    local_model_dir = ensure_topic_embedding_model_cached(model_name)
    tokenizer = AutoTokenizer.from_pretrained(str(local_model_dir))
    model = AutoModel.from_pretrained(str(local_model_dir))
    return tokenizer, model


def embed_text_with_roberta(text: str, tokenizer, model) -> Optional[np.ndarray]:
    if not text or not text.strip():
        return None
    encoded = tokenizer(text, truncation=True, padding='max_length', max_length=128, return_tensors='pt')
    with np.errstate(all='ignore'):
        outputs = model(**encoded)
    last_hidden = outputs.last_hidden_state
    attention_mask = encoded['attention_mask'].unsqueeze(-1).expand(last_hidden.size()).float()
    summed = (last_hidden * attention_mask).sum(1)
    counts = attention_mask.sum(1).clamp(min=1e-9)
    pooled = (summed / counts).squeeze(0)
    vector = pooled.cpu().detach().numpy()
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm
    return vector


def compute_topic_label_embeddings(labels: List[str], tokenizer, model) -> np.ndarray:
    embeddings: List[np.ndarray] = []
    for label in labels:
        embedding = embed_text_with_roberta(label, tokenizer, model)
        if embedding is not None:
            embeddings.append(embedding)
    return np.stack(embeddings) if embeddings else np.zeros((0, 0), dtype=np.float32)


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


def load_jsonl_texts(path: Path) -> Dict[str, str]:
    texts = {}
    if not path.exists():
        return texts
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except Exception:
                continue
            song_id = str(item.get('id', '')).strip()
            if song_id and song_id not in texts:
                texts[song_id] = str(item.get('text', ''))
    return texts


def load_jsonl_items(path: Path) -> List[Dict]:
    items = []
    if not path.exists():
        return items
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except Exception:
                continue
            if isinstance(item, dict):
                items.append(item)
    return items


def append_jsonl_texts(path: Path, items: List[Tuple[str, str]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = 'a' if path.exists() else 'w'
    with path.open(mode, encoding='utf-8') as f:
        for song_id, text in items:
            f.write(json.dumps({'id': song_id, 'text': text}, ensure_ascii=False) + '\n')


def append_jsonl_items(path: Path, items: List[Dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = 'a' if path.exists() else 'w'
    with path.open(mode, encoding='utf-8') as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def write_jsonl_items(path: Path, items: List[Dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def build_query_results(axis_items: List[Dict], queries: List[Dict], top_k: int = 5) -> List[Dict]:
    query_results = []
    for query in queries:
        scored = []
        for item in axis_items:
            score = 0.0
            for axis_labels in item.get('axes', {}).values():
                for label_data in axis_labels:
                    label = label_data.get('label')
                    if label in query['weights']:
                        score += float(label_data.get('score', 0.0)) * float(query['weights'][label])
            scored.append({'file_path': item.get('file_path'), 'text': item.get('text', ''), 'score': score})
        scored.sort(key=lambda x: x['score'], reverse=True)
        query_results.append({'query_name': query['name'], 'query_weights': query['weights'], 'top_songs': scored[:top_k]})
    return query_results


def normalize_raw_text(text: str) -> str:
    return re.sub(r'\s+', ' ', text.replace('\r', '').strip()).strip()


def transcribe_file(audio_path: Path, model) -> str:
    y, sr = load_audio(str(audio_path), sr=LYRICS_DEFAULT_SAMPLE_RATE)
    transcription = transcribe_audio_segment(y, sr, model)
    return transcription['text']


def main():
    start_time = time.perf_counter()
    songs_dir = LYRICS_SONGS_DIR
    if not songs_dir.exists():
        raise SystemExit(f'Songs directory not found: {songs_dir}')

    cache_dir = LYRICS_MODEL_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)

    corpus_path = cache_dir / 'corpus.jsonl'
    cleaned_path = cache_dir / 'corpus_cleaned.jsonl'
    axis_path = cache_dir / 'axis.jsonl'

    existing_transcripts = load_jsonl_texts(corpus_path)
    existing_cleaned = load_jsonl_texts(cleaned_path)

    print(f'Using songs directory: {songs_dir}')
    print(f'Using cache directory: {cache_dir}')
    if existing_transcripts:
        print(f'Loaded {len(existing_transcripts)} existing transcripts')
    if existing_cleaned:
        print(f'Loaded {len(existing_cleaned)} existing cleaned entries')

    audio_files = find_audio_files(songs_dir)
    if not audio_files:
        raise SystemExit('No audio files found in the songs directory.')
    if len(audio_files) > LYRICS_MAX_SONGS_TO_ANALYZE:
        audio_files = audio_files[:LYRICS_MAX_SONGS_TO_ANALYZE]
        print(f'Limiting analysis to first {LYRICS_MAX_SONGS_TO_ANALYZE} songs')

    LYRICS_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    whisper_model = load_whisper_model(model_name=LYRICS_WHISPER_MODEL, device='cpu', num_threads=None)
    ensure_llm_model_exists()
    llama_model = load_llama_cpp_model(str(LYRICS_LLM_MODEL_PATH), num_threads=1)

    tokenizer, model = load_roberta_embedding_model()
    axis_label_map, axis_embeddings = compute_axis_label_embeddings(MUSIC_ANALYSIS_AXES, tokenizer, model)

    existing_axis_items = load_jsonl_items(axis_path) if axis_path.exists() else []
    axis_existing_map = {item.get('file_path'): item for item in existing_axis_items}
    axis_output_items = list(existing_axis_items)

    total_songs = len(audio_files)
    song_times: List[float] = []

    for index, audio_path in enumerate(audio_files, start=1):
        song_start = time.perf_counter()
        song_id = audio_path.name
        transcribe_seconds = 0.0
        translate_seconds = 0.0
        clean_seconds = 0.0
        embed_seconds = 0.0
        print(f'[{index}/{total_songs}] Processing {song_id}...')

        raw_text = existing_transcripts.get(song_id)
        if raw_text is None:
            print('  Transcribing...')
            step_start = time.perf_counter()
            raw_text = transcribe_file(audio_path, whisper_model)
            transcribe_seconds = time.perf_counter() - step_start
            raw_text = normalize_raw_text(raw_text)
            append_jsonl_texts(corpus_path, [(song_id, raw_text)])
            existing_transcripts[song_id] = raw_text
            print(f'  Appended transcript for {song_id}')
        else:
            print('  Transcript already available')

        cleaned_text = existing_cleaned.get(song_id)
        if cleaned_text is None:
            word_count = len(raw_text.split())
            if word_count < 50:
                print(f'  Skipping cleanup for {song_id}: only {word_count} words (<50)')
                cleaned_text = ''
            else:
                print('  Cleaning transcript...')
                lang, confidence = detect_language(raw_text)
                if lang != 'en' and confidence >= 0.7:
                    translate_start = time.perf_counter()
                    translated_text = translate_text_to_english(raw_text, source_lang=lang)
                    translate_seconds = time.perf_counter() - translate_start
                else:
                    translated_text = raw_text
                clean_start = time.perf_counter()
                cleaned_text = clean_text_with_llama(translated_text, llama_model)
                clean_seconds = time.perf_counter() - clean_start
                cleaned_text = normalize_cleaned_text(cleaned_text) if cleaned_text else translated_text
            append_jsonl_texts(cleaned_path, [(song_id, cleaned_text)])
            existing_cleaned[song_id] = cleaned_text
            print(f'  Appended cleaned transcript for {song_id} (length={len(cleaned_text.split())} words)')
        else:
            print('  Cleaned transcript already available')

        if not existing_cleaned[song_id]:
            print('  Skipping axis analysis because cleaned text is empty')
        elif song_id in axis_existing_map:
            print('  Axis analysis already available')
        else:
            print('  Performing axis analysis...')
            embed_start = time.perf_counter()
            axis_item = {
                'file_path': song_id,
                'text': existing_cleaned[song_id],
                'axes': score_text_against_axes(existing_cleaned[song_id], tokenizer, model, axis_label_map, axis_embeddings, temperature=0.1),
            }
            embed_seconds = time.perf_counter() - embed_start
            axis_output_items.append(axis_item)
            axis_existing_map[song_id] = axis_item
            append_jsonl_items(axis_path, [axis_item])
            print(f'  Axis analysis complete and appended to {axis_path}')

        song_seconds = time.perf_counter() - song_start
        song_times.append(song_seconds)
        print(f'  Whisper transcribing: {transcribe_seconds:.2f} sec')
        print(f'  Helsinki translating: {translate_seconds:.2f} sec')
        print(f'  LLM cleaning: {clean_seconds:.2f} sec')
        print(f'  e5-base-v2 embedding: {embed_seconds:.2f} sec')
        print(f'  Tot: {song_seconds:.2f} sec')

    print(f'Wrote axis scores to {axis_path}')

    total_seconds = time.perf_counter() - start_time
    average_seconds = total_seconds / total_songs if total_songs else 0.0
    print(f'Analyzed {total_songs} song(s) in {total_seconds:.1f} seconds, average {average_seconds:.2f} seconds per song.')


if __name__ == '__main__':
    import json
    main()
