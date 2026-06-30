import os
import tempfile

TASK_STATUS_PENDING = 'PENDING'
TASK_STATUS_STARTED = 'STARTED'
TASK_STATUS_PROGRESS = 'PROGRESS'
TASK_STATUS_SUCCESS = 'SUCCESS'
TASK_STATUS_FAILURE = 'FAILURE'
TASK_STATUS_REVOKED = 'REVOKED'

MEDIASERVER_TYPE = os.environ.get("MEDIASERVER_TYPE", "jellyfin").lower()


JELLYFIN_URL = os.environ.get("JELLYFIN_URL", "")
JELLYFIN_USER_ID = os.environ.get("JELLYFIN_USER_ID", "")
JELLYFIN_TOKEN = os.environ.get("JELLYFIN_TOKEN", "")

EMBY_URL = os.environ.get("EMBY_URL", "")
EMBY_USER_ID = os.environ.get("EMBY_USER_ID", "")
EMBY_TOKEN = os.environ.get("EMBY_TOKEN", "")


MUSIC_LIBRARIES = os.environ.get("MUSIC_LIBRARIES", "")
PROBE_TOP_PLAYED_LIMIT = int(os.environ.get("PROBE_TOP_PLAYED_LIMIT", "1"))
MIGRATION_UNMATCHED_ALBUMS_PAYLOAD_LIMIT = max(
    1, int(os.environ.get("MIGRATION_UNMATCHED_ALBUMS_PAYLOAD_LIMIT", "200"))
)
MIGRATION_MAX_COLLISION_DETAILS = int(os.environ.get("MIGRATION_MAX_COLLISION_DETAILS", "1000"))
TEMP_DIR = os.environ.get("TEMP_DIR", "/app/temp_audio")


def jellyfin_auth_header(token):
    return {"Authorization": f'MediaBrowser Token="{token}"'} if token else {}


def _compute_headers():
    if MEDIASERVER_TYPE == "jellyfin":
        return jellyfin_auth_header(JELLYFIN_TOKEN)
    if MEDIASERVER_TYPE == "emby":
        return {"X-Emby-Token": EMBY_TOKEN}
    return {}


HEADERS = _compute_headers()

NAVIDROME_URL = os.environ.get("NAVIDROME_URL", "")
NAVIDROME_USER = os.environ.get("NAVIDROME_USER", "")
NAVIDROME_PASSWORD = os.environ.get("NAVIDROME_PASSWORD", "")

LYRION_URL = os.environ.get("LYRION_URL", "")

MEDIASERVER_FIELDS_BY_TYPE = {
    'jellyfin': ['JELLYFIN_URL', 'JELLYFIN_USER_ID', 'JELLYFIN_TOKEN'],
    'navidrome': ['NAVIDROME_URL', 'NAVIDROME_USER', 'NAVIDROME_PASSWORD'],
    'lyrion': ['LYRION_URL'],
    'emby': ['EMBY_URL', 'EMBY_USER_ID', 'EMBY_TOKEN'],
}

MEDIASERVER_OBSOLETE_FIELDS_BY_TYPE = {
    media_type: [
        field
        for other_type, fields in MEDIASERVER_FIELDS_BY_TYPE.items()
        if other_type != media_type
        for field in fields
    ]
    for media_type in MEDIASERVER_FIELDS_BY_TYPE
}

SETUP_BOOTSTRAP_EXCLUDED_KEYS = {
    'DATABASE_URL',
    'POSTGRES_USER',
    'POSTGRES_PASSWORD',
    'POSTGRES_HOST',
    'POSTGRES_PORT',
    'POSTGRES_DB',
    'REDIS_URL',
    'MEDIASERVER_FIELDS_BY_TYPE',
    'MEDIASERVER_OBSOLETE_FIELDS_BY_TYPE',
    'APP_VERSION',
    'AUDIOMUSE_USER',
    'AUDIOMUSE_PASSWORD',
    'LYRICS_INSTRUMENTAL_EMBEDDING',
    'LYRICS_INSTRUMENTAL_AXIS_FILL',
}

APP_VERSION = "v2.3.3"
MAX_DISTANCE = float(os.environ.get("MAX_DISTANCE", "0.5"))
MAX_SONGS_PER_CLUSTER = int(os.environ.get("MAX_SONGS_PER_CLUSTER", "0"))
MAX_SONGS_PER_ARTIST = int(os.getenv("MAX_SONGS_PER_ARTIST", "3"))
SIMILARITY_ELIMINATE_DUPLICATES_DEFAULT = (
    os.environ.get("SIMILARITY_ELIMINATE_DUPLICATES_DEFAULT", "True").lower() == 'true'
)
SIMILARITY_RADIUS_DEFAULT = os.environ.get("SIMILARITY_RADIUS_DEFAULT", "True").lower() == 'true'
RADIUS_INSTRUMENTATION = os.environ.get("RADIUS_INSTRUMENTATION", "False").lower() == 'true'
NUM_RECENT_ALBUMS = int(os.getenv("NUM_RECENT_ALBUMS", "0"))
TOP_N_PLAYLISTS = int(os.environ.get("TOP_N_PLAYLISTS", "8"))
MIN_PLAYLIST_SIZE_FOR_TOP_N = int(os.environ.get("MIN_PLAYLIST_SIZE_FOR_TOP_N", "20"))

CLUSTER_ALGORITHM = os.environ.get("CLUSTER_ALGORITHM", "kmeans")
AI_MODEL_PROVIDER = os.environ.get("AI_MODEL_PROVIDER", "NONE").upper()
ENABLE_CLUSTERING_EMBEDDINGS = (
    os.environ.get("ENABLE_CLUSTERING_EMBEDDINGS", "True").lower() == "true"
)

USE_GPU_CLUSTERING = os.environ.get("USE_GPU_CLUSTERING", "False").lower() == "true"

CLUSTERING_CLEANING = os.environ.get("CLUSTERING_CLEANING", "True").lower() == "true"

DBSCAN_EPS_MIN = float(os.getenv("DBSCAN_EPS_MIN", "0.1"))
DBSCAN_EPS_MAX = float(os.getenv("DBSCAN_EPS_MAX", "0.5"))
DBSCAN_MIN_SAMPLES_MIN = int(os.getenv("DBSCAN_MIN_SAMPLES_MIN", "5"))
DBSCAN_MIN_SAMPLES_MAX = int(os.getenv("DBSCAN_MIN_SAMPLES_MAX", "20"))


NUM_CLUSTERS_MIN = int(os.getenv("NUM_CLUSTERS_MIN", "40"))
NUM_CLUSTERS_MAX = int(os.getenv("NUM_CLUSTERS_MAX", "100"))
USE_MINIBATCH_KMEANS = os.environ.get("USE_MINIBATCH_KMEANS", "False").lower() == "true"
MINIBATCH_KMEANS_PROCESSING_BATCH_SIZE = int(
    os.getenv("MINIBATCH_KMEANS_PROCESSING_BATCH_SIZE", "1000")
)

GMM_N_COMPONENTS_MIN = int(os.getenv("GMM_N_COMPONENTS_MIN", "40"))
GMM_N_COMPONENTS_MAX = int(os.getenv("GMM_N_COMPONENTS_MAX", "100"))
GMM_COVARIANCE_TYPE = os.environ.get("GMM_COVARIANCE_TYPE", "full")

SPECTRAL_N_CLUSTERS_MIN = int(os.getenv("SPECTRAL_N_CLUSTERS_MIN", "40"))
SPECTRAL_N_CLUSTERS_MAX = int(os.getenv("SPECTRAL_N_CLUSTERS_MAX", "100"))
SPECTRAL_N_NEIGHBORS = int(os.getenv("SPECTRAL_N_NEIGHBORS", "20"))

PCA_COMPONENTS_MIN = int(os.getenv("PCA_COMPONENTS_MIN", "0"))
PCA_COMPONENTS_MAX = int(os.getenv("PCA_COMPONENTS_MAX", "199"))

CLUSTERING_RUNS = int(os.environ.get("CLUSTERING_RUNS", "1000"))
MAX_QUEUED_ANALYSIS_JOBS = int(os.environ.get("MAX_QUEUED_ANALYSIS_JOBS", "25"))

ITERATIONS_PER_BATCH_JOB = int(os.environ.get("ITERATIONS_PER_BATCH_JOB", "20"))
MAX_CONCURRENT_BATCH_JOBS = int(os.environ.get("MAX_CONCURRENT_BATCH_JOBS", "10"))
DB_FETCH_CHUNK_SIZE = int(os.environ.get("DB_FETCH_CHUNK_SIZE", "1000"))


CLUSTERING_BATCH_TIMEOUT_MINUTES = int(os.environ.get("CLUSTERING_BATCH_TIMEOUT_MINUTES", "60"))
CLUSTERING_MAX_FAILED_BATCHES = int(os.environ.get("CLUSTERING_MAX_FAILED_BATCHES", "10"))
CLUSTERING_BATCH_CHECK_INTERVAL_SECONDS = int(
    os.environ.get("CLUSTERING_BATCH_CHECK_INTERVAL_SECONDS", "30")
)

REBUILD_INDEX_BATCH_SIZE = int(os.environ.get("REBUILD_INDEX_BATCH_SIZE", "1000"))
AUDIO_LOAD_TIMEOUT = int(os.getenv("AUDIO_LOAD_TIMEOUT", "600"))
ANALYSIS_MONITOR_DB_INTERVAL = int(os.environ.get("ANALYSIS_MONITOR_DB_INTERVAL", "10"))

TOP_N_ELITES = int(os.environ.get("CLUSTERING_TOP_N_ELITES", "10"))
EXPLOITATION_START_FRACTION = float(os.environ.get("CLUSTERING_EXPLOITATION_START_FRACTION", "0.2"))
EXPLOITATION_PROBABILITY_CONFIG = float(
    os.environ.get("CLUSTERING_EXPLOITATION_PROBABILITY", "0.7")
)
MUTATION_INT_ABS_DELTA = int(os.environ.get("CLUSTERING_MUTATION_INT_ABS_DELTA", "3"))
MUTATION_FLOAT_ABS_DELTA = float(os.environ.get("CLUSTERING_MUTATION_FLOAT_ABS_DELTA", "0.05"))
MUTATION_KMEANS_COORD_FRACTION = float(
    os.environ.get("CLUSTERING_MUTATION_KMEANS_COORD_FRACTION", "0.05")
)

SCORE_WEIGHT_DIVERSITY = float(os.environ.get("SCORE_WEIGHT_DIVERSITY", "2.0"))
SCORE_WEIGHT_PURITY = float(os.environ.get("SCORE_WEIGHT_PURITY", "1.0"))
SCORE_WEIGHT_OTHER_FEATURE_DIVERSITY = float(
    os.environ.get("SCORE_WEIGHT_OTHER_FEATURE_DIVERSITY", "0.0")
)
SCORE_WEIGHT_OTHER_FEATURE_PURITY = float(
    os.environ.get("SCORE_WEIGHT_OTHER_FEATURE_PURITY", "0.0")
)
SCORE_WEIGHT_SILHOUETTE = float(os.environ.get("SCORE_WEIGHT_SILHOUETTE", "0.0"))
SCORE_WEIGHT_DAVIES_BOULDIN = float(os.environ.get("SCORE_WEIGHT_DAVIES_BOULDIN", "0.0"))
SCORE_WEIGHT_CALINSKI_HARABASZ = float(os.environ.get("SCORE_WEIGHT_CALINSKI_HARABASZ", "0.0"))
TOP_K_MOODS_FOR_PURITY_CALCULATION = int(os.environ.get("TOP_K_MOODS_FOR_PURITY_CALCULATION", "3"))

LN_MOOD_DIVERSITY_STATS = {
    "min": float(os.environ.get("LN_MOOD_DIVERSITY_MIN", "-0.1863")),
    "max": float(os.environ.get("LN_MOOD_DIVERSITY_MAX", "1.5518")),
    "mean": float(os.environ.get("LN_MOOD_DIVERSITY_MEAN", "0.9995")),
    "sd": float(os.environ.get("LN_MOOD_DIVERSITY_SD", "0.3541")),
}

LN_MOOD_DIVERSITY_EMBEDING_STATS = {
    "min": float(os.environ.get("LN_MOOD_DIVERSITY_EMBEDDING_MIN", "-0.174")),
    "max": float(os.environ.get("LN_MOOD_DIVERSITY_EMBEDDING_MAX", "0.570")),
    "mean": float(os.environ.get("LN_MOOD_DIVERSITY_EMBEDDING_MEAN", "-0.101")),
    "sd": float(os.environ.get("LN_MOOD_DIVERSITY_EMBEDDING_SD", "0.245")),
}

LN_MOOD_PURITY_STATS = {
    "min": float(os.environ.get("LN_MOOD_PURITY_MIN", "0.6981")),
    "max": float(os.environ.get("LN_MOOD_PURITY_MAX", "7.2848")),
    "mean": float(os.environ.get("LN_MOOD_PURITY_MEAN", "5.8679")),
    "sd": float(os.environ.get("LN_MOOD_PURITY_SD", "1.1557")),
}

LN_MOOD_PURITY_EMBEDING_STATS = {
    "min": float(os.environ.get("LN_MOOD_PURITY_EMBEDDING_MIN", "-0.494")),
    "max": float(os.environ.get("LN_MOOD_PURITY_EMBEDDING_MAX", "2.583")),
    "mean": float(os.environ.get("LN_MOOD_PURITY_EMBEDDING_MEAN", "0.673")),
    "sd": float(os.environ.get("LN_MOOD_PURITY_EMBEDDING_SD", "1.063")),
}

LN_OTHER_FEATURES_DIVERSITY_STATS = {
    "min": float(os.environ.get("LN_OTHER_FEAT_DIV_MIN", "-0.19")),
    "max": float(os.environ.get("LN_OTHER_FEAT_DIV_MAX", "2.06")),
    "mean": float(os.environ.get("LN_OTHER_FEAT_DIV_MEAN", "1.5")),
    "sd": float(os.environ.get("LN_OTHER_FEAT_DIV_SD", "0.46")),
}

LN_OTHER_FEATURES_PURITY_STATS = {
    "min": float(os.environ.get("LN_OTHER_FEAT_PUR_MIN", "8.67")),
    "max": float(os.environ.get("LN_OTHER_FEAT_PUR_MAX", "8.95")),
    "mean": float(os.environ.get("LN_OTHER_FEAT_PUR_MEAN", "8.84")),
    "sd": float(os.environ.get("LN_OTHER_FEAT_PUR_SD", "0.07")),
}

OTHER_FEATURE_PREDOMINANCE_THRESHOLD_FOR_PURITY = float(
    os.environ.get("OTHER_FEATURE_PREDOMINANCE_THRESHOLD_FOR_PURITY", "0.3")
)

OLLAMA_SERVER_URL = os.environ.get("OLLAMA_SERVER_URL", "http://192.168.3.211:11434/api/generate")
OLLAMA_MODEL_NAME = os.environ.get("OLLAMA_MODEL_NAME", "qwen3.5:9b")

MAX_SONGS_IN_AI_PROMPT = int(os.environ.get("MAX_SONGS_IN_AI_PROMPT", "25"))

OPENAI_SERVER_URL = os.environ.get(
    "OPENAI_SERVER_URL",
    os.environ.get("OLLAMA_SERVER_URL", "http://192.168.3.211:11434/api/generate"),
)
OPENAI_MODEL_NAME = os.environ.get(
    "OPENAI_MODEL_NAME", os.environ.get("OLLAMA_MODEL_NAME", "llama3.1:8b")
)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "no-key-needed")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL_NAME = os.environ.get("GEMINI_MODEL_NAME", "gemini-2.5-pro")

MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY", "")
MISTRAL_MODEL_NAME = os.environ.get("MISTRAL_MODEL_NAME", "ministral-3b-latest")

AI_REQUEST_TIMEOUT_SECONDS = int(os.environ.get("AI_REQUEST_TIMEOUT_SECONDS", "300"))
REDIS_URL = os.environ.get('REDIS_URL', 'redis://localhost:6379/0')

RQ_MAX_JOBS = int(os.getenv('RQ_MAX_JOBS', '50'))
RQ_MAX_JOBS_HIGH = int(os.getenv('RQ_MAX_JOBS_HIGH', '100'))
RQ_LOGGING_LEVEL = os.getenv('RQ_LOGGING_LEVEL', 'INFO').upper()

POSTGRES_USER = os.environ.get("POSTGRES_USER", "audiomuse")
POSTGRES_PASSWORD = os.environ.get("POSTGRES_PASSWORD", "audiomusepassword")
POSTGRES_HOST = os.environ.get("POSTGRES_HOST", "postgres-service.playlist")
POSTGRES_PORT = os.environ.get("POSTGRES_PORT", "5432")
POSTGRES_DB = os.environ.get("POSTGRES_DB", "audiomusedb")

from urllib.parse import quote

_pg_user_esc = quote(POSTGRES_USER, safe='')
_pg_pass_esc = quote(POSTGRES_PASSWORD, safe='')

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    f"postgresql://{_pg_user_esc}:{_pg_pass_esc}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}",
)

DATABASE_TYPE = os.environ.get("DATABASE_TYPE", "postgres").lower()
QUEUE_TYPE = os.environ.get("QUEUE_TYPE", "redis").lower()
APP_DATA_DIR = os.environ.get("APP_DATA_DIR", "")
AUDIOMUSE_PLATFORM = os.environ.get("AUDIOMUSE_PLATFORM", "").lower()
AUDIOMUSE_CONTROL_SOCKET = os.environ.get("AUDIOMUSE_CONTROL_SOCKET", "")
AUDIOMUSE_CONTROL_HOST = os.environ.get("AUDIOMUSE_CONTROL_HOST", "")
AUDIOMUSE_CONTROL_PORT = os.environ.get("AUDIOMUSE_CONTROL_PORT", "")

AI_CHAT_DB_USER_NAME = os.environ.get("AI_CHAT_DB_USER_NAME", "ai_user")
AI_CHAT_DB_USER_PASSWORD = os.environ.get(
    "AI_CHAT_DB_USER_PASSWORD", "ChangeThisSecurePassword123!"
)

MOOD_LABELS = [
    'rock',
    'pop',
    'alternative',
    'indie',
    'electronic',
    'female vocalists',
    'dance',
    '00s',
    'alternative rock',
    'jazz',
    'beautiful',
    'metal',
    'chillout',
    'male vocalists',
    'classic rock',
    'soul',
    'indie rock',
    'Mellow',
    'electronica',
    '80s',
    'folk',
    '90s',
    'chill',
    'instrumental',
    'punk',
    'oldies',
    'blues',
    'hard rock',
    'ambient',
    'acoustic',
    'experimental',
    'female vocalist',
    'guitar',
    'Hip-Hop',
    '70s',
    'party',
    'country',
    'easy listening',
    'sexy',
    'catchy',
    'funk',
    'electro',
    'heavy metal',
    'Progressive rock',
    '60s',
    'rnb',
    'indie pop',
    'sad',
    'House',
    'happy',
]

TOP_N_MOODS = int(os.environ.get("TOP_N_MOODS", "5"))
EMBEDDING_MODEL_PATH = os.environ.get("EMBEDDING_MODEL_PATH", "/app/model/musicnn_embedding.onnx")
PREDICTION_MODEL_PATH = os.environ.get(
    "PREDICTION_MODEL_PATH", "/app/model/musicnn_prediction.onnx"
)
EMBEDDING_DIMENSION = 200

CLAP_ENABLED = os.environ.get("CLAP_ENABLED", "true").lower() == "true"
LYRICS_ENABLED = os.environ.get("LYRICS_ENABLED", "true").lower() == "true"
LYRICS_API_ENABLE = os.environ.get("LYRICS_API_ENABLE", "true").lower() == "true"
LYRICS_ASR_ENABLE = os.environ.get("LYRICS_ASR_ENABLE", "true").lower() == "true"
LYRICS_MUSICNN_SKIP = os.environ.get("LYRICS_MUSICNN_SKIP", "true").lower() == "true"
MUSICSERVER_LYRICS_TIMEOUT = float(os.environ.get("MUSICSERVER_LYRICS_TIMEOUT", "2.5"))
LYRICS_API_1_URL_TEMPLATE = os.environ.get("LYRICS_API_1_URL_TEMPLATE", "")
LYRICS_API_1_ARTIST_PARAM = os.environ.get("LYRICS_API_1_ARTIST_PARAM", "artist_name")
LYRICS_API_1_TITLE_PARAM = os.environ.get("LYRICS_API_1_TITLE_PARAM", "track_name")
LYRICS_API_1_LYRICS_FIELD = os.environ.get("LYRICS_API_1_LYRICS_FIELD", "plainLyrics")
LYRICS_API_1_APIKEY_PARAM = os.environ.get("LYRICS_API_1_APIKEY_PARAM", "")
LYRICS_API_1_APIKEY_VALUE = os.environ.get("LYRICS_API_1_APIKEY_VALUE", "")
LYRICS_API_1_TIMEOUT = float(os.environ.get("LYRICS_API_1_TIMEOUT", "5.0"))
LYRICS_API_2_URL_TEMPLATE = os.environ.get("LYRICS_API_2_URL_TEMPLATE", "")
LYRICS_API_2_ARTIST_PARAM = os.environ.get("LYRICS_API_2_ARTIST_PARAM", "artist")
LYRICS_API_2_TITLE_PARAM = os.environ.get("LYRICS_API_2_TITLE_PARAM", "title")
LYRICS_API_2_LYRICS_FIELD = os.environ.get("LYRICS_API_2_LYRICS_FIELD", "lyrics")
LYRICS_API_2_APIKEY_PARAM = os.environ.get("LYRICS_API_2_APIKEY_PARAM", "")
LYRICS_API_2_APIKEY_VALUE = os.environ.get("LYRICS_API_2_APIKEY_VALUE", "")
LYRICS_API_2_TIMEOUT = float(os.environ.get("LYRICS_API_2_TIMEOUT", "5.0"))
LYRICS_ASR_BEAM_SIZE = int(os.environ.get("LYRICS_ASR_BEAM_SIZE", "5"))
LYRICS_ASR_MIN_AVG_LOGPROB = float(os.environ.get("LYRICS_ASR_MIN_AVG_LOGPROB", "-1.0"))
LYRICS_ASR_NON_ENGLISH_MIN_LOGPROB = float(
    os.environ.get("LYRICS_ASR_NON_ENGLISH_MIN_LOGPROB", "-0.85")
)
LYRICS_WHISPER_MODEL_DIR = os.environ.get(
    "LYRICS_WHISPER_MODEL_DIR",
    os.path.join(os.environ.get("LYRICS_MODEL_DIR", "/app/model"), "whisper-small-onnx"),
)
LYRICS_MODEL_DIR = os.environ.get("LYRICS_MODEL_DIR", "/app/model")
LYRICS_MAX_SONGS_TO_ANALYZE = 1000
LYRICS_SUPPORTED_AUDIO_EXTENSIONS = {
    '.wav',
    '.mp3',
    '.m4a',
    '.flac',
    '.ogg',
    '.opus',
    '.aac',
    '.aiff',
    '.aif',
    '.mp4',
}
VAD_VOICE_RECOGNITION = int(os.environ.get("VAD_VOICE_RECOGNITION", "25"))

LYRICS_DEFAULT_SAMPLE_RATE = 16000
LYRICS_DEFAULT_SEGMENT_DURATION = 60.0
LYRICS_DEFAULT_TOPIC_EMBEDDING_MODEL = 'Alibaba-NLP/gte-multilingual-base'
LYRICS_DEFAULT_TOPIC_EMBEDDING_CACHE_DIR = os.path.join(LYRICS_MODEL_DIR, 'gte-multilingual-base')
LYRICS_EMBEDDING_DIMENSION = int(os.environ.get("LYRICS_EMBEDDING_DIMENSION", "768"))

LYRICS_MIN_CHARS_FOR_EMBEDDING = int(os.environ.get("LYRICS_MIN_CHARS_FOR_EMBEDDING", "250"))
LYRICS_TEXT_MAX_COMPRESSION_RATIO = float(
    os.environ.get("LYRICS_TEXT_MAX_COMPRESSION_RATIO", "15.0")
)
LYRICS_LANG_CONFIDENCE_MIN = float(os.environ.get("LYRICS_LANG_CONFIDENCE_MIN", "0.70"))
LYRICS_CJK_SCRIPT_MIN_RATIO = float(os.environ.get("LYRICS_CJK_SCRIPT_MIN_RATIO", "0.10"))
LYRICS_GTE_MAX_TOKENS = int(os.environ.get("LYRICS_GTE_MAX_TOKENS", "512"))

LYRICS_VAD_THRESHOLD = float(os.environ.get("LYRICS_VAD_THRESHOLD", "0.2"))
LYRICS_VAD_NEG_THRESHOLD = (
    float(os.environ["LYRICS_VAD_NEG_THRESHOLD"])
    if "LYRICS_VAD_NEG_THRESHOLD" in os.environ
    else max(0.01, LYRICS_VAD_THRESHOLD - 0.15)
)
LYRICS_VAD_RETRY_FLOOR = float(os.environ.get("LYRICS_VAD_RETRY_FLOOR", "0.15"))
LYRICS_VAD_MIN_SILENCE_MS = int(os.environ.get("LYRICS_VAD_MIN_SILENCE_MS", "1000"))
LYRICS_VAD_MIN_SPEECH_MS = int(os.environ.get("LYRICS_VAD_MIN_SPEECH_MS", "250"))
LYRICS_VAD_SPEECH_PAD_MS = int(os.environ.get("LYRICS_VAD_SPEECH_PAD_MS", "400"))

SEM_GROVE_WEIGHT_LYRICS = float(os.environ.get("SEM_GROVE_WEIGHT_LYRICS", "0.75"))
SEM_GROVE_WEIGHT_AUDIO = float(os.environ.get("SEM_GROVE_WEIGHT_AUDIO", "0.25"))

import numpy as _np

LYRICS_INSTRUMENTAL_EMBEDDING = _np.zeros(LYRICS_EMBEDDING_DIMENSION, dtype=_np.float32)
LYRICS_INSTRUMENTAL_EMBEDDING[0] = 1.0
LYRICS_INSTRUMENTAL_EMBEDDING.flags.writeable = False

LYRICS_INSTRUMENTAL_AXIS_FILL = -0.19245009

CLAP_AUDIO_MODEL_PATH = os.environ.get("CLAP_AUDIO_MODEL_PATH", "/app/model/model_epoch_36.onnx")

CLAP_AUDIO_N_MELS = int(os.environ.get("CLAP_AUDIO_N_MELS", "128"))
CLAP_AUDIO_N_FFT = int(os.environ.get("CLAP_AUDIO_N_FFT", "2048"))
CLAP_AUDIO_HOP_LENGTH = int(os.environ.get("CLAP_AUDIO_HOP_LENGTH", "480"))
CLAP_AUDIO_FMIN = int(os.environ.get("CLAP_AUDIO_FMIN", "0"))
CLAP_AUDIO_FMAX = int(os.environ.get("CLAP_AUDIO_FMAX", "14000"))
CLAP_AUDIO_MEL_TRANSPOSE = os.environ.get("CLAP_AUDIO_MEL_TRANSPOSE", "false").lower() == "true"

CLAP_TEXT_MODEL_PATH = os.environ.get("CLAP_TEXT_MODEL_PATH", "/app/model/clap_text_model.onnx")
CLAP_EMBEDDING_DIMENSION = 512
CLAP_PYTHON_MULTITHREADS = os.environ.get("CLAP_PYTHON_MULTITHREADS", "False").lower() == "true"

PER_SONG_MODEL_RELOAD = os.environ.get("PER_SONG_MODEL_RELOAD", "true").lower() == "true"

CLAP_CATEGORY_WEIGHTS_DEFAULT = {
    "Genre_Style": 1.0,
    "Instrumentation_Vocal": 1.0,
    "Emotion_Mood": 1.0,
    "Voice_Type": 1.0,
}
import json

CLAP_CATEGORY_WEIGHTS = json.loads(
    os.environ.get("CLAP_CATEGORY_WEIGHTS", json.dumps(CLAP_CATEGORY_WEIGHTS_DEFAULT))
)

CLAP_TOP_QUERIES_COUNT = int(os.environ.get("CLAP_TOP_QUERIES_COUNT", "1000"))

CLAP_TEXT_SEARCH_WARMUP_DURATION = int(os.environ.get("CLAP_TEXT_SEARCH_WARMUP_DURATION", "300"))

LYRICS_GTE_WARMUP_DURATION = int(os.environ.get("LYRICS_GTE_WARMUP_DURATION", "300"))

INDEX_NAME = os.environ.get("IVF_INDEX_NAME", "music_library")
IVF_METRIC = os.environ.get("IVF_METRIC", "angular")
IVF_QUERY_EF = int(os.environ.get("IVF_QUERY_EF", "1024"))

IVF_STORAGE_DTYPE = os.environ.get("IVF_STORAGE_DTYPE", "i8").lower()
IVF_NLIST_MAX = int(os.environ.get("IVF_NLIST_MAX", "8192"))
IVF_TRAIN_POINTS_PER_CELL = int(os.environ.get("IVF_TRAIN_POINTS_PER_CELL", "50"))
IVF_MAX_CELL_MB = int(os.environ.get("IVF_MAX_CELL_MB", "12"))
IVF_MAX_PART_SIZE_MB = int(os.environ.get("IVF_MAX_PART_SIZE_MB", "50"))
IVF_NPROBE = int(os.environ.get("IVF_NPROBE", "1024"))
IVF_RERANK_OVERFETCH = int(os.environ.get("IVF_RERANK_OVERFETCH", "4"))
IVF_QUERY_CACHE_MB = int(os.environ.get("IVF_QUERY_CACHE_MB", "128"))
IVF_READ_BATCH_CELLS = int(os.environ.get("IVF_READ_BATCH_CELLS", "16"))
IVF_QUERY_PARALLEL_MIN_VECTORS = int(os.environ.get("IVF_QUERY_PARALLEL_MIN_VECTORS", "8192"))
IVF_GLOBAL_CACHE_MB = int(os.environ.get("IVF_GLOBAL_CACHE_MB", "1024"))
IVF_PRELOAD_ALL = os.environ.get("IVF_PRELOAD_ALL", "false").lower() == "true"
IVF_GLOBAL_CACHE_IDLE_SECONDS = int(os.environ.get("IVF_GLOBAL_CACHE_IDLE_SECONDS", "300"))
IVF_RESULT_CACHE_SECONDS = int(os.environ.get("IVF_RESULT_CACHE_SECONDS", "300"))
IVF_RESULT_CACHE_MAX = int(os.environ.get("IVF_RESULT_CACHE_MAX", "2048"))
IVF_MAX_DISTANCE_NPROBE = int(os.environ.get("IVF_MAX_DISTANCE_NPROBE", "256"))
IVF_DISK_CACHE_ENABLED = os.environ.get("IVF_DISK_CACHE_ENABLED", "true").lower() == "true"
IVF_DISK_CACHE_IDLE_SECONDS = int(os.environ.get("IVF_DISK_CACHE_IDLE_SECONDS", "300"))
IVF_DISK_CACHE_DIR = os.environ.get("IVF_DISK_CACHE_DIR", "") or (
    os.path.join(APP_DATA_DIR, "ivf_cache")
    if APP_DATA_DIR
    else (
        "/app/ivf_cache"
        if os.path.isdir("/app")
        else os.path.join(tempfile.gettempdir(), "audiomuse_ivf_cache")
    )
)

PATH_DISTANCE_METRIC = os.environ.get("PATH_DISTANCE_METRIC", "angular").lower()
PATH_DEFAULT_LENGTH = int(os.environ.get("PATH_DEFAULT_LENGTH", "25"))
PATH_AVG_JUMP_SAMPLE_SIZE = int(os.environ.get("PATH_AVG_JUMP_SAMPLE_SIZE", "200"))
PATH_CANDIDATES_PER_STEP = int(os.environ.get("PATH_CANDIDATES_PER_STEP", "25"))
PATH_LCORE_MULTIPLIER = int(os.environ.get("PATH_LCORE_MULTIPLIER", "3"))

PATH_FIX_SIZE = os.environ.get("PATH_FIX_SIZE", "False").lower() == 'true'

MOOD_CENTROIDS_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'mood_centroids_real_080_clap.json'
)

ALCHEMY_DEFAULT_N_RESULTS = int(os.environ.get("ALCHEMY_DEFAULT_N_RESULTS", "100"))
ALCHEMY_MAX_N_RESULTS = int(os.environ.get("ALCHEMY_MAX_N_RESULTS", "200"))
ALCHEMY_TEMPERATURE = float(os.environ.get("ALCHEMY_TEMPERATURE", "1.0"))
ALCHEMY_SUBTRACT_DISTANCE_ANGULAR = float(
    os.environ.get("ALCHEMY_SUBTRACT_DISTANCE_ANGULAR", "0.2")
)
ALCHEMY_SUBTRACT_DISTANCE_EUCLIDEAN = float(
    os.environ.get("ALCHEMY_SUBTRACT_DISTANCE_EUCLIDEAN", "5.0")
)

ALCHEMY_PLAYLIST_MAX_SONGS = int(os.environ.get("ALCHEMY_PLAYLIST_MAX_SONGS", "500"))
ALCHEMY_PLAYLIST_MAX_CENTROIDS = int(os.environ.get("ALCHEMY_PLAYLIST_MAX_CENTROIDS", "10"))
ALCHEMY_MAX_ANCHOR_POINTS = int(os.environ.get("ALCHEMY_MAX_ANCHOR_POINTS", "16"))


ENERGY_MIN = float(os.getenv("ENERGY_MIN", "0.01"))
ENERGY_MAX = float(os.getenv("ENERGY_MAX", "0.15"))

TEMPO_MIN_BPM = float(os.getenv("TEMPO_MIN_BPM", "40.0"))
TEMPO_MAX_BPM = float(os.getenv("TEMPO_MAX_BPM", "200.0"))
OTHER_FEATURE_LABELS = ['danceable', 'aggressive', 'happy', 'party', 'relaxed', 'sad']

VOICE_VOCAB = ["female vocalists", "female vocalist", "male vocalists"]

AI_FALLBACK_GENRES = (
    "rock, pop, metal, jazz, electronic, dance, alternative, indie, punk, blues, "
    "hard rock, heavy metal, hip-hop, funk, country, soul"
)

CLAP_OTHER_FEATURES_REDIS_KEY = os.environ.get(
    "CLAP_OTHER_FEATURES_REDIS_KEY", "audiomuse:clap_other_feature_text_embeddings"
)

SONIC_FINGERPRINT_TOP_N_SONGS = int(os.environ.get("SONIC_FINGERPRINT_TOP_N_SONGS", "20"))
SONIC_FINGERPRINT_MAX_SONGS_PER_ALBUM = int(
    os.environ.get("SONIC_FINGERPRINT_MAX_SONGS_PER_ALBUM", "3")
)
SONIC_FINGERPRINT_NEIGHBORS = int(os.environ.get("SONIC_FINGERPRINT_NEIGHBORS", "100"))
SONIC_FINGERPRINT_CRON_PLAYLIST_NAME = os.environ.get(
    "SONIC_FINGERPRINT_CRON_PLAYLIST_NAME",
    "Sonic Fingerprint by AudioMuse-AI",
)

CLEANING_SAFETY_LIMIT = int(os.environ.get("CLEANING_SAFETY_LIMIT", "100"))

STRATIFIED_GENRES = [
    'rock',
    'pop',
    'alternative',
    'indie',
    'electronic',
    'jazz',
    'metal',
    'classic rock',
    'soul',
    'indie rock',
    'electronica',
    'folk',
    'punk',
    'blues',
    'hard rock',
    'ambient',
    'acoustic',
    'experimental',
    'Hip-Hop',
    'country',
    'funk',
    'electro',
    'heavy metal',
    'Progressive rock',
    'rnb',
    'indie pop',
    'House',
]

MIN_SONGS_PER_GENRE_FOR_STRATIFICATION = int(
    os.getenv("MIN_SONGS_PER_GENRE_FOR_STRATIFICATION", "100")
)

STRATIFIED_SAMPLING_TARGET_PERCENTILE = int(
    os.getenv("STRATIFIED_SAMPLING_TARGET_PERCENTILE", "50")
)

SAMPLING_PERCENTAGE_CHANGE_PER_RUN = float(os.getenv("SAMPLING_PERCENTAGE_CHANGE_PER_RUN", "0.2"))


DUPLICATE_DISTANCE_THRESHOLD_COSINE = float(
    os.getenv("DUPLICATE_DISTANCE_THRESHOLD_COSINE", "0.01")
)
DUPLICATE_DISTANCE_THRESHOLD_COSINE_LYRICS = float(
    os.getenv("DUPLICATE_DISTANCE_THRESHOLD_COSINE_LYRICS", "0.05")
)
DUPLICATE_DISTANCE_THRESHOLD_EUCLIDEAN = float(
    os.getenv("DUPLICATE_DISTANCE_THRESHOLD_EUCLIDEAN", "0.15")
)
DUPLICATE_DISTANCE_CHECK_LOOKBACK = int(os.getenv("DUPLICATE_DISTANCE_CHECK_LOOKBACK", "1"))

MOOD_SIMILARITY_THRESHOLD = float(os.getenv("MOOD_SIMILARITY_THRESHOLD", "0.15"))
MOOD_SIMILARITY_ENABLE = os.environ.get("MOOD_SIMILARITY_ENABLE", "False").lower() == 'true'

ENABLE_PROXY_FIX = os.environ.get("ENABLE_PROXY_FIX", "False").lower() == "true"

MAX_SONGS_PER_ARTIST_PLAYLIST = int(os.environ.get("MAX_SONGS_PER_ARTIST_PLAYLIST", "5"))
PLAYLIST_ENERGY_ARC = os.environ.get("PLAYLIST_ENERGY_ARC", "False").lower() == "true"

AI_BRAINSTORM_SOUND_DESCRIPTIONS_MAX = int(
    os.environ.get("AI_BRAINSTORM_SOUND_DESCRIPTIONS_MAX", "3")
)
AI_BRAINSTORM_SEED_ARTISTS_MAX = int(os.environ.get("AI_BRAINSTORM_SEED_ARTISTS_MAX", "4"))
AI_BRAINSTORM_USE_ARTIST_SEEDS = (
    os.environ.get("AI_BRAINSTORM_USE_ARTIST_SEEDS", "true").lower() == "true"
)
AI_BRAINSTORM_SIMILAR_ARTISTS_PER_SEED = int(
    os.environ.get("AI_BRAINSTORM_SIMILAR_ARTISTS_PER_SEED", "8")
)
AI_BRAINSTORM_LYRIC_THEMES_MAX = int(os.environ.get("AI_BRAINSTORM_LYRIC_THEMES_MAX", "2"))
AI_BRAINSTORM_GENRE_SCORE_THRESHOLD = float(
    os.environ.get("AI_BRAINSTORM_GENRE_SCORE_THRESHOLD", "0.3")
)
AI_BRAINSTORM_POOL_FLOOR = int(os.environ.get("AI_BRAINSTORM_POOL_FLOOR", "40"))
AI_BRAINSTORM_RELAX_YEAR_PAD = int(os.environ.get("AI_BRAINSTORM_RELAX_YEAR_PAD", "5"))

AUDIOMUSE_USER = os.environ.get("AUDIOMUSE_USER", "")
AUDIOMUSE_PASSWORD = os.environ.get("AUDIOMUSE_PASSWORD", "")
API_TOKEN = os.environ.get("API_TOKEN", "")

JWT_SECRET = os.environ.get("JWT_SECRET", "")

AUTH_ENABLED = os.environ.get("AUTH_ENABLED", "True").lower() == "true"


def _apply_db_overrides():
    global HEADERS, refresh_config
    try:
        from tasks.setup_manager import SetupManager

        _setup_manager = SetupManager()
        worker_mode = os.environ.get('AUDIOMUSE_ROLE', '').lower() == 'worker'
        if worker_mode:
            if _setup_manager.config_table_exists():
                _overrides = _setup_manager.get_raw_overrides(ensure_table=False)
            else:
                _overrides = {}
        else:
            _setup_manager.ensure_table()
            _overrides = _setup_manager.get_raw_overrides()
        _excluded_override_keys = globals().get('SETUP_BOOTSTRAP_EXCLUDED_KEYS', set())
        for _key, _value in _overrides.items():
            if _key in _excluded_override_keys:
                continue
            if _key in globals():
                globals()[_key] = _setup_manager.cast_value(globals()[_key], _value)

        HEADERS = _compute_headers()

        def refresh_config():
            import importlib
            import sys

            importlib.reload(sys.modules[__name__])
    except Exception as _exc:
        import logging

        logging.getLogger(__name__).warning(f"Could not load config overrides from DB: {_exc}")

        def refresh_config():
            pass


_apply_db_overrides()
