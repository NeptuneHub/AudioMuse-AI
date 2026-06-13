#AudioMuse-AI/config.py
import os

# --- Media Server Type ---
MEDIASERVER_TYPE = os.environ.get("MEDIASERVER_TYPE", "jellyfin").lower() # Possible values: jellyfin, navidrome, lyrion, emby

# --- Jellyfin and DB Constants (Read from Environment Variables first) ---

# JELLYFIN_USER_ID and JELLYFIN_TOKEN come from a Kubernetes Secret
JELLYFIN_URL = os.environ.get("JELLYFIN_URL", "") # Replace with your default URL
JELLYFIN_USER_ID = os.environ.get("JELLYFIN_USER_ID", "")  # Replace with a suitable default or handle missing case
JELLYFIN_TOKEN = os.environ.get("JELLYFIN_TOKEN", "")  # Replace with a suitable default or handle missing case

# EMBY_USER_ID and JELLYFIN_TOKEN come from a Kubernetes Secret
EMBY_URL = os.environ.get("EMBY_URL", "") # Replace with your default URL
EMBY_USER_ID = os.environ.get("EMBY_USER_ID", "")  # Replace with a suitable default or handle missing case
EMBY_TOKEN = os.environ.get("EMBY_TOKEN", "")  # Replace with a suitable default or handle missing case


# NEW: Allow specifying music libraries/folders for analysis across all media servers.
# Comma-separated list of library/folder names or paths. If empty, all music libraries/folders are scanned.
# For Lyrion: Use folder paths like "/music/myfolder"
# For Jellyfin/Navidrome: Use library/folder names
MUSIC_LIBRARIES = os.environ.get("MUSIC_LIBRARIES", "")
# Maximum number of items to fetch during the connection probe.
# Set to 0 to scan all top-played items, or a small positive integer to keep the probe fast.
PROBE_TOP_PLAYED_LIMIT = int(os.environ.get("PROBE_TOP_PLAYED_LIMIT", "1"))
# Hard cap on the number of unmatched albums returned to the migration
# wizard's step-4 review list. Real libraries can produce thousands of
# unmatched groups (e.g. wrong path format) and the page becomes unusable
# beyond a couple hundred entries. The full count is still surfaced as a
# warning so the user knows the list is truncated.
MIGRATION_UNMATCHED_ALBUMS_PAYLOAD_LIMIT = int(os.environ.get("MIGRATION_UNMATCHED_ALBUMS_PAYLOAD_LIMIT", "200"))
# Hard cap on the per-collision detail rows persisted into migration_session.state.
# collision_details is display-only (it tells the user which albums to re-match),
# so storing one entry per collision would let this single JSONB field grow with
# the library and eventually breach PG's ~1 GB field cap on a heavily-duplicated
# collection. The true total is preserved separately as collision_details_total.
MIGRATION_MAX_COLLISION_DETAILS = int(os.environ.get("MIGRATION_MAX_COLLISION_DETAILS", "1000"))
TEMP_DIR = os.environ.get("TEMP_DIR", "/app/temp_audio")


def _compute_headers():
    if MEDIASERVER_TYPE == "jellyfin":
        return {"X-Emby-Token": JELLYFIN_TOKEN}
    if MEDIASERVER_TYPE == "emby":
        return {"X-Emby-Token": EMBY_TOKEN}
    return {}

HEADERS = _compute_headers()

# --- Navidrome (Subsonic API) Constants ---
# These are used only if MEDIASERVER_TYPE is "navidrome".
NAVIDROME_URL = os.environ.get("NAVIDROME_URL", "")
NAVIDROME_USER = os.environ.get("NAVIDROME_USER", "")
NAVIDROME_PASSWORD = os.environ.get("NAVIDROME_PASSWORD", "") # Use the password directly

# --- Lyrion (LMS) Constants ---
# These are used only if MEDIASERVER_TYPE is "lyrion".
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
    # Admin identity lives in audiomuse_users only. Never mirror it into
    # app_config - stale rows there cause deleted admins to resurrect.
    'AUDIOMUSE_USER',
    'AUDIOMUSE_PASSWORD',
    # Computed numpy/precomputed constants — persisting them through
    # setup_manager would stringify the ndarray ("[1. 0. 0. ...]") and
    # corrupt the value on reload (cast_value can't reverse str(ndarray)).
    'LYRICS_INSTRUMENTAL_EMBEDDING',
    'LYRICS_INSTRUMENTAL_AXIS_FILL',
}

# --- General Constants (Read from Environment Variables where applicable) ---
APP_VERSION = "v2.2.0"
MAX_DISTANCE = float(os.environ.get("MAX_DISTANCE", "0.5"))
MAX_SONGS_PER_CLUSTER = int(os.environ.get("MAX_SONGS_PER_CLUSTER", "0"))
MAX_SONGS_PER_ARTIST = int(os.getenv("MAX_SONGS_PER_ARTIST", "3")) # Max songs per artist in similarity results and clustering
# New: Default behavior for eliminating duplicates in similarity search. If param not passed to API, this is the default.
SIMILARITY_ELIMINATE_DUPLICATES_DEFAULT = os.environ.get("SIMILARITY_ELIMINATE_DUPLICATES_DEFAULT", "True").lower() == 'true'
# Default behavior for radius similarity mode. Can be toggled via environment variable.
SIMILARITY_RADIUS_DEFAULT = os.environ.get("SIMILARITY_RADIUS_DEFAULT", "True").lower() == 'true'
NUM_RECENT_ALBUMS = int(os.getenv("NUM_RECENT_ALBUMS", "0")) # Convert to int
TOP_N_PLAYLISTS = int(os.environ.get("TOP_N_PLAYLISTS", "8")) # *** NEW: Default for Top N diverse playlists ***
MIN_PLAYLIST_SIZE_FOR_TOP_N = int(os.environ.get("MIN_PLAYLIST_SIZE_FOR_TOP_N", "20")) # Min songs for a playlist to be considered in the first pass of Top-N selection.

# --- Algorithm Choose Constants (Read from Environment Variables) ---
CLUSTER_ALGORITHM = os.environ.get("CLUSTER_ALGORITHM", "kmeans") # accepted dbscan, kmeans, gmm, or spectral
AI_MODEL_PROVIDER = os.environ.get("AI_MODEL_PROVIDER", "NONE").upper() # Accepted: OLLAMA, OPENAI, GEMINI, MISTRAL, NONE
ENABLE_CLUSTERING_EMBEDDINGS = os.environ.get("ENABLE_CLUSTERING_EMBEDDINGS", "True").lower() == "true"

# --- GPU Acceleration for Clustering (Optional, requires NVIDIA GPU and RAPIDS cuML) ---
USE_GPU_CLUSTERING = os.environ.get("USE_GPU_CLUSTERING", "False").lower() == "true"

# --- Clustering Cleanup Behavior ---
# When True (default), existing '_automatic' playlists are deleted before new clusters are created.
# Set to False to preserve old automatic playlists when running clustering.
CLUSTERING_CLEANING = os.environ.get("CLUSTERING_CLEANING", "True").lower() == "true"

# --- DBSCAN Only Constants (Ranges for Evolutionary Approach) ---
# Default ranges for DBSCAN parameters
DBSCAN_EPS_MIN = float(os.getenv("DBSCAN_EPS_MIN", "0.1"))
DBSCAN_EPS_MAX = float(os.getenv("DBSCAN_EPS_MAX", "0.5"))
DBSCAN_MIN_SAMPLES_MIN = int(os.getenv("DBSCAN_MIN_SAMPLES_MIN", "5"))
DBSCAN_MIN_SAMPLES_MAX = int(os.getenv("DBSCAN_MIN_SAMPLES_MAX", "20"))


# --- KMEANS Only Constants (Ranges for Evolutionary Approach) ---
# Default ranges for KMeans parameters
NUM_CLUSTERS_MIN = int(os.getenv("NUM_CLUSTERS_MIN", "40"))
NUM_CLUSTERS_MAX = int(os.getenv("NUM_CLUSTERS_MAX", "100"))
# New for MiniBatchKMeans
USE_MINIBATCH_KMEANS = os.environ.get("USE_MINIBATCH_KMEANS", "False").lower() == "true" # Enable MiniBatchKMeans
MINIBATCH_KMEANS_PROCESSING_BATCH_SIZE = int(os.getenv("MINIBATCH_KMEANS_PROCESSING_BATCH_SIZE", "1000")) # Internal batch size for MiniBatchKMeans partial_fit

# --- GMM Only Constants (Ranges for Evolutionary Approach) ---
# Default ranges for GMM parameters
GMM_N_COMPONENTS_MIN = int(os.getenv("GMM_N_COMPONENTS_MIN", "40"))
GMM_N_COMPONENTS_MAX = int(os.getenv("GMM_N_COMPONENTS_MAX", "100"))
GMM_COVARIANCE_TYPE = os.environ.get("GMM_COVARIANCE_TYPE", "full") # 'full', 'tied', 'diag', 'spherical'

# --- SpectralClustering Only Constants (Ranges for Evolutionary Approach) ---
SPECTRAL_N_CLUSTERS_MIN = int(os.getenv("SPECTRAL_N_CLUSTERS_MIN", "40"))
SPECTRAL_N_CLUSTERS_MAX = int(os.getenv("SPECTRAL_N_CLUSTERS_MAX", "100"))
SPECTRAL_N_NEIGHBORS = int(os.getenv("SPECTRAL_N_NEIGHBORS", "20"))

# --- PCA Constants (Ranges for Evolutionary Approach) ---
# Default ranges for PCA components
PCA_COMPONENTS_MIN = int(os.getenv("PCA_COMPONENTS_MIN", "0")) # 0 to disable PCA
PCA_COMPONENTS_MAX = int(os.getenv("PCA_COMPONENTS_MAX", "199")) # Max components for PCA 8 for score vectore, 199 for embeding

# --- Clustering Runs for Diversity (New Constant) ---
CLUSTERING_RUNS = int(os.environ.get("CLUSTERING_RUNS", "1000")) # Default to 100 runs for evolutionary search
MAX_QUEUED_ANALYSIS_JOBS = int(os.environ.get("MAX_QUEUED_ANALYSIS_JOBS", "25")) # Max album analysis jobs to keep in RQ queue (reduced from 100 to prevent resource exhaustion)

# --- Batching Constants for Clustering Runs ---
ITERATIONS_PER_BATCH_JOB = int(os.environ.get("ITERATIONS_PER_BATCH_JOB", "20")) # Number of clustering iterations per RQ batch job
MAX_CONCURRENT_BATCH_JOBS = int(os.environ.get("MAX_CONCURRENT_BATCH_JOBS", "10")) # Max number of batch jobs to run concurrently
DB_FETCH_CHUNK_SIZE = int(os.environ.get("DB_FETCH_CHUNK_SIZE", "1000")) # Chunk size for fetching full track data from DB in batch jobs

# IMPORTANT: Lower MAX_QUEUED_ANALYSIS_JOBS if experiencing resource exhaustion or server crashes
# Recommended values: 10-25 for servers with limited resources, 50-100 for powerful servers

# --- Clustering Batch Timeout and Failure Recovery ---
CLUSTERING_BATCH_TIMEOUT_MINUTES = int(os.environ.get("CLUSTERING_BATCH_TIMEOUT_MINUTES", "60")) # Max time a batch can run before being considered failed
CLUSTERING_MAX_FAILED_BATCHES = int(os.environ.get("CLUSTERING_MAX_FAILED_BATCHES", "10")) # Max number of failed batches before stopping
CLUSTERING_BATCH_CHECK_INTERVAL_SECONDS = int(os.environ.get("CLUSTERING_BATCH_CHECK_INTERVAL_SECONDS", "30")) # How often to check batch status

# --- Batching Constants for Analysis ---
REBUILD_INDEX_BATCH_SIZE = int(os.environ.get("REBUILD_INDEX_BATCH_SIZE", "1000")) # Rebuild Voyager index after this many albums are analyzed.
AUDIO_LOAD_TIMEOUT = int(os.getenv("AUDIO_LOAD_TIMEOUT", "600")) # Timeout in seconds for loading a single audio file.

# --- Guided Evolutionary Clustering Constants ---
TOP_N_ELITES = int(os.environ.get("CLUSTERING_TOP_N_ELITES", "10")) # Number of best solutions to keep as elites
EXPLOITATION_START_FRACTION = float(os.environ.get("CLUSTERING_EXPLOITATION_START_FRACTION", "0.2")) # Fraction of runs before starting to use elites (e.g., 0.2 means after 20% of runs)
EXPLOITATION_PROBABILITY_CONFIG = float(os.environ.get("CLUSTERING_EXPLOITATION_PROBABILITY", "0.7")) # Probability of mutating an elite vs. random generation, once exploitation starts
MUTATION_INT_ABS_DELTA = int(os.environ.get("CLUSTERING_MUTATION_INT_ABS_DELTA", "3")) # Max absolute change for integer parameter mutation
MUTATION_FLOAT_ABS_DELTA = float(os.environ.get("CLUSTERING_MUTATION_FLOAT_ABS_DELTA", "0.05")) # Max absolute change for float parameter mutation (e.g., for DBSCAN eps)
MUTATION_KMEANS_COORD_FRACTION = float(os.environ.get("CLUSTERING_MUTATION_KMEANS_COORD_FRACTION", "0.05")) # Fractional change for KMeans centroid coordinates based on data range

# --- Scoring Weights for Enhanced Diversity Score ---
SCORE_WEIGHT_DIVERSITY = float(os.environ.get("SCORE_WEIGHT_DIVERSITY", "2.0")) # Weight for the base diversity (inter-playlist mood diversity)
SCORE_WEIGHT_PURITY = float(os.environ.get("SCORE_WEIGHT_PURITY", "1.0"))    # Weight for playlist purity (intra-playlist mood consistency)
SCORE_WEIGHT_OTHER_FEATURE_DIVERSITY = float(os.environ.get("SCORE_WEIGHT_OTHER_FEATURE_DIVERSITY", "0.0")) # New: Weight for inter-playlist other feature diversity
SCORE_WEIGHT_OTHER_FEATURE_PURITY = float(os.environ.get("SCORE_WEIGHT_OTHER_FEATURE_PURITY", "0.0"))       # New: Weight for intra-playlist other feature consistency
# --- Weights for Internal Validation Metrics ---
SCORE_WEIGHT_SILHOUETTE = float(os.environ.get("SCORE_WEIGHT_SILHOUETTE", "0.0")) # ex 0.6 - Weight for Silhouette Score - This metric measures how similar an object is to its own cluster compared to other clusters.
SCORE_WEIGHT_DAVIES_BOULDIN = float(os.environ.get("SCORE_WEIGHT_DAVIES_BOULDIN", "0.0")) # Set to 0 to effectively disable - This index quantifies the average similarity between each cluster and its most similar one
SCORE_WEIGHT_CALINSKI_HARABASZ = float(os.environ.get("SCORE_WEIGHT_CALINSKI_HARABASZ", "0.0")) # Set to 0 to effectively disable - This metric focuses on the ratio of between-cluster dispersion to within-cluster dispersion
TOP_K_MOODS_FOR_PURITY_CALCULATION = int(os.environ.get("TOP_K_MOODS_FOR_PURITY_CALCULATION", "3")) # Number of centroid's top moods to consider for purity

# --- Statistics for Raw Score Scaling (Mood Diversity and Purity) ---
# These are based on observed typical ranges for the raw scores.
# The 'sd' (standard deviation) is stored as requested but not used in the current LN + MinMax scaling.
# Constants for Log-Transformed and Standardized Mood Diversity
LN_MOOD_DIVERSITY_STATS = {
    "min": float(os.environ.get("LN_MOOD_DIVERSITY_MIN", "-0.1863")),
    "max": float(os.environ.get("LN_MOOD_DIVERSITY_MAX", "1.5518")),
    "mean": float(os.environ.get("LN_MOOD_DIVERSITY_MEAN", "0.9995")),
    "sd": float(os.environ.get("LN_MOOD_DIVERSITY_SD", "0.3541"))
}

# Constants for Log-Transformed and Standardized Mood Diversity WHEN EMBEDDINGS ARE USED
LN_MOOD_DIVERSITY_EMBEDING_STATS = { # Corrected spelling to "EMBEDING"
    "min": float(os.environ.get("LN_MOOD_DIVERSITY_EMBEDDING_MIN", "-0.174")),
    "max": float(os.environ.get("LN_MOOD_DIVERSITY_EMBEDDING_MAX", "0.570")),
    "mean": float(os.environ.get("LN_MOOD_DIVERSITY_EMBEDDING_MEAN", "-0.101")),
    "sd": float(os.environ.get("LN_MOOD_DIVERSITY_EMBEDDING_SD", "0.245")) # Kept env var name consistent for now
}

# Constants for Log-Transformed and Standardized Mood Purity
LN_MOOD_PURITY_STATS = {
    "min": float(os.environ.get("LN_MOOD_PURITY_MIN", "0.6981")),
    "max": float(os.environ.get("LN_MOOD_PURITY_MAX", "7.2848")),
    "mean": float(os.environ.get("LN_MOOD_PURITY_MEAN", "5.8679")),
    "sd": float(os.environ.get("LN_MOOD_PURITY_SD", "1.1557"))
}

# Constants for Log-Transformed and Standardized Mood Purity WHEN EMBEDDINGS ARE USED
LN_MOOD_PURITY_EMBEDING_STATS = { # Note: User provided "EMBEDING" spelling
    "min": float(os.environ.get("LN_MOOD_PURITY_EMBEDDING_MIN", "-0.494")),
    "max": float(os.environ.get("LN_MOOD_PURITY_EMBEDDING_MAX", "2.583")),
    "mean": float(os.environ.get("LN_MOOD_PURITY_EMBEDDING_MEAN", "0.673")),
    "sd": float(os.environ.get("LN_MOOD_PURITY_EMBEDDING_SD", "1.063"))
}

# --- Statistics for Log-Transformed and Standardized "Other Features" Scores ---
# IMPORTANT: Replace these placeholder values with actual statistics derived from your data.
# These are used for Z-score standardization of the "other features" diversity and purity.
LN_OTHER_FEATURES_DIVERSITY_STATS = {
    "min": float(os.environ.get("LN_OTHER_FEAT_DIV_MIN", "-0.19")), # Placeholder
    "max": float(os.environ.get("LN_OTHER_FEAT_DIV_MAX", "2.06")), # Placeholder
    "mean": float(os.environ.get("LN_OTHER_FEAT_DIV_MEAN", "1.5")), # Placeholder
    "sd": float(os.environ.get("LN_OTHER_FEAT_DIV_SD", "0.46"))      # Placeholder
}

LN_OTHER_FEATURES_PURITY_STATS = {
    "min": float(os.environ.get("LN_OTHER_FEAT_PUR_MIN", "8.67")),   # Updated value
    "max": float(os.environ.get("LN_OTHER_FEAT_PUR_MAX", "8.95")),   # Updated value
    "mean": float(os.environ.get("LN_OTHER_FEAT_PUR_MEAN", "8.84")),  # Updated value
    "sd": float(os.environ.get("LN_OTHER_FEAT_PUR_SD", "0.07"))     # Updated value
}

# Threshold for considering an "other feature" predominant in a playlist for purity calculation
OTHER_FEATURE_PREDOMINANCE_THRESHOLD_FOR_PURITY = float(os.environ.get("OTHER_FEATURE_PREDOMINANCE_THRESHOLD_FOR_PURITY", "0.3"))

# --- AI Playlist Naming ---
# USE_AI_PLAYLIST_NAMING is replaced by AI_MODEL_PROVIDER
OLLAMA_SERVER_URL = os.environ.get("OLLAMA_SERVER_URL", "http://192.168.3.211:11434/api/generate") # URL for your Ollama instance
OLLAMA_MODEL_NAME = os.environ.get("OLLAMA_MODEL_NAME", "llama3.1:8b") # Ollama model to use

# Maximum number of songs to include in AI naming prompts (to avoid token limit issues)
# Large playlists will use only the first N songs for naming
MAX_SONGS_IN_AI_PROMPT = int(os.environ.get("MAX_SONGS_IN_AI_PROMPT", "25"))

# OpenAI API (also used for OpenRouter) - uses same API standard as Ollama
OPENAI_SERVER_URL = os.environ.get("OPENAI_SERVER_URL", os.environ.get("OLLAMA_SERVER_URL", "http://192.168.3.211:11434/api/generate"))
OPENAI_MODEL_NAME = os.environ.get("OPENAI_MODEL_NAME", os.environ.get("OLLAMA_MODEL_NAME", "llama3.1:8b"))
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "no-key-needed") # Set to "no-key-needed" for Ollama, or your actual API key for OpenAI/OpenRouter

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "") # Default API key
GEMINI_MODEL_NAME = os.environ.get("GEMINI_MODEL_NAME", "gemini-2.5-pro") # Default Gemini model gemini-2.5-pro, alternative gemini-2.5-flash

MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY", "")
MISTRAL_MODEL_NAME = os.environ.get("MISTRAL_MODEL_NAME", "ministral-3b-latest")

# AI Request Timeout Configuration
# Timeout in seconds for AI API requests. Increase this value if using slower hardware or larger models.
# For CPU-only Ollama instances or large models that take longer to generate responses, consider setting to 300-600 seconds.
# Default: 120 seconds for Ollama (tool calling/instant playlist), 60 seconds for OpenAI/Mistral
AI_REQUEST_TIMEOUT_SECONDS = int(os.environ.get("AI_REQUEST_TIMEOUT_SECONDS", "300"))
REDIS_URL = os.environ.get('REDIS_URL', 'redis://localhost:6379/0')

# Construct DATABASE_URL from individual components for better security in K8s
POSTGRES_USER = os.environ.get("POSTGRES_USER", "audiomuse")
POSTGRES_PASSWORD = os.environ.get("POSTGRES_PASSWORD", "audiomusepassword")
POSTGRES_HOST = os.environ.get("POSTGRES_HOST", "postgres-service.playlist") # Default for K8s
POSTGRES_PORT = os.environ.get("POSTGRES_PORT", "5432")
POSTGRES_DB = os.environ.get("POSTGRES_DB", "audiomusedb")

# Allow an explicit DATABASE_URL to override construction (useful for docker-compose or direct env override)
from urllib.parse import quote

# Percent-encode username and password to safely include special characters like '@' in the URI
_pg_user_esc = quote(POSTGRES_USER, safe='')
_pg_pass_esc = quote(POSTGRES_PASSWORD, safe='')

# If DATABASE_URL is set in the environment, prefer it; otherwise build one using the escaped credentials
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    f"postgresql://{_pg_user_esc}:{_pg_pass_esc}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
)

DATABASE_TYPE = os.environ.get("DATABASE_TYPE", "postgres").lower()
QUEUE_TYPE = os.environ.get("QUEUE_TYPE", "redis").lower()
APP_DATA_DIR = os.environ.get("APP_DATA_DIR", "")
AUDIOMUSE_PLATFORM = os.environ.get("AUDIOMUSE_PLATFORM", "").lower()
AUDIOMUSE_CONTROL_SOCKET = os.environ.get("AUDIOMUSE_CONTROL_SOCKET", "")
AUDIOMUSE_CONTROL_HOST = os.environ.get("AUDIOMUSE_CONTROL_HOST", "")
AUDIOMUSE_CONTROL_PORT = os.environ.get("AUDIOMUSE_CONTROL_PORT", "")

# --- AI User for Chat SQL Execution ---
AI_CHAT_DB_USER_NAME = os.environ.get("AI_CHAT_DB_USER_NAME", "ai_user")
AI_CHAT_DB_USER_PASSWORD = os.environ.get("AI_CHAT_DB_USER_PASSWORD", "ChangeThisSecurePassword123!") # IMPORTANT: Change this default and use environment variables

# --- Classifier Constant ---
MOOD_LABELS = [
    'rock', 'pop', 'alternative', 'indie', 'electronic', 'female vocalists', 'dance', '00s', 'alternative rock', 'jazz',
    'beautiful', 'metal', 'chillout', 'male vocalists', 'classic rock', 'soul', 'indie rock', 'Mellow', 'electronica', '80s',
    'folk', '90s', 'chill', 'instrumental', 'punk', 'oldies', 'blues', 'hard rock', 'ambient', 'acoustic', 'experimental',
    'female vocalist', 'guitar', 'Hip-Hop', '70s', 'party', 'country', 'easy listening', 'sexy', 'catchy', 'funk', 'electro',
    'heavy metal', 'Progressive rock', '60s', 'rnb', 'indie pop', 'sad', 'House', 'happy'
]

TOP_N_MOODS = int(os.environ.get("TOP_N_MOODS", "5"))  # Number of top moods to consider (configurable via env)
EMBEDDING_MODEL_PATH = os.environ.get("EMBEDDING_MODEL_PATH", "/app/model/musicnn_embedding.onnx")
PREDICTION_MODEL_PATH = os.environ.get("PREDICTION_MODEL_PATH", "/app/model/musicnn_prediction.onnx")

# --- Pluggable Sonic Backend ---
# Which model produces the per-track similarity embedding stored in the
# ``embedding`` table and indexed by Voyager. ``musicnn`` is the
# historical default (200-dim, ONNX, behaves bit-for-bit like prior
# releases). ``mert`` uses the self-supervised MERT music foundation
# model (768-dim @ MERT-95M / 1024-dim @ MERT-330M, via HF transformers)
# and keeps the MusiCNN-prediction ONNX head for mood/genre tags so
# mood_vector rows stay schema-compatible.
#
# Switching backends is a destructive operation for the Voyager / audio
# embedding tables — embedding dimensions differ — so a full
# re-analysis is required after toggling. See tasks/sonic_backends/.
SONIC_BACKEND = os.environ.get("SONIC_BACKEND", "musicnn").lower()
MERT_MODEL_ID = os.environ.get("MERT_MODEL_ID", "m-a-p/MERT-v1-95M")
# Which transformer layer to mean-pool. ``-1`` = last hidden layer (a
# safe default across MERT variants). Music-tagging downstream tasks
# often peak around layers 5–7 for MERT-95M; tune if you have a
# labeled eval set.
MERT_LAYER = int(os.environ.get("MERT_LAYER", "-1"))
# ``auto`` picks CUDA when torch reports it available, else CPU.
MERT_DEVICE = os.environ.get("MERT_DEVICE", "auto").lower()
MERT_TARGET_SR = int(os.environ.get("MERT_TARGET_SR", "24000"))
# Override the Hugging Face cache location when, e.g., the default
# ~/.cache/huggingface path is not writable inside the worker
# container. Empty string = use HF default.
MERT_HF_CACHE_DIR = os.environ.get("MERT_HF_CACHE_DIR", "")

# --- Mood centroids (per-backend derivation) ---
# Number of KMeans clusters built per ``OTHER_FEATURE_LABELS`` mood. The
# legacy frozen JSON used a BIC sweep to pick k; a fixed 4 hits the
# typical sweet spot for the available label volume and keeps the rebuild
# cheap. Override per deployment if you have a much larger library.
MOOD_CENTROIDS_K = int(os.environ.get("MOOD_CENTROIDS_K", "4"))
# Minimum CLAP "other_features" score for a track to count toward a
# mood's cluster pool. 0.4 mirrors the heuristic the legacy JSON
# implicitly used (tracks with a clear-enough mood signal).
MOOD_CENTROIDS_MIN_SCORE = float(os.environ.get("MOOD_CENTROIDS_MIN_SCORE", "0.4"))

# Resolved per-backend embedding dim. Voyager builders / persistence
# code import ``EMBEDDING_DIMENSION`` directly as a scalar, so we
# resolve at import time. We deliberately do NOT import the backend
# registry here — that pulls in ``tasks.analysis_helper`` ➜
# ``app_helper`` ➜ Flask machinery, which is not safe at config import
# time. The dim lookup table below mirrors the ``embedding_dim`` class
# attributes on the shipped backends; adding a new backend means
# adding a row here too. The unit test in
# ``test/unit/test_sonic_backends.py`` enforces parity.
_BACKEND_EMBEDDING_DIMS = {
    "musicnn": 200,
    "mert": {
        "m-a-p/MERT-v1-95M": 768,
        "m-a-p/MERT-v1-330M": 1024,
    }.get(MERT_MODEL_ID, 768),
}
EMBEDDING_DIMENSION = _BACKEND_EMBEDDING_DIMS.get(SONIC_BACKEND, 200)

# --- CLAP Model Constants (for text search) ---
CLAP_ENABLED = os.environ.get("CLAP_ENABLED", "true").lower() == "true"
# Lyrics analysis feature toggle. When false, the lyrics step is skipped entirely.
LYRICS_ENABLED = os.environ.get("LYRICS_ENABLED", "true").lower() == "true"
# When true, look up lyrics from user-configured external APIs before falling back to Whisper-small ASR.
LYRICS_API_ENABLE = os.environ.get("LYRICS_API_ENABLE", "true").lower() == "true"
LYRICS_ASR_ENABLE = os.environ.get("LYRICS_ASR_ENABLE", "true").lower() == "true"
# Timeout (seconds) for fetching embedded lyrics from the configured media server
# (Jellyfin / Emby / Navidrome / Lyrion). Increase if your server fetches lyrics
# on-the-fly via plugins (e.g. Navidrome lyrics plugins) that may take several
# seconds to respond. Set 0 to disable the lookup.
MUSICSERVER_LYRICS_TIMEOUT = float(os.environ.get("MUSICSERVER_LYRICS_TIMEOUT", "2.5"))
# User-configurable lyrics API slots (up to 2).
# Each slot stores: url_template, lyrics_field, artist_param, title_param, api_key_param, api_key_value
# e.g. LYRICS_API_1_URL_TEMPLATE = "https://example.com/api/get?{artist_param}={artist}&{title_param}={title}"
LYRICS_API_1_URL_TEMPLATE  = os.environ.get("LYRICS_API_1_URL_TEMPLATE",  "")
LYRICS_API_1_ARTIST_PARAM  = os.environ.get("LYRICS_API_1_ARTIST_PARAM",  "artist_name")
LYRICS_API_1_TITLE_PARAM   = os.environ.get("LYRICS_API_1_TITLE_PARAM",   "track_name")
LYRICS_API_1_LYRICS_FIELD  = os.environ.get("LYRICS_API_1_LYRICS_FIELD",  "plainLyrics")
LYRICS_API_1_APIKEY_PARAM  = os.environ.get("LYRICS_API_1_APIKEY_PARAM",  "")
LYRICS_API_1_APIKEY_VALUE  = os.environ.get("LYRICS_API_1_APIKEY_VALUE",  "")
LYRICS_API_1_TIMEOUT       = float(os.environ.get("LYRICS_API_1_TIMEOUT",   "5.0"))
LYRICS_API_2_URL_TEMPLATE  = os.environ.get("LYRICS_API_2_URL_TEMPLATE",  "")
LYRICS_API_2_ARTIST_PARAM  = os.environ.get("LYRICS_API_2_ARTIST_PARAM",  "artist")
LYRICS_API_2_TITLE_PARAM   = os.environ.get("LYRICS_API_2_TITLE_PARAM",   "title")
LYRICS_API_2_LYRICS_FIELD  = os.environ.get("LYRICS_API_2_LYRICS_FIELD",  "lyrics")
LYRICS_API_2_APIKEY_PARAM  = os.environ.get("LYRICS_API_2_APIKEY_PARAM",  "")
LYRICS_API_2_APIKEY_VALUE  = os.environ.get("LYRICS_API_2_APIKEY_VALUE",  "")
LYRICS_API_2_TIMEOUT       = float(os.environ.get("LYRICS_API_2_TIMEOUT",   "5.0"))
# Beam search width for the Whisper-small ASR decoder. 1 = pure greedy
# (fastest, most error-prone), 2 = sweet spot (catches stuck-loop
# attractors at ~2× greedy cost), 5 = Whisper-upstream default (max
# quality, ~5× cost). Each extra beam adds one decoder.run per generated
# token plus its own KV cache (~30-80 MB at a full 30 s chunk).
LYRICS_ASR_BEAM_SIZE = int(os.environ.get("LYRICS_ASR_BEAM_SIZE", "5"))
LYRICS_ASR_MIN_AVG_LOGPROB = float(os.environ.get("LYRICS_ASR_MIN_AVG_LOGPROB", "-1.0"))
LYRICS_ASR_NON_ENGLISH_MIN_LOGPROB = float(os.environ.get("LYRICS_ASR_NON_ENGLISH_MIN_LOGPROB", "-0.85"))
# Where the Whisper-small ONNX bundle (encoder + merged decoder +
# tokenizer files) is extracted. Pre-bundled in the official Docker
# image from lyrics_model_whisper.tar.gz (project release).
LYRICS_WHISPER_MODEL_DIR = os.environ.get(
    "LYRICS_WHISPER_MODEL_DIR",
    os.path.join(os.environ.get("LYRICS_MODEL_DIR", "/app/model"), "whisper-small-onnx"),
)
LYRICS_MODEL_DIR = os.environ.get("LYRICS_MODEL_DIR", "/app/model")
LYRICS_MAX_SONGS_TO_ANALYZE = 1000
LYRICS_SUPPORTED_AUDIO_EXTENSIONS = {
    '.wav', '.mp3', '.m4a', '.flac', '.ogg', '.opus', '.aac', '.aiff', '.aif', '.mp4'
}
# Minimum seconds of voiced audio Silero VAD must detect for a track to be
# sent to Whisper. Below this, the song is treated as instrumental and the
# instrumental embedding sentinel is used instead. Setting this very high
# effectively disables Whisper transcription for most tracks.
VAD_VOICE_RECOGNITION = int(os.environ.get("VAD_VOICE_RECOGNITION", "25"))

LYRICS_DEFAULT_SAMPLE_RATE = 16000
LYRICS_DEFAULT_SEGMENT_DURATION = 60.0
LYRICS_DEFAULT_TOPIC_EMBEDDING_MODEL = 'Alibaba-NLP/gte-multilingual-base'
LYRICS_DEFAULT_TOPIC_EMBEDDING_CACHE_DIR = os.path.join(LYRICS_MODEL_DIR, 'gte-multilingual-base')
# Dimension of the gte-multilingual-base sentence embedding stored in
# lyrics_embedding.embedding and used to build the lyrics voyager index.
LYRICS_EMBEDDING_DIMENSION = int(os.environ.get("LYRICS_EMBEDDING_DIMENSION", "768"))

# Minimum number of CHARACTERS (not words) a transcript must have for the
# lyrics pipeline to compute an embedding. Below this, the song is treated
# as having no usable lyrics and gets the instrumental sentinel. Char-based
# instead of word-based so CJK / Thai / Lao scripts — which have no spaces
# between words — aren't all collapsed to "1 word" by str.split() and
# spuriously dropped. 250 chars ~ 50 English words at 5 chars/word average,
# or ~150 CJK chars (roughly equivalent lyrical content).
LYRICS_MIN_CHARS_FOR_EMBEDDING = int(os.environ.get("LYRICS_MIN_CHARS_FOR_EMBEDDING", "250"))
# Repetition gate for text-source lyrics (media server / external API). Pure
# ad-lib or filler content ("woo woo woo...") compresses far more than real
# lyrics, so a high zlib compression ratio flags it. Above this ratio the text
# is dropped before embedding, preventing nonsensical content from polluting
# embeddings (issue #543). Set deliberately high so genuinely chorus-heavy real
# songs (~7-8) survive while extreme ad-lib repetition (~30-40+) is removed.
# Set to 0 to disable the gate.
LYRICS_TEXT_MAX_COMPRESSION_RATIO = float(os.environ.get("LYRICS_TEXT_MAX_COMPRESSION_RATIO", "15.0"))
# Minimum langdetect confidence (API / music-server lyrics) to accept the text as
# real lyrics. Below this the lyrics are dropped to the instrumental sentinel rather
# than let unidentifiable junk (garbled API responses, mojibake, filler) pollute the
# embedding (issue #543). Embedding is multilingual now, so this is purely a quality
# gate — no translation is performed.
LYRICS_LANG_CONFIDENCE_MIN = float(os.environ.get("LYRICS_LANG_CONFIDENCE_MIN", "0.70"))
# Minimum fraction of letters that must be CJK script (Hangul / kana / Han) for the
# lyrics to be treated as genuine CJK regardless of what langdetect reports. Code-mixed
# K-pop / J-pop is frequently scored low-confidence by langdetect because of its
# Latin-script bias; script presence is a far more reliable CJK signal, letting real
# CJK lyrics bypass the confidence gate instead of being dropped (issue #553). Set 0
# to disable.
LYRICS_CJK_SCRIPT_MIN_RATIO = float(os.environ.get("LYRICS_CJK_SCRIPT_MIN_RATIO", "0.10"))
# Maximum number of tokens fed to the gte-multilingual-base embedding model per
# track. The model supports up to 8192; lyrics are truncated here (default 512,
# roughly a full song — ~500 tokens for English, fewer characters for CJK which
# fragments into more tokens). Higher values embed more of long songs at extra
# CPU cost. Changing this alters the embeddings, so re-embed (drop the lyrics
# tables) afterwards for consistency.
LYRICS_GTE_MAX_TOKENS = int(os.environ.get("LYRICS_GTE_MAX_TOKENS", "512"))

# --- SemGrove (Semantic + Groove) merged index weights ---
# Controls how much each signal contributes to the merged cosine similarity.
# Values are the squared scale factors so that:
#   merged cosine = WEIGHT_LYRICS * cos(lyrics) + WEIGHT_AUDIO * cos(audio)
# Both values must be in [0.0, 1.0]. They are baked into the index at build
# time; changing them requires rebuilding the SemGrove index.
SEM_GROVE_WEIGHT_LYRICS = float(os.environ.get("SEM_GROVE_WEIGHT_LYRICS", "0.75"))
SEM_GROVE_WEIGHT_AUDIO  = float(os.environ.get("SEM_GROVE_WEIGHT_AUDIO",  "0.25"))

# --- Sentinel vectors for tracks with no detectable lyrics ("instrumental") ---
# These give us three things at once:
#   1. analyze_lyrics() can still write a row, so future runs skip the track
#      instead of re-attempting transcription every time.
#   2. The vectors are non-zero so cosine similarity is always well-defined.
#   3. Querying the index with the same sentinel lists every instrumental at
#      the top, while real songs cannot match them: the embedding sentinel sits on
#      a single basis axis (cosine to typical text embeddings is ~0), and the
#      axis sentinel is uniformly negative, which a softmax-derived axis_vector
#      can never produce.
import numpy as _np

LYRICS_INSTRUMENTAL_EMBEDDING = _np.zeros(LYRICS_EMBEDDING_DIMENSION, dtype=_np.float32)
LYRICS_INSTRUMENTAL_EMBEDDING[0] = 1.0
LYRICS_INSTRUMENTAL_EMBEDDING.flags.writeable = False

# Fill value used for every entry of the instrumental axis_vector. Any negative
# constant works because real axis_vectors come from softmax (always >= 0), so
# they cannot occupy the negative orthant. Hardcoded so we never compute
# sqrt() at runtime.
LYRICS_INSTRUMENTAL_AXIS_FILL = -0.19245009  # = -1 / sqrt(27), precomputed

# Split CLAP models: audio model for analysis, text model for search
# Default points to the distilled student model (EfficientAT, epoch 36).
# The companion external-data file (model_epoch_36.onnx.data) must sit next to it.
# To revert to the original teacher model set CLAP_AUDIO_MODEL_PATH=/app/model/clap_audio_model.onnx
# and override the mel params (see CLAP_AUDIO_* variables below).
CLAP_AUDIO_MODEL_PATH = os.environ.get("CLAP_AUDIO_MODEL_PATH", "/app/model/model_epoch_36.onnx")

# Mel-spectrogram parameters for the CLAP audio model.
# Defaults match the distilled student model (EfficientAT, model_epoch_36.onnx).
# For the original teacher model (clap_audio_model.onnx) override to:
#   CLAP_AUDIO_N_MELS=64  CLAP_AUDIO_N_FFT=1024  CLAP_AUDIO_HOP_LENGTH=480
#   CLAP_AUDIO_FMIN=50    CLAP_AUDIO_MEL_TRANSPOSE=true
CLAP_AUDIO_N_MELS = int(os.environ.get("CLAP_AUDIO_N_MELS", "128"))
CLAP_AUDIO_N_FFT = int(os.environ.get("CLAP_AUDIO_N_FFT", "2048"))
CLAP_AUDIO_HOP_LENGTH = int(os.environ.get("CLAP_AUDIO_HOP_LENGTH", "480"))
CLAP_AUDIO_FMIN = int(os.environ.get("CLAP_AUDIO_FMIN", "0"))
CLAP_AUDIO_FMAX = int(os.environ.get("CLAP_AUDIO_FMAX", "14000"))
# Teacher model (HTSAT) transposes mel to (time, mels); student does not.
CLAP_AUDIO_MEL_TRANSPOSE = os.environ.get("CLAP_AUDIO_MEL_TRANSPOSE", "false").lower() == "true"

CLAP_TEXT_MODEL_PATH = os.environ.get("CLAP_TEXT_MODEL_PATH", "/app/model/clap_text_model.onnx")
CLAP_EMBEDDING_DIMENSION = 512
# CPU threading for CLAP analysis:
# - False (default): Use ONNX internal threading (auto-detects all CPU cores, recommended)
# - True: Use Python ThreadPoolExecutor with auto-calculated threads: (physical_cores - 1) + (logical_cores // 2)
CLAP_PYTHON_MULTITHREADS = os.environ.get("CLAP_PYTHON_MULTITHREADS", "False").lower() == "true"

# Model reloading strategy to prevent GPU VRAM accumulation
# - true (default): Unload both MusiCNN and CLAP models after each song
#   Pros: Stable memory usage, prevents VRAM leaks
#   Cons: Slower (~2-3 seconds overhead per song for model loading)
# - false: MusiCNN reloads every 20 songs, CLAP at album end (faster but may accumulate memory)
#   Pros: Faster processing (no per-song reload overhead)
#   Cons: May see gradual VRAM growth on some systems
PER_SONG_MODEL_RELOAD = os.environ.get("PER_SONG_MODEL_RELOAD", "true").lower() == "true"

# Category weights for CLAP query generation (affects random query sampling probabilities)
# Higher weights favor categories where CLAP excels (Genre, Instrumentation)
# Format: JSON string with category names as keys and float weights as values
CLAP_CATEGORY_WEIGHTS_DEFAULT = {
    "Genre_Style": 1.0,           # CLAP excels at genre detection
    "Instrumentation_Vocal": 1.0, # CLAP excels at instrument detection
    "Emotion_Mood": 1.0,
    "Voice_Type": 1.0
}
import json
CLAP_CATEGORY_WEIGHTS = json.loads(
    os.environ.get("CLAP_CATEGORY_WEIGHTS", json.dumps(CLAP_CATEGORY_WEIGHTS_DEFAULT))
)

# Number of random queries to generate for top query recommendations
CLAP_TOP_QUERIES_COUNT = int(os.environ.get("CLAP_TOP_QUERIES_COUNT", "1000"))

# Duration (in seconds) to keep CLAP model loaded for text search after last use
# Model auto-unloads after this period of inactivity to free ~500MB RAM
CLAP_TEXT_SEARCH_WARMUP_DURATION = int(os.environ.get("CLAP_TEXT_SEARCH_WARMUP_DURATION", "300"))

# Duration (in seconds) to keep the gte-multilingual-base lyrics-search model
# loaded after last use. Auto-unloads after this idle period to free RAM.
LYRICS_GTE_WARMUP_DURATION = int(os.environ.get("LYRICS_GTE_WARMUP_DURATION", "300"))

# --- Voyager Index Constants ---
INDEX_NAME = os.environ.get("VOYAGER_INDEX_NAME", "music_library") # The primary key for our index in the DB
VOYAGER_METRIC = os.environ.get("VOYAGER_METRIC", "angular") # Options: 'angular' (Cosine), 'euclidean', 'dot' (InnerProduct)
VOYAGER_EF_CONSTRUCTION = int(os.environ.get("VOYAGER_EF_CONSTRUCTION", "200"))
VOYAGER_M = int(os.environ.get("VOYAGER_M", "32"))
VOYAGER_QUERY_EF = int(os.environ.get("VOYAGER_QUERY_EF", "1024"))
VOYAGER_MAX_PART_SIZE_MB = int(os.environ.get("VOYAGER_MAX_PART_SIZE_MB", "50"))  # Max part size (MB) for voyager index storage
ARTIST_INDEX_MAX_PART_SIZE_MB = int(os.environ.get("ARTIST_INDEX_MAX_PART_SIZE_MB", "50"))  # Max part size (MB) for artist index storage

# --- Pathfinding Constants ---
# The distance metric to use for pathfinding. Options: 'angular', 'euclidean'.
PATH_DISTANCE_METRIC = os.environ.get("PATH_DISTANCE_METRIC", "angular").lower()
# Default number of songs in the path if not specified in the API request.
PATH_DEFAULT_LENGTH = int(os.environ.get("PATH_DEFAULT_LENGTH", "25"))
# Number of random songs to sample for calculating the average jump distance.
PATH_AVG_JUMP_SAMPLE_SIZE = int(os.environ.get("PATH_AVG_JUMP_SAMPLE_SIZE", "200"))
# Number of candidate songs to retrieve from Voyager for each step in the path.
PATH_CANDIDATES_PER_STEP = int(os.environ.get("PATH_CANDIDATES_PER_STEP", "25"))
# Multiplier for the core number of steps (Lcore) to generate more backbone centroids.
PATH_LCORE_MULTIPLIER = int(os.environ.get("PATH_LCORE_MULTIPLIER", "3"))

# When True (default) the path generation attempts to produce exactly the requested
# path length using centroid merging and backfilling. When False, the algorithm
# will *not* perform centroid merging: it will attempt a single best pick per
# centroid and skip centroids that don't yield a non-duplicate song (resulting
# in potentially shorter paths). Can be overridden via env var PATH_FIX_SIZE.
PATH_FIX_SIZE = os.environ.get("PATH_FIX_SIZE", "False").lower() == 'true'

# Path to the JSON file containing mood centroids for the path-to-mood feature.
MOOD_CENTROIDS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mood_centroids_real_080_clap.json')

# --- Song Alchemy Defaults ---
# Number of similar songs to return when creating the Alchemy result (default 100, max 200)
ALCHEMY_DEFAULT_N_RESULTS = int(os.environ.get("ALCHEMY_DEFAULT_N_RESULTS", "100"))
ALCHEMY_MAX_N_RESULTS = int(os.environ.get("ALCHEMY_MAX_N_RESULTS", "200"))
# Temperature for probabilistic sampling in Song Alchemy (softmax temperature)
ALCHEMY_TEMPERATURE = float(os.environ.get("ALCHEMY_TEMPERATURE", "1.0"))
# Minimum distance from the subtract-centroid to keep a candidate (metric-dependent).
# For angular (cosine-derived) distances this is in [0,1] where higher means more distant.
ALCHEMY_SUBTRACT_DISTANCE_ANGULAR = float(os.environ.get("ALCHEMY_SUBTRACT_DISTANCE_ANGULAR", "0.2"))
ALCHEMY_SUBTRACT_DISTANCE_EUCLIDEAN = float(os.environ.get("ALCHEMY_SUBTRACT_DISTANCE_EUCLIDEAN", "5.0"))


# --- Other Feature Labels (computed via CLAP text-audio similarity) ---
# These features are computed by comparing CLAP audio embeddings against
# cached CLAP text embeddings for each label (no separate ONNX models needed).
# Mood-specific models (danceability, mood_aggressive, etc.) have been removed.

# --- Energy Normalization Range ---
ENERGY_MIN = float(os.getenv("ENERGY_MIN", "0.01"))
ENERGY_MAX = float(os.getenv("ENERGY_MAX", "0.15"))

# --- Tempo Normalization Range (BPM) ---
TEMPO_MIN_BPM = float(os.getenv("TEMPO_MIN_BPM", "40.0"))
TEMPO_MAX_BPM = float(os.getenv("TEMPO_MAX_BPM", "200.0"))
OTHER_FEATURE_LABELS = ['danceable', 'aggressive', 'happy', 'party', 'relaxed', 'sad']

# Redis cache key for CLAP text embeddings of OTHER_FEATURE_LABELS
CLAP_OTHER_FEATURES_REDIS_KEY = os.environ.get("CLAP_OTHER_FEATURES_REDIS_KEY", "audiomuse:clap_other_feature_text_embeddings")

# --- Sonic Fingerprint Constants ---
SONIC_FINGERPRINT_TOP_N_SONGS = int(os.environ.get("SONIC_FINGERPRINT_TOP_N_SONGS", "20"))
# Max tracks a single album may contribute to the seed pool, so one large album
# (e.g. a 100+ track DJ mix) cannot dominate the fingerprint — see issue #603.
SONIC_FINGERPRINT_MAX_SONGS_PER_ALBUM = int(os.environ.get("SONIC_FINGERPRINT_MAX_SONGS_PER_ALBUM", "3"))
SONIC_FINGERPRINT_NEIGHBORS = int(os.environ.get("SONIC_FINGERPRINT_NEIGHBORS", "100"))
SONIC_FINGERPRINT_CRON_PLAYLIST_NAME = os.environ.get(
    "SONIC_FINGERPRINT_CRON_PLAYLIST_NAME",
    "Sonic Fingerprint by AudioMuse-AI",
)

# --- Database Cleaning Safety ---
CLEANING_SAFETY_LIMIT = int(os.environ.get("CLEANING_SAFETY_LIMIT", "100"))  # Max orphaned albums to delete in one run

# --- Stratified Sampling Constants (New) ---
# Genres for which to enforce equal representation during stratified sampling
STRATIFIED_GENRES = [
    'rock', 'pop', 'alternative', 'indie', 'electronic', 'jazz', 'metal', 'classic rock', 'soul',
    'indie rock', 'electronica', 'folk', 'punk', 'blues', 'hard rock', 'ambient', 'acoustic',
    'experimental', 'Hip-Hop', 'country', 'funk', 'electro', 'heavy metal', 'Progressive rock',
    'rnb', 'indie pop', 'House'
]

# Minimum number of songs to target per genre for stratified sampling.
# This will be dynamically adjusted based on actual available songs.
MIN_SONGS_PER_GENRE_FOR_STRATIFICATION = int(os.getenv("MIN_SONGS_PER_GENRE_FOR_STRATIFICATION", "100"))

# Percentile to use for determining the target number of songs per genre in stratified sampling.
# E.g., 75 means the target will be based on the 75th percentile of song counts among stratified genres.
STRATIFIED_SAMPLING_TARGET_PERCENTILE = int(os.getenv("STRATIFIED_SAMPLING_TARGET_PERCENTILE", "50"))

# Percentage of songs to change in the stratified sample between clustering runs (0.0 to 1.0)
SAMPLING_PERCENTAGE_CHANGE_PER_RUN = float(os.getenv("SAMPLING_PERCENTAGE_CHANGE_PER_RUN", "0.2"))


# --- NEW: Duplicate Detection by Distance ---
# Threshold for considering songs as duplicates based on their distance in the vector space.
# This helps catch identical songs with slightly different metadata (e.g., from different albums).
DUPLICATE_DISTANCE_THRESHOLD_COSINE = float(os.getenv("DUPLICATE_DISTANCE_THRESHOLD_COSINE", "0.01"))
DUPLICATE_DISTANCE_THRESHOLD_COSINE_LYRICS = float(os.getenv("DUPLICATE_DISTANCE_THRESHOLD_COSINE_LYRICS", "0.05"))
DUPLICATE_DISTANCE_THRESHOLD_EUCLIDEAN = float(os.getenv("DUPLICATE_DISTANCE_THRESHOLD_EUCLIDEAN", "0.15"))
DUPLICATE_DISTANCE_CHECK_LOOKBACK = int(os.getenv("DUPLICATE_DISTANCE_CHECK_LOOKBACK", "1"))

# --- Mood Similarity Filtering ---
# Threshold for mood similarity filtering. Lower values = stricter filtering (more similar moods required).
# Range: 0.0 (identical moods only) to 1.0 (any mood difference allowed)
MOOD_SIMILARITY_THRESHOLD = float(os.getenv("MOOD_SIMILARITY_THRESHOLD", "0.15"))
# Enable or disable mood similarity filtering globally (default: disabled for radius experiments)
MOOD_SIMILARITY_ENABLE = os.environ.get("MOOD_SIMILARITY_ENABLE", "False").lower() == 'true'

# --- Enable Proxy Fix for Flask when behind a reverse proxy ---
# Actually only one proxy is allowed between client and app.
# Example nginx configuration:
# location /audiomuseai/ {
#   proxy_pass http://127.0.0.1:8000/;
#   proxy_http_version 1.1;
#   proxy_set_header X-Forwarded-Host myhostname;
#   proxy_set_header X-Forwarded-For $remote_addr;
#   proxy_set_header X-Forwarded-Port 443;
#   proxy_set_header X-Forwarded-Proto https;
#   proxy_set_header X-Forwarded-Prefix /audiomuseai;
# }
ENABLE_PROXY_FIX = os.environ.get("ENABLE_PROXY_FIX", "False").lower() == "true"

# --- Instant Playlist Optimization ---
# Max songs from a single artist in the instant playlist (diversity enforcement)
MAX_SONGS_PER_ARTIST_PLAYLIST = int(os.environ.get("MAX_SONGS_PER_ARTIST_PLAYLIST", "5"))
# Enable energy-arc shaping for playlist ordering (gentle start -> peak -> cool down)
PLAYLIST_ENERGY_ARC = os.environ.get("PLAYLIST_ENERGY_ARC", "False").lower() == "true"
# --- Authentication ---
# Set all three to enable authentication. Leave any blank to disable (legacy mode).
AUDIOMUSE_USER = os.environ.get("AUDIOMUSE_USER", "")
AUDIOMUSE_PASSWORD = os.environ.get("AUDIOMUSE_PASSWORD", "")
API_TOKEN = os.environ.get("API_TOKEN", "")

# JWT secret for signing session tokens. Auto-generated if not set (sessions lost on restart).
# Note: the warning for missing JWT_SECRET is emitted in app.py after logging is configured
JWT_SECRET = os.environ.get("JWT_SECRET", "")

# Enable or disable authentication independently of whether credentials are set.
# Default is True to preserve the current secure behavior.
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
            # Skip any keys that are explicitly excluded from overrides (Redis and Postgres)
            if _key in _excluded_override_keys:
                continue
            # Read the value from the db and override the variable
            if _key in globals():
                globals()[_key] = _setup_manager.cast_value(globals()[_key], _value)

        HEADERS = _compute_headers()

        def refresh_config():
            """Reload the config module from the current database and environment."""
            import importlib
            import sys
            importlib.reload(sys.modules[__name__])
    except Exception as _exc:
        import logging
        logging.getLogger(__name__).warning(f"Could not load config overrides from DB: {_exc}")
        def refresh_config():
            pass


_apply_db_overrides()
