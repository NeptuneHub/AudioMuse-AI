# app_clustering.py
from flask import Blueprint, jsonify, request
import uuid
import logging
import traceback

# Import all necessary configuration variables
from config import JELLYFIN_URL, JELLYFIN_USER_ID, JELLYFIN_TOKEN, HEADERS, TEMP_DIR, \
    REDIS_URL, DATABASE_URL, MAX_DISTANCE, MAX_SONGS_PER_CLUSTER, MAX_SONGS_PER_ARTIST, NUM_RECENT_ALBUMS, \
    SCORE_WEIGHT_DIVERSITY, SCORE_WEIGHT_SILHOUETTE, SCORE_WEIGHT_DAVIES_BOULDIN, SCORE_WEIGHT_CALINSKI_HARABASZ, \
    SCORE_WEIGHT_PURITY, SCORE_WEIGHT_OTHER_FEATURE_DIVERSITY, SCORE_WEIGHT_OTHER_FEATURE_PURITY, \
    MIN_SONGS_PER_GENRE_FOR_STRATIFICATION, STRATIFIED_SAMPLING_TARGET_PERCENTILE, \
    CLUSTER_ALGORITHM, NUM_CLUSTERS_MIN, NUM_CLUSTERS_MAX, DBSCAN_EPS_MIN, DBSCAN_EPS_MAX, GMM_COVARIANCE_TYPE, \
    DBSCAN_MIN_SAMPLES_MIN, DBSCAN_MIN_SAMPLES_MAX, GMM_N_COMPONENTS_MIN, GMM_N_COMPONENTS_MAX, \
    SPECTRAL_N_CLUSTERS_MIN, SPECTRAL_N_CLUSTERS_MAX, ENABLE_CLUSTERING_EMBEDDINGS, \
    PCA_COMPONENTS_MIN, PCA_COMPONENTS_MAX, CLUSTERING_RUNS, MOOD_LABELS, TOP_N_MOODS, \
    AI_MODEL_PROVIDER, OLLAMA_SERVER_URL, OLLAMA_MODEL_NAME, \
    OPENAI_SERVER_URL, OPENAI_MODEL_NAME, OPENAI_API_KEY, \
    GEMINI_API_KEY, GEMINI_MODEL_NAME, \
    TOP_N_PLAYLISTS, MISTRAL_API_KEY, MISTRAL_MODEL_NAME

# RQ import
from rq import Retry
from psycopg2.extras import DictCursor


logger = logging.getLogger(__name__)

# Create a Blueprint for clustering-related routes
clustering_bp = Blueprint('clustering_bp', __name__)

def clustering_task_failure_handler(job, connection, type, value, tb):
    """A failure handler for the main clustering task, executed by the worker."""
    from app import app
    from app_helper import save_task_status, TASK_STATUS_FAILURE
    with app.app_context():
        task_id = job.get_id()
        
        # --- FIX: Handle different traceback types, especially from rq-janitor ---
        tb_formatted = ""
        if isinstance(tb, traceback.StackSummary):
            tb_formatted = "".join(tb.format())
        else:
            tb_formatted = "".join(traceback.format_exception(type, value, tb))

        error_details = {
            "message": "Clustering task failed permanently after all retries.",
            "error_type": str(type.__name__),
            "error_value": str(value),
            "traceback": tb_formatted
        }
        save_task_status(
            task_id,
            "main_clustering",
            TASK_STATUS_FAILURE,
            progress=100,
            details=error_details
        )
        app.logger.error(f"Main clustering task {task_id} failed permanently. DB status updated.")

@clustering_bp.route('/api/clustering/start', methods=['POST'])
def start_clustering_endpoint():
    """
    Start the music clustering and playlist generation process.
    This endpoint enqueues a main clustering task.
    Note: Starting a new clustering task will archive previously successful tasks by setting their status to REVOKED.
    ---
    tags:
      - Clustering
    requestBody:
      description: Configuration for the clustering task.
      required: true
      content:
        application/json:
          schema:
            type: object
            properties:
              top_n_playlists:
                type: integer
                description: "If > 0, returns only the Top N most diverse playlists. If 0 or not provided, returns all. Defaults to value from config."
                default: "Configured TOP_N_PLAYLISTS"
              clustering_method:
                type: string
                description: Algorithm to use for clustering (e.g., kmeans, dbscan, gmm, spectral).
                default: "Configured CLUSTER_ALGORITHM"
              num_clusters_min:
                type: integer
                description: Minimum number of clusters (for kmeans/gmm).
                default: "Configured NUM_CLUSTERS_MIN"
              num_clusters_max:
                type: integer
                description: Maximum number of clusters (for kmeans/gmm).
                default: "Configured NUM_CLUSTERS_MAX"
              dbscan_eps_min:
                type: number
                format: float
                description: Minimum epsilon for DBSCAN.
                default: "Configured DBSCAN_EPS_MIN"
              dbscan_eps_max:
                type: number
                format: float
                description: Maximum epsilon for DBSCAN.
                default: "Configured DBSCAN_EPS_MAX"
              dbscan_min_samples_min:
                type: integer
                description: Minimum min_samples for DBSCAN.
                default: "Configured DBSCAN_MIN_SAMPLES_MIN"
              dbscan_min_samples_max:
                type: integer
                description: Maximum min_samples for DBSCAN.
                default: "Configured DBSCAN_MIN_SAMPLES_MAX"
              gmm_n_components_min:
                type: integer
                description: Minimum number of components for GMM.
                default: "Configured GMM_N_COMPONENTS_MIN"
              gmm_n_components_max:
                type: integer
                description: Maximum number of components for GMM.
                default: "Configured GMM_N_COMPONENTS_MAX"
              spectral_n_clusters_min:
                type: integer
                description: Minimum number of clusters for SpectralClustering.
                default: "Configured SPECTRAL_N_CLUSTERS_MIN"
              spectral_n_clusters_max:
                type: integer
                description: Maximum number of clusters for SpectralClustering.
                default: "Configured SPECTRAL_N_CLUSTERS_MAX"
              pca_components_min:
                type: integer
                description: Minimum number of PCA components.
                default: "Configured PCA_COMPONENTS_MIN"
              pca_components_max:
                type: integer
                description: Maximum number of PCA components.
                default: "Configured PCA_COMPONENTS_MAX"
              clustering_runs:
                type: integer
                description: Number of clustering iterations to perform.
                default: "Configured CLUSTERING_RUNS"
              score_weight_diversity:
                type: number
                format: float
                description: Weight for the inter-playlist mood diversity score component.
                default: "Configured SCORE_WEIGHT_DIVERSITY"
              score_weight_silhouette:
                type: number
                format: float
                description: Weight for the Silhouette score component.
                default: "Configured SCORE_WEIGHT_SILHOUETTE"
              score_weight_davies_bouldin:
                type: number
                format: float
                description: Weight for the Davies-Bouldin score component (higher is better for score calculation).
                default: "Configured SCORE_WEIGHT_DAVIES_BOULDIN"
              score_weight_calinski_harabasz:
                type: number
                format: float
                description: Weight for the Calinski-Harabasz score component (higher is better).
                default: "Configured SCORE_WEIGHT_CALINSKI_HARABASZ"
              score_weight_purity:
                type: number
                format: float
                description: Weight for playlist purity (intra-playlist mood consistency).
                default: "Configured SCORE_WEIGHT_PURITY"
              score_weight_other_feature_diversity:
                type: number
                format: float
                description: Weight for inter-playlist diversity of other features (e.g., danceability).
                default: "Configured SCORE_WEIGHT_OTHER_FEATURE_DIVERSITY"
              score_weight_other_feature_purity:
                type: number
                format: float
                description: Weight for intra-playlist consistency of other features (e.g., danceability).
                default: "Configured SCORE_WEIGHT_OTHER_FEATURE_PURITY"
              min_songs_per_genre_for_stratification:
                type: integer
                description: Minimum number of songs to target per stratified genre.
                default: "Configured MIN_SONGS_PER_GENRE_FOR_STRATIFICATION"
              stratified_sampling_target_percentile:
                type: integer
                description: Percentile of genre song counts to use for target songs per genre.
                minimum: 0
                maximum: 100
                default: "Configured STRATIFIED_SAMPLING_TARGET_PERCENTILE"
              max_songs_per_cluster:
                type: integer
                description: Maximum number of songs per generated playlist/cluster.
                default: "Configured MAX_SONGS_PER_CLUSTER"
              ai_model_provider:
                type: string
                description: AI provider for playlist naming (OLLAMA, GEMINI, MISTRAL, NONE).
                default: "Configured AI_MODEL_PROVIDER"
              ollama_server_url:
                type: string
                description: Override for the Ollama server URL for this run.
                nullable: true
                default: "Defaults to server-configured OLLAMA_SERVER_URL"
              ollama_model_name:
                type: string
                description: Override for the Ollama model name for this run.
                nullable: true
                default: "Defaults to server-configured OLLAMA_MODEL_NAME"
              gemini_api_key:
                type: string
                description: Override for the Gemini API key for this run (optional, defaults to server configuration).
                nullable: true
              gemini_model_name:
                type: string
                description: Override for the Gemini model name for this run.
                nullable: true
                default: "Defaults to server-configured GEMINI_MODEL_NAME"
              mistral_api_key:
                type: string
                description: Override for the Mistral API key for this run (optional, defaults to server configuration).
                nullable: true
              mistral_model_name:
                type: string
                description: Override for the Mistral model name for this run.
                nullable: true
                default: "Defaults to server-configured MISTRAL_MODEL_NAME"
              top_n_moods:
                type: integer
                description: Number of top moods to consider for clustering feature vectors (uses the first N from global MOOD_LABELS).
                default: "Configured TOP_N_MOODS"
              enable_clustering_embeddings:
                type: boolean
                description: Whether to use embeddings for clustering (True) or score_vector (False).
                default: true
    responses:
      202:
        description: Clustering task successfully enqueued.
        content:
          application/json:
            schema:
              type: object
              properties:
                task_id:
                  type: string
                  description: The ID of the enqueued main clustering task.
                task_type:
                  type: string
                  description: Type of the task (e.g., main_clustering).
                  example: main_clustering
      409:
        description: An active clustering task is already in progress.
        content:
            application/json:
                schema:
                    type: object
                    properties:
                        error:
                            type: string
                        task_id:
                            type: string
                        status:
                            type: string
    """
    # Local imports to prevent circular dependency at startup
    from app_helper import rq_queue_high, get_db
    from app_helper import (
        clean_up_previous_main_tasks,
        save_task_status,
        TASK_STATUS_PENDING,
        TASK_STATUS_STARTED,
        TASK_STATUS_PROGRESS,
        TASK_STATUS_SUCCESS,
        TASK_STATUS_FAILURE,
        TASK_STATUS_REVOKED
    )

    # Check for an existing active task to prevent parallel runs
    db = get_db()
    cur = db.cursor(cursor_factory=DictCursor)
    non_terminal_statuses = (TASK_STATUS_PENDING, TASK_STATUS_STARTED, TASK_STATUS_PROGRESS)
    cur.execute("""
        SELECT task_id, status FROM task_status 
        WHERE task_type = 'main_clustering' AND status IN %s
    """, (non_terminal_statuses,))
    active_task = cur.fetchone()
    cur.close()

    if active_task:
        return jsonify({
            "error": "An active clustering task is already in progress.",
            "task_id": active_task['task_id'],
            "status": active_task['status']
        }), 409

    data = request.json
    job_id = str(uuid.uuid4())

    # Clean up details of previously successful or stale tasks before starting a new one
    clean_up_previous_main_tasks()
    save_task_status(job_id, "main_clustering", TASK_STATUS_PENDING, details={"message": "Task enqueued."})

    job = rq_queue_high.enqueue(
        'tasks.clustering.run_clustering_task', # Enqueue by string path
        kwargs={ # Pass all arguments as a dictionary
            "clustering_method": data.get('clustering_method', CLUSTER_ALGORITHM),
            "num_clusters_min": int(data.get('num_clusters_min', NUM_CLUSTERS_MIN)),
            "num_clusters_max": int(data.get('num_clusters_max', NUM_CLUSTERS_MAX)),
            "dbscan_eps_min": float(data.get('dbscan_eps_min', DBSCAN_EPS_MIN)),
            "dbscan_eps_max": float(data.get('dbscan_eps_max', DBSCAN_EPS_MAX)),
            "dbscan_min_samples_min": int(data.get('dbscan_min_samples_min', DBSCAN_MIN_SAMPLES_MIN)),
            "dbscan_min_samples_max": int(data.get('dbscan_min_samples_max', DBSCAN_MIN_SAMPLES_MAX)),
            "gmm_n_components_min": int(data.get('gmm_n_components_min', GMM_N_COMPONENTS_MIN)),
            "gmm_n_components_max": int(data.get('gmm_n_components_max', GMM_N_COMPONENTS_MAX)),
            "spectral_n_clusters_min": int(data.get('spectral_n_clusters_min', SPECTRAL_N_CLUSTERS_MIN)),
            "spectral_n_clusters_max": int(data.get('spectral_n_clusters_max', SPECTRAL_N_CLUSTERS_MAX)),
            "pca_components_min": int(data.get('pca_components_min', PCA_COMPONENTS_MIN)),
            "pca_components_max": int(data.get('pca_components_max', PCA_COMPONENTS_MAX)),
            "num_clustering_runs": int(data.get('clustering_runs', CLUSTERING_RUNS)),
            "max_songs_per_cluster_val": int(data.get('max_songs_per_cluster', MAX_SONGS_PER_CLUSTER)),
            "top_n_playlists_param": int(data.get('top_n_playlists', TOP_N_PLAYLISTS)),
            "min_songs_per_genre_for_stratification_param": int(data.get('min_songs_per_genre_for_stratification', MIN_SONGS_PER_GENRE_FOR_STRATIFICATION)),
            "stratified_sampling_target_percentile_param": int(data.get('stratified_sampling_target_percentile', STRATIFIED_SAMPLING_TARGET_PERCENTILE)),
            "score_weight_diversity_param": float(data.get('score_weight_diversity', SCORE_WEIGHT_DIVERSITY)),
            "score_weight_silhouette_param": float(data.get('score_weight_silhouette', SCORE_WEIGHT_SILHOUETTE)),
            "score_weight_davies_bouldin_param": float(data.get('score_weight_davies_bouldin', SCORE_WEIGHT_DAVIES_BOULDIN)),
            "score_weight_calinski_harabasz_param": float(data.get('score_weight_calinski_harabasz', SCORE_WEIGHT_CALINSKI_HARABASZ)),
            "score_weight_purity_param": float(data.get('score_weight_purity', SCORE_WEIGHT_PURITY)),
            "score_weight_other_feature_diversity_param": float(data.get('score_weight_other_feature_diversity', SCORE_WEIGHT_OTHER_FEATURE_DIVERSITY)),
            "score_weight_other_feature_purity_param": float(data.get('score_weight_other_feature_purity', SCORE_WEIGHT_OTHER_FEATURE_PURITY)),
            "ai_model_provider_param": data.get('ai_model_provider', AI_MODEL_PROVIDER).upper(),
            "ollama_server_url_param": data.get('ollama_server_url', OLLAMA_SERVER_URL),
            "ollama_model_name_param": data.get('ollama_model_name', OLLAMA_MODEL_NAME),
            "openai_server_url_param": data.get('openai_server_url', OPENAI_SERVER_URL),
            "openai_model_name_param": data.get('openai_model_name', OPENAI_MODEL_NAME),
            "openai_api_key_param": data.get('openai_api_key') or OPENAI_API_KEY,  # Use env var if empty string
            # This line already falls back to the config value if the request doesn't contain it.
            "gemini_api_key_param": data.get('gemini_api_key', GEMINI_API_KEY),
            "gemini_model_name_param": data.get('gemini_model_name', GEMINI_MODEL_NAME),
            "mistral_api_key_param": data.get('mistral_api_key', MISTRAL_API_KEY),
            "mistral_model_name_param": data.get('mistral_model_name', MISTRAL_MODEL_NAME),
            "top_n_moods_for_clustering_param": int(data.get('top_n_moods', TOP_N_MOODS)),
            "enable_clustering_embeddings_param": data.get('enable_clustering_embeddings', ENABLE_CLUSTERING_EMBEDDINGS),
        },
        job_id=job_id,
        description="Main Music Clustering",
        retry=Retry(max=3),
        job_timeout=-1, # No timeout
        on_failure=clustering_task_failure_handler
    )
    return jsonify({"task_id": job.id, "task_type": "main_clustering", "status": job.get_status()}), 202
