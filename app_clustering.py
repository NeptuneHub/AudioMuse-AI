# app_clustering.py
from flask import Blueprint, jsonify, request
import uuid
import logging

# Import all necessary configuration variables
from config import JELLYFIN_URL, JELLYFIN_USER_ID, JELLYFIN_TOKEN, HEADERS, TEMP_DIR, \
    REDIS_URL, DATABASE_URL, MAX_DISTANCE, MAX_SONGS_PER_CLUSTER, MAX_SONGS_PER_ARTIST, NUM_RECENT_ALBUMS, \
    SCORE_WEIGHT_DIVERSITY, SCORE_WEIGHT_SILHOUETTE, SCORE_WEIGHT_DAVIES_BOULDIN, SCORE_WEIGHT_CALINSKI_HARABASZ, \
    SCORE_WEIGHT_PURITY, SCORE_WEIGHT_OTHER_FEATURE_DIVERSITY, SCORE_WEIGHT_OTHER_FEATURE_PURITY, \
    MIN_SONGS_PER_GENRE_FOR_STRATIFICATION, STRATIFIED_SAMPLING_TARGET_PERCENTILE, \
    CLUSTER_ALGORITHM, NUM_CLUSTERS_MIN, NUM_CLUSTERS_MAX, DBSCAN_EPS_MIN, DBSCAN_EPS_MAX, GMM_COVARIANCE_TYPE, \
    DBSCAN_MIN_SAMPLES_MIN, DBSCAN_MIN_SAMPLES_MAX, GMM_N_COMPONENTS_MIN, GMM_N_COMPONENTS_MAX, ENABLE_CLUSTERING_EMBEDDINGS, \
    PCA_COMPONENTS_MIN, PCA_COMPONENTS_MAX, CLUSTERING_RUNS, MOOD_LABELS, TOP_N_MOODS, \
    AI_MODEL_PROVIDER, OLLAMA_SERVER_URL, OLLAMA_MODEL_NAME, GEMINI_API_KEY, GEMINI_MODEL_NAME

# RQ import
from rq import Retry

logger = logging.getLogger(__name__)

# Create a Blueprint for clustering-related routes
clustering_bp = Blueprint('clustering_bp', __name__)

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
              clustering_method:
                type: string
                description: Algorithm to use for clustering (e.g., kmeans, dbscan, gmm).
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
                description: AI provider for playlist naming (OLLAMA, GEMINI, NONE).
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
                description: Override for the Gemini API key for this run.
                nullable: true
                default: "Defaults to server-configured GEMINI_API_KEY"
              gemini_model_name:
                type: string
                description: Override for the Gemini model name for this run.
                nullable: true
                default: "Defaults to server-configured GEMINI_MODEL_NAME"
              top_n_moods:
                type: integer
                description: Number of top moods to consider for clustering feature vectors (uses the first N from global MOOD_LABELS).
                default: "Configured TOP_N_MOODS"
              enable_clustering_embeddings:
                type: boolean
                description: Whether to use embeddings for clustering (True) or score_vector (False).
                default: false
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
                status:
                  type: string
                  description: The initial status of the job in the queue (e.g., queued).
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
    # Import app-level components here to avoid circular imports at module level
    from app import (
        rq_queue_high,
        clean_successful_task_details_on_new_start,
        save_task_status,
        TASK_STATUS_PENDING
    )

    # Enqueue by string to avoid circular imports
    data = request.json
    job_id = str(uuid.uuid4())

    # Clean up details of previously successful tasks before starting a new one
    clean_successful_task_details_on_new_start()
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
            "pca_components_min": int(data.get('pca_components_min', PCA_COMPONENTS_MIN)),
            "pca_components_max": int(data.get('pca_components_max', PCA_COMPONENTS_MAX)),
            "num_clustering_runs": int(data.get('clustering_runs', CLUSTERING_RUNS)),
            "max_songs_per_cluster_val": int(data.get('max_songs_per_cluster', MAX_SONGS_PER_CLUSTER)),
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
            "gemini_api_key_param": data.get('gemini_api_key', GEMINI_API_KEY),
            "gemini_model_name_param": data.get('gemini_model_name', GEMINI_MODEL_NAME),
            "top_n_moods_for_clustering_param": int(data.get('top_n_moods', TOP_N_MOODS)),
            "enable_clustering_embeddings_param": data.get('enable_clustering_embeddings', ENABLE_CLUSTERING_EMBEDDINGS),
        },
        job_id=job_id,
        description="Main Music Clustering",
        retry=Retry(max=3),
        job_timeout=-1 # No timeout
    )
    return jsonify({"task_id": job.id, "task_type": "main_clustering", "status": job.get_status()}), 202
