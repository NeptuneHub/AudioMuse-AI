import os
import psycopg2
from psycopg2.extras import DictCursor
from flask import Flask, jsonify, request, render_template, g, current_app, url_for
import json
import logging
import threading
import uuid # For generating job IDs if needed directly in API, though tasks handle their own
import time

# RQ imports
from rq.job import Job, JobStatus
from rq.exceptions import NoSuchJobError

# Redis client
from redis import Redis

# Swagger imports
from flasgger import Swagger, swag_from

# Import configuration
from config import JELLYFIN_URL, JELLYFIN_USER_ID, JELLYFIN_TOKEN, HEADERS, TEMP_DIR, \
  REDIS_URL, DATABASE_URL, MAX_DISTANCE, MAX_SONGS_PER_CLUSTER, MAX_SONGS_PER_ARTIST, NUM_RECENT_ALBUMS, \
  SCORE_WEIGHT_DIVERSITY, SCORE_WEIGHT_SILHOUETTE, SCORE_WEIGHT_DAVIES_BOULDIN, SCORE_WEIGHT_CALINSKI_HARABASZ, \
  SCORE_WEIGHT_PURITY, SCORE_WEIGHT_OTHER_FEATURE_DIVERSITY, SCORE_WEIGHT_OTHER_FEATURE_PURITY, \
  MIN_SONGS_PER_GENRE_FOR_STRATIFICATION, STRATIFIED_SAMPLING_TARGET_PERCENTILE, \
  CLUSTER_ALGORITHM, NUM_CLUSTERS_MIN, NUM_CLUSTERS_MAX, DBSCAN_EPS_MIN, DBSCAN_EPS_MAX, GMM_COVARIANCE_TYPE, \
  DBSCAN_MIN_SAMPLES_MIN, DBSCAN_MIN_SAMPLES_MAX, GMM_N_COMPONENTS_MIN, GMM_N_COMPONENTS_MAX, \
  SPECTRAL_N_CLUSTERS_MIN, SPECTRAL_N_CLUSTERS_MAX, ENABLE_CLUSTERING_EMBEDDINGS, \
  PCA_COMPONENTS_MIN, PCA_COMPONENTS_MAX, CLUSTERING_RUNS, MOOD_LABELS, TOP_N_MOODS, APP_VERSION, \
  AI_MODEL_PROVIDER, OLLAMA_SERVER_URL, OLLAMA_MODEL_NAME, OPENAI_SERVER_URL, OPENAI_MODEL_NAME, GEMINI_API_KEY, GEMINI_MODEL_NAME, MISTRAL_MODEL_NAME, \
  TOP_N_PLAYLISTS, PATH_DISTANCE_METRIC, ALCHEMY_DEFAULT_N_RESULTS, ALCHEMY_MAX_N_RESULTS, ALCHEMY_SUBTRACT_DISTANCE, \
  ENABLE_PROXY_FIX, \
  ALCHEMY_SUBTRACT_DISTANCE_ANGULAR, ALCHEMY_SUBTRACT_DISTANCE_EUCLIDEAN  # --- NEW: Import path distance metric and alchemy defaults ---

if ENABLE_PROXY_FIX:
  # Werkzeug import for reverse proxy support
  from werkzeug.middleware.proxy_fix import ProxyFix

# --- Flask App Setup ---
app = Flask(__name__)

# Import helper functions
from app_helper import (
    init_db, get_db, close_db,
    redis_conn, rq_queue_high, rq_queue_default,
    clean_up_previous_main_tasks,
    save_task_status,
    get_task_info_from_db,
    cancel_job_and_children_recursive,
    TASK_STATUS_PENDING, TASK_STATUS_STARTED, TASK_STATUS_PROGRESS,
    TASK_STATUS_SUCCESS, TASK_STATUS_FAILURE, TASK_STATUS_REVOKED
)

# NOTE: Annoy Manager import is moved to be local where used to prevent circular imports.

logger = logging.getLogger(__name__)

# Configure basic logging for the entire application
logging.basicConfig(
    level=logging.INFO, # Set the default logging level (e.g., INFO, DEBUG, WARNING, ERROR, CRITICAL)
    format='[%(levelname)s]-[%(asctime)s]-%(message)s', # Custom format string
    datefmt='%d-%m-%Y %H-%M-%S' # Custom date/time format
)

if ENABLE_PROXY_FIX:
  app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)

# Log the application version on startup
app.logger.info(f"Starting AudioMuse-AI Backend version {APP_VERSION}")

# --- Context Processor to Inject Version ---
@app.context_processor
def inject_version():
    """Injects the app version into all templates."""
    return dict(app_version=APP_VERSION)

# --- Swagger Setup ---
app.config['SWAGGER'] = {
    'title': 'AudioMuse-AI API',
    'uiversion': 3,
    'openapi': '3.0.0'
}
swagger = Swagger(app)

@app.teardown_appcontext
def teardown_db(e=None):
    close_db(e)

# Initialize the database schema when the application module is loaded.
# This is safe because it doesn't import other application modules.
with app.app_context():
    init_db()


# --- API Endpoints ---

@app.route('/')
def index():
    """
    Serve the main HTML page.
    ---
    tags:
      - UI
    responses:
      200:
        description: HTML content of the main page.
        content:
          text/html:
            schema:
              type: string
    """
    return render_template('index.html', title = 'AudioMuse-AI - Home Page', active='index')


@app.route('/api/status/<task_id>', methods=['GET'])
def get_task_status_endpoint(task_id):
    """
    Get the status of a specific task.
    Retrieves status information from both RQ and the database.
    ---
    tags:
      - Status
    parameters:
      - name: task_id
        in: path
        required: true
        description: The ID of the task.
        schema:
          type: string
    responses:
      200:
        description: Status information for the task.
        content:
          application/json:
            schema:
              type: object
              properties:
                task_id:
                  type: string
                state:
                  type: string
                  description: Current state of the task (e.g., PENDING, STARTED, PROGRESS, SUCCESS, FAILURE, REVOKED, queued, finished, failed, canceled).
                status_message:
                  type: string
                  description: A human-readable status message.
                progress:
                  type: integer
                  description: Task progress percentage (0-100).
                running_time_seconds:
                  type: number
                  description: The total running time of the task in seconds. Updates live for running tasks.
                details:
                  type: object
                  description: Detailed information about the task. Structure varies by task type and state.
                  additionalProperties: true
                  example: {"log": ["Log message 1"], "current_album": "Album X"}
                task_type_from_db:
                  type: string
                  nullable: true
                  description: The type of the task as recorded in the database (e.g., main_analysis, album_analysis, main_clustering, clustering_batch).
      404:
        description: Task ID not found in RQ or database.
        content:
          application/json:
            schema:
              type: object
              properties:
                task_id:
                  type: string
                state:
                  type: string
                  example: UNKNOWN
                status_message:
                  type: string
                  example: Task ID not found in RQ or DB.
    """
    response = {'task_id': task_id, 'state': 'UNKNOWN', 'status_message': 'Task ID not found in RQ or DB.', 'progress': 0, 'details': {}, 'task_type_from_db': None, 'running_time_seconds': 0}
    try:
        job = Job.fetch(task_id, connection=redis_conn)
        response['state'] = job.get_status() # e.g., queued, started, finished, failed
        response['status_message'] = job.meta.get('status_message', response['state'])
        response['progress'] = job.meta.get('progress', 0)
        response['details'] = job.meta.get('details', {})
        if job.is_failed:
            response['details']['error_message'] = job.exc_info if job.exc_info else "Job failed without error info."
            response['status_message'] = "FAILED"
        elif job.is_finished:
             response['status_message'] = "SUCCESS" # RQ uses 'finished' for success
             response['progress'] = 100
        elif job.is_canceled:
            response['status_message'] = "CANCELED"
            response['progress'] = 100

    except NoSuchJobError:
        # If not in RQ, it might have been cleared or never existed. Check DB.
        pass # Will fall through to DB check

    # Augment with DB data, DB is source of truth for persisted details
    db_task_info = get_task_info_from_db(task_id)
    if db_task_info:
        response['task_type_from_db'] = db_task_info.get('task_type')
        response['running_time_seconds'] = db_task_info.get('running_time_seconds', 0)
        # If RQ state is more final (e.g. failed/finished), prefer that, else use DB
        if response['state'] not in [JobStatus.FINISHED, JobStatus.FAILED, JobStatus.CANCELED]:
            response['state'] = db_task_info.get('status', response['state']) # Use DB status if RQ is still active

        response['progress'] = db_task_info.get('progress', response['progress'])
        db_details = json.loads(db_task_info.get('details')) if db_task_info.get('details') else {}
        # Merge details: RQ meta (live) can override DB details (persisted)
        response['details'] = {**db_details, **response['details']}

        # If task is marked REVOKED in DB, this is the most accurate status for cancellation
        if db_task_info.get('status') == TASK_STATUS_REVOKED:
            response['state'] = 'REVOKED'
            response['status_message'] = 'Task revoked.'
            response['progress'] = 100
    elif response['state'] == 'UNKNOWN': # Not in RQ and not in DB
        return jsonify(response), 404

    # Prune 'checked_album_ids' from details if the task is analysis-related
    if response.get('task_type_from_db') and 'analysis' in response['task_type_from_db']:
        if isinstance(response.get('details'), dict):
            response['details'].pop('checked_album_ids', None)
    
    # Truncate log entries to last 10 entries for all task types
    if isinstance(response.get('details'), dict) and 'log' in response['details']:
        log_entries = response['details']['log']
        if isinstance(log_entries, list) and len(log_entries) > 10:
            response['details']['log'] = [
                f"... ({len(log_entries) - 10} earlier log entries truncated)",
                *log_entries[-10:]
            ]
    
    # Clean up the final response to remove confusing raw time columns
    response.pop('timestamp', None)
    response.pop('start_time', None)
    response.pop('end_time', None)

    return jsonify(response)

@app.route('/api/cancel/<task_id>', methods=['POST'])
def cancel_task_endpoint(task_id):
    """
    Cancel a specific task and its children.
    Marks the task and its descendants as REVOKED in the database and attempts to stop/cancel them in RQ.
    ---
    tags:
      - Control
    parameters:
      - name: task_id
        in: path
        required: true
        description: The ID of the task.
        schema:
          type: string
    responses:
      200:
        description: Cancellation initiated for the task and its children.
        content:
          application/json:
            schema:
              type: object
              properties:
                message:
                  type: string
                task_id:
                  type: string
                cancelled_jobs_count:
                  type: integer
      400:
        description: Task could not be cancelled (e.g., already completed or not in an active state).
      404:
        description: Task ID not found in the database.
    """
    db_task_info = get_task_info_from_db(task_id)
    if not db_task_info:
        return jsonify({"message": f"Task {task_id} not found in database.", "task_id": task_id}), 404

    if db_task_info.get('status') in [TASK_STATUS_SUCCESS, TASK_STATUS_FAILURE, TASK_STATUS_REVOKED]:
        return jsonify({"message": f"Task {task_id} is already in a terminal state ({db_task_info.get('status')}) and cannot be cancelled.", "task_id": task_id}), 400

    cancelled_count = cancel_job_and_children_recursive(task_id, reason=f"Cancellation requested for task {task_id} via API.")

    if cancelled_count > 0:
        return jsonify({"message": f"Task {task_id} and its children cancellation initiated. {cancelled_count} total jobs affected.", "task_id": task_id, "cancelled_jobs_count": cancelled_count}), 200
    return jsonify({"message": "Task could not be cancelled (e.g., already completed or not found in active state).", "task_id": task_id}), 400


@app.route('/api/cancel_all/<task_type_prefix>', methods=['POST'])
def cancel_all_tasks_by_type_endpoint(task_type_prefix):
    """
    Cancel all active tasks of a specific type (e.g., main_analysis, main_clustering) and their children.
    ---
    tags:
      - Control
    parameters:
      - name: task_type_prefix
        in: path
        required: true
        description: The type of main tasks to cancel (e.g., "main_analysis", "main_clustering").
        schema:
          type: string
    responses:
      200:
        description: Cancellation initiated for all matching active tasks and their children.
        content:
          application/json:
            schema:
              type: object
              properties:
                message:
                  type: string
                cancelled_main_tasks:
                  type: array
                  items:
                    type: string
      404:
        description: No active tasks of the specified type found to cancel.
    """
    db = get_db()
    cur = db.cursor(cursor_factory=DictCursor)
    # Exclude terminal statuses
    terminal_statuses = (TASK_STATUS_SUCCESS, TASK_STATUS_FAILURE, TASK_STATUS_REVOKED)
    cur.execute("SELECT task_id, task_type FROM task_status WHERE task_type = %s AND status NOT IN %s", (task_type_prefix, terminal_statuses))
    tasks_to_cancel = cur.fetchall()
    cur.close()

    total_cancelled_jobs = 0
    cancelled_main_task_ids = []
    for task_row in tasks_to_cancel:
        cancelled_jobs_for_this_main_task = cancel_job_and_children_recursive(task_row['task_id'], reason=f"Bulk cancellation for task type '{task_type_prefix}' via API.")
        if cancelled_jobs_for_this_main_task > 0:
           total_cancelled_jobs += cancelled_jobs_for_this_main_task
           cancelled_main_task_ids.append(task_row['task_id'])

    if total_cancelled_jobs > 0:
        return jsonify({"message": f"Cancellation initiated for {len(cancelled_main_task_ids)} main tasks of type '{task_type_prefix}' and their children. Total jobs affected: {total_cancelled_jobs}.", "cancelled_main_tasks": cancelled_main_task_ids}), 200
    return jsonify({"message": f"No active tasks of type '{task_type_prefix}' found to cancel."}), 404

@app.route('/api/last_task', methods=['GET'])
def get_last_overall_task_status_endpoint():
    """
    Get the status of the most recent overall main task (analysis, clustering, or cleaning).
    """
    db = get_db()
    cur = db.cursor(cursor_factory=DictCursor)
    cur.execute("""
        SELECT task_id, task_type, status, progress, details, start_time, end_time
        FROM task_status 
        WHERE parent_task_id IS NULL 
        ORDER BY timestamp DESC 
        LIMIT 1
    """)
    last_task_row = cur.fetchone()
    cur.close()

    if last_task_row:
        last_task_data = dict(last_task_row)
        if last_task_data.get('details'):
            try: last_task_data['details'] = json.loads(last_task_data['details'])
            except json.JSONDecodeError: pass

        # Calculate running time in Python
        start_time = last_task_data.get('start_time')
        end_time = last_task_data.get('end_time')
        if start_time:
            effective_end_time = end_time if end_time is not None else time.time()
            last_task_data['running_time_seconds'] = max(0, effective_end_time - start_time)
        else:
            last_task_data['running_time_seconds'] = 0.0
        
        # Truncate log entries to last 10 entries
        if isinstance(last_task_data.get('details'), dict) and 'log' in last_task_data['details']:
            log_entries = last_task_data['details']['log']
            if isinstance(log_entries, list) and len(log_entries) > 10:
                last_task_data['details']['log'] = [
                    f"... ({len(log_entries) - 10} earlier log entries truncated)",
                    *log_entries[-10:]
                ]
        
        # Clean up raw time columns before sending response
        last_task_data.pop('start_time', None)
        last_task_data.pop('end_time', None)
        last_task_data.pop('timestamp', None)

        return jsonify(last_task_data), 200
        
    return jsonify({"task_id": None, "task_type": None, "status": "NO_PREVIOUS_MAIN_TASK", "details": {"log": ["No previous main task found."] }}), 200

@app.route('/api/active_tasks', methods=['GET'])
def get_active_tasks_endpoint():
    """
    Get the status of the currently active main task, if any.
    """
    db = get_db()
    cur = db.cursor(cursor_factory=DictCursor)
    non_terminal_statuses = (TASK_STATUS_PENDING, TASK_STATUS_STARTED, TASK_STATUS_PROGRESS)
    cur.execute("""
        SELECT task_id, parent_task_id, task_type, sub_type_identifier, status, progress, details, start_time, end_time
        FROM task_status
        WHERE parent_task_id IS NULL AND status IN %s
        ORDER BY timestamp DESC
        LIMIT 1
    """, (non_terminal_statuses,))
    active_main_task_row = cur.fetchone()
    cur.close()

    if active_main_task_row:
        task_item = dict(active_main_task_row)
        
        # Calculate running time in Python
        start_time = task_item.get('start_time')
        if start_time:
            task_item['running_time_seconds'] = max(0, time.time() - start_time)
        else:
            task_item['running_time_seconds'] = 0.0

        if task_item.get('details'):
            try:
                task_item['details'] = json.loads(task_item['details'])
                # Prune specific large or internal keys from details
                if isinstance(task_item['details'], dict):
                    task_item['details'].pop('clustering_run_job_ids', None)
                    task_item['details'].pop('checked_album_ids', None)
                    if 'best_params' in task_item['details'] and \
                       isinstance(task_item['details']['best_params'], dict) and \
                       'clustering_method_config' in task_item['details']['best_params'] and \
                       isinstance(task_item['details']['best_params']['clustering_method_config'], dict) and \
                       'params' in task_item['details']['best_params']['clustering_method_config']['params'] and \
                       isinstance(task_item['details']['best_params']['clustering_method_config']['params'], dict):
                        task_item['details']['best_params']['clustering_method_config']['params'].pop('initial_centroids', None)

            except json.JSONDecodeError:
                task_item['details'] = {"raw_details": task_item['details'], "error": "Failed to parse details JSON."}

        # Clean up raw time columns before sending response
        task_item.pop('start_time', None)
        task_item.pop('end_time', None)
        task_item.pop('timestamp', None)

        return jsonify(task_item), 200
    return jsonify({}), 200 # Return empty object if no active main task

@app.route('/api/config', methods=['GET'])
def get_config_endpoint():
    """
    Get the current server configuration values.
    """
    return jsonify({
        "num_recent_albums": NUM_RECENT_ALBUMS, "max_distance": MAX_DISTANCE,
        "max_songs_per_cluster": MAX_SONGS_PER_CLUSTER, "max_songs_per_artist": MAX_SONGS_PER_ARTIST,
        "cluster_algorithm": CLUSTER_ALGORITHM, "num_clusters_min": NUM_CLUSTERS_MIN, "num_clusters_max": NUM_CLUSTERS_MAX,
        "dbscan_eps_min": DBSCAN_EPS_MIN, "dbscan_eps_max": DBSCAN_EPS_MAX, "gmm_covariance_type": GMM_COVARIANCE_TYPE,
        "dbscan_min_samples_min": DBSCAN_MIN_SAMPLES_MIN, "dbscan_min_samples_max": DBSCAN_MIN_SAMPLES_MAX,
        "gmm_n_components_min": GMM_N_COMPONENTS_MIN, "gmm_n_components_max": GMM_N_COMPONENTS_MAX,
        "spectral_n_clusters_min": SPECTRAL_N_CLUSTERS_MIN, "spectral_n_clusters_max": SPECTRAL_N_CLUSTERS_MAX,
        "pca_components_min": PCA_COMPONENTS_MIN, "pca_components_max": PCA_COMPONENTS_MAX,
        "min_songs_per_genre_for_stratification": MIN_SONGS_PER_GENRE_FOR_STRATIFICATION,
        "stratified_sampling_target_percentile": STRATIFIED_SAMPLING_TARGET_PERCENTILE,
        "ai_model_provider": AI_MODEL_PROVIDER,
        "ollama_server_url": OLLAMA_SERVER_URL, "ollama_model_name": OLLAMA_MODEL_NAME,
        "openai_server_url": OPENAI_SERVER_URL, "openai_model_name": OPENAI_MODEL_NAME,
        "gemini_model_name": GEMINI_MODEL_NAME,
        "mistral_model_name": MISTRAL_MODEL_NAME,
        "top_n_moods": TOP_N_MOODS, "mood_labels": MOOD_LABELS, "clustering_runs": CLUSTERING_RUNS,
        "top_n_playlists": TOP_N_PLAYLISTS,
        "enable_clustering_embeddings": ENABLE_CLUSTERING_EMBEDDINGS,
        "score_weight_diversity": SCORE_WEIGHT_DIVERSITY,
        "score_weight_silhouette": SCORE_WEIGHT_SILHOUETTE,
        "score_weight_davies_bouldin": SCORE_WEIGHT_DAVIES_BOULDIN,
        "score_weight_calinski_harabasz": SCORE_WEIGHT_CALINSKI_HARABASZ,
        "score_weight_purity": SCORE_WEIGHT_PURITY,
        "score_weight_other_feature_diversity": SCORE_WEIGHT_OTHER_FEATURE_DIVERSITY,
        "score_weight_other_feature_purity": SCORE_WEIGHT_OTHER_FEATURE_PURITY,
        "path_distance_metric": PATH_DISTANCE_METRIC
      ,"alchemy_default_n_results": ALCHEMY_DEFAULT_N_RESULTS
      ,"alchemy_max_n_results": ALCHEMY_MAX_N_RESULTS
      ,"alchemy_subtract_distance": ALCHEMY_SUBTRACT_DISTANCE
      ,"alchemy_subtract_distance_angular": ALCHEMY_SUBTRACT_DISTANCE_ANGULAR
      ,"alchemy_subtract_distance_euclid": ALCHEMY_SUBTRACT_DISTANCE_EUCLIDEAN
    })

@app.route('/api/playlists', methods=['GET'])
def get_playlists_endpoint():
    """
    Get all generated playlists and their tracks from the database.
    """
    from collections import defaultdict # Local import if not used elsewhere globally
    conn = get_db()
    cur = conn.cursor(cursor_factory=DictCursor)
    cur.execute("SELECT playlist_name, item_id, title, author FROM playlist ORDER BY playlist_name")
    rows = cur.fetchall()
    cur.close()
    playlists_data = defaultdict(list)
    for row in rows:
        playlists_data[row['playlist_name']].append({"item_id": row['item_id'], "title": row['title'], "author": row['author']})
    return jsonify(dict(playlists_data)), 200


# --- Redis index reload listener (restored pre-e308673 logic, with map reload added) ---
def listen_for_index_reloads():
  """
  Runs in a background thread to listen for messages on a Redis Pub/Sub channel.
  When a 'reload' message is received, it triggers the in-memory Voyager index and map to be reloaded.
  This is the recommended pattern for inter-process communication in this architecture,
  avoiding direct HTTP calls from workers to the web server.
  """
  # Create a new Redis connection for this thread.
  # Sharing the main redis_conn object across threads is not recommended.
  from redis import Redis
  thread_redis_conn = Redis.from_url(REDIS_URL)
  pubsub = thread_redis_conn.pubsub()
  pubsub.subscribe('index-updates')
  logger.info("Background thread started. Listening for Voyager index reloads on Redis channel 'index-updates'.")

  for message in pubsub.listen():
    # The first message is a confirmation of subscription, so we skip it.
    if message['type'] == 'message':
      message_data = message['data'].decode('utf-8')
      logger.info(f"Received '{message_data}' message on 'index-updates' channel.")
      if message_data == 'reload':
        # We need the application context to access 'g' and the database connection.
        with app.app_context():
          logger.info("Triggering in-memory Voyager index and map reload from background listener.")
          try:
            from tasks.voyager_manager import load_voyager_index_for_querying
            load_voyager_index_for_querying(force_reload=True)
            from tasks.artist_gmm_manager import load_artist_index_for_querying
            load_artist_index_for_querying(force_reload=True)
            from app_helper import load_map_projection, load_artist_projection
            load_map_projection('main_map', force_reload=True)
            load_artist_projection('artist_map', force_reload=True)
            # Rebuild the map JSON cache used by the /api/map endpoint
            from app_map import build_map_cache
            build_map_cache()
            # Reload CLAP cache
            from tasks.clap_text_search import refresh_clap_cache
            refresh_clap_cache()
            logger.info("In-memory Voyager index, artist index, map projections, and CLAP cache reloaded successfully by background listener.")
          except Exception as e:
            logger.error(f"Error reloading indexes/maps from background listener: {e}", exc_info=True)
      elif message_data == 'reload-artist':
        # Reload artist similarity index only (legacy support)
        with app.app_context():
          logger.info("Triggering in-memory artist similarity index reload from background listener.")
          try:
            from tasks.artist_gmm_manager import load_artist_index_for_querying
            load_artist_index_for_querying(force_reload=True)
            logger.info("In-memory artist similarity index reloaded successfully by background listener.")
          except Exception as e:
            logger.error(f"Error reloading artist similarity index from background listener: {e}", exc_info=True)





# --- Import and Register Blueprints ---
# This is the original, working structure.
from app_helper import get_child_tasks_from_db, get_score_data_by_ids, get_tracks_by_ids, save_track_analysis_and_embedding, track_exists, update_playlist_table

# Import tasks modules to ensure they're available to RQ workers
import tasks.clustering
import tasks.analysis


from app_chat import chat_bp
from app_clustering import clustering_bp
from app_analysis import analysis_bp
from app_cron import cron_bp, run_due_cron_jobs
from app_voyager import voyager_bp
from app_sonic_fingerprint import sonic_fingerprint_bp
from app_path import path_bp
from app_collection import collection_bp
from app_external import external_bp # --- NEW: Import the external blueprint ---
from app_alchemy import alchemy_bp
from app_map import map_bp
from app_waveform import waveform_bp
from app_artist_similarity import artist_similarity_bp
from app_clap_search import clap_search_bp

app.register_blueprint(chat_bp, url_prefix='/chat')
app.register_blueprint(clustering_bp)
app.register_blueprint(analysis_bp)
app.register_blueprint(cron_bp)
app.register_blueprint(voyager_bp)
app.register_blueprint(sonic_fingerprint_bp)
app.register_blueprint(path_bp)
app.register_blueprint(collection_bp)
app.register_blueprint(external_bp, url_prefix='/external') # --- NEW: Register the external blueprint ---
app.register_blueprint(alchemy_bp)
app.register_blueprint(map_bp)
app.register_blueprint(waveform_bp)
app.register_blueprint(artist_similarity_bp)
app.register_blueprint(clap_search_bp)

if __name__ == '__main__':
  os.makedirs(TEMP_DIR, exist_ok=True)

  with app.app_context():
    # --- Initial Voyager Index Load ---
    from tasks.voyager_manager import load_voyager_index_for_querying
    load_voyager_index_for_querying()
    # --- Load Artist Similarity Index ---
    from tasks.artist_gmm_manager import load_artist_index_for_querying
    try:
      load_artist_index_for_querying()
      logger.info("Artist similarity index loaded at startup.")
    except Exception as e:
      logger.warning(f"Failed to load artist similarity index at startup: {e}")
    # Also try to load precomputed map projection into memory if available
    try:
      from app_helper import load_map_projection
      load_map_projection('main_map')
      logger.info("In-memory map projection loaded at startup.")
    except Exception as e:
      logger.debug(f"No precomputed map projection to load at startup or load failed: {e}")
    # Also try to load artist component projection into memory
    try:
      from app_helper import load_artist_projection
      load_artist_projection('artist_map')
      logger.info("In-memory artist component projection loaded at startup.")
    except Exception as e:
      logger.debug(f"No precomputed artist projection to load at startup or load failed: {e}")
    # Load CLAP text search cache
    try:
      from tasks.clap_text_search import load_clap_cache_from_db
      if load_clap_cache_from_db():
        logger.info("CLAP text search cache loaded at startup.")
    except Exception as e:
      logger.debug(f"CLAP cache not loaded at startup (may be disabled or no embeddings): {e}")
    # Initialize map JSON cache once at startup (reads DB one time)
    # Run this in a background daemon thread so the Flask process doesn't block on the heavy DB read.
    def _start_map_init_background():
      try:
        from app_map import init_map_cache
        logger.info('Starting background map JSON cache build.')
        # Ensure we run the heavy cache build inside an application context
        with app.app_context():
          init_map_cache()
        logger.info('Background map JSON cache build finished.')
      except Exception:
        logger.exception('Background init_map_cache failed')

    t = threading.Thread(target=_start_map_init_background, daemon=True)
    t.start()

  # --- Start Background Listener Thread ---
  listener_thread = threading.Thread(target=listen_for_index_reloads, daemon=True)
  listener_thread.start()

  # Start a cron manager thread that checks enabled cron entries every 60 seconds
  def _cron_manager_loop():
    try:
      from time import sleep
      while True:
        try:
          with app.app_context():
            run_due_cron_jobs()
        except Exception:
          app.logger.exception('cron manager failed')
        sleep(60)
    except Exception:
      app.logger.exception('cron manager main loop error')

  cron_thread = threading.Thread(target=_cron_manager_loop, daemon=True)
  cron_thread.start()

  app.run(debug=False, host='0.0.0.0', port=8000)
