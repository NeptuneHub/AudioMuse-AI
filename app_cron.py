from flask import Blueprint, render_template, jsonify, request
from psycopg2.extras import DictCursor
from app_helper import get_db, rq_queue_high, save_task_status, TASK_STATUS_PENDING
import uuid, time, logging
import config

cron_bp = Blueprint('cron_bp', __name__)


@cron_bp.route('/cron')
def cron_page():
    return render_template('cron.html', title = 'AudioMuse-AI - Scheduled Tasks', active='cron')


@cron_bp.route('/api/cron', methods=['GET'])
def get_cron_entries():
    db = get_db()
    cur = db.cursor(cursor_factory=DictCursor)
    cur.execute("SELECT id, name, task_type, cron_expr, enabled, last_run, created_at FROM cron ORDER BY id")
    rows = cur.fetchall()
    cur.close()
    entries = []
    for r in rows:
        entries.append({
            'id': r['id'], 'name': r['name'], 'task_type': r['task_type'], 'cron_expr': r['cron_expr'],
            'enabled': bool(r['enabled']), 'last_run': r['last_run'], 'created_at': str(r['created_at'])
        })
    # Remove the special-case append for sonic_fingerprint; now handled by DB init
    return jsonify(entries), 200


@cron_bp.route('/api/cron', methods=['POST'])
def save_cron_entry():
    data = request.json or {}
    # Expected fields: id (optional), name, task_type, cron_expr, enabled
    db = get_db()
    cur = db.cursor()
    if data.get('id'):
        cur.execute("UPDATE cron SET name=%s, task_type=%s, cron_expr=%s, enabled=%s WHERE id=%s", (
            data.get('name'), data.get('task_type'), data.get('cron_expr'), bool(data.get('enabled')), data.get('id')
        ))
    else:
        cur.execute("INSERT INTO cron (name, task_type, cron_expr, enabled) VALUES (%s,%s,%s,%s)", (
            data.get('name'), data.get('task_type'), data.get('cron_expr'), bool(data.get('enabled'))
        ))
    db.commit()
    cur.close()
    return jsonify({'message': 'saved'}), 200


def _field_matches(field_expr, value):
    # very small cron field matcher supporting '*', single number, list (comma), and ranges (a-b)
    if field_expr.strip() == '*':
        return True
    parts = field_expr.split(',')
    for p in parts:
        p = p.strip()
        if '-' in p:
            a, b = p.split('-', 1)
            try:
                if int(a) <= value <= int(b):
                    return True
            except ValueError:
                continue
        else:
            try:
                if int(p) == value:
                    return True
            except ValueError:
                continue
    return False


def cron_matches_now(expr, ts=None):
    # expr expected as 'min hour day month dow'
    t = time.localtime(ts) if ts else time.localtime()
    parts = expr.strip().split()
    if len(parts) < 5:
        return False
    minute, hour, dom, month, dow = parts[:5]
    if not _field_matches(minute, t.tm_min):
        return False
    if not _field_matches(hour, t.tm_hour):
        return False
    # day of week: in cron 0=Sun..6=Sat, Python tm_wday 0=Mon..6=Sun -> convert
    py_dow = (t.tm_wday + 1) % 7
    if not _field_matches(dow, py_dow):
        return False
    return True


def run_due_cron_jobs():
    """Read enabled cron rows and enqueue analysis/clustering/sonic_fingerprint when cron matches now and not recently run."""
    logger = logging.getLogger(__name__)
    db = get_db()
    cur = db.cursor(cursor_factory=DictCursor)
    cur.execute("SELECT id, name, task_type, cron_expr, enabled, last_run FROM cron WHERE enabled = true")
    rows = cur.fetchall()
    now_ts = time.time()
    for r in rows:
        try:
            last_run = r['last_run'] or 0
            # avoid duplicate runs within 55 seconds
            if now_ts - float(last_run) < 55:
                continue
            if cron_matches_now(r['cron_expr'], now_ts):
                task_type = r['task_type']
                job_id = str(uuid.uuid4())
                if task_type == 'analysis':
                    # mark queued in task_status
                    save_task_status(job_id, f"main_{task_type}", TASK_STATUS_PENDING, details={"message": "Enqueued by cron."})
                    rq_queue_high.enqueue('tasks.analysis.run_analysis_task', args=(0, config.TOP_N_MOODS), job_id=job_id, description='Cron Analysis', job_timeout=-1)
                    logger.info(f"Cron: enqueued analysis job {job_id}")
                elif task_type == 'clustering':
                    # mark queued in task_status
                    save_task_status(job_id, f"main_{task_type}", TASK_STATUS_PENDING, details={"message": "Enqueued by cron."})
                    clustering_kwargs = {
                        "clustering_method": config.CLUSTER_ALGORITHM,
                        "num_clusters_min": int(config.NUM_CLUSTERS_MIN),
                        "num_clusters_max": int(config.NUM_CLUSTERS_MAX),
                        "dbscan_eps_min": float(config.DBSCAN_EPS_MIN),
                        "dbscan_eps_max": float(config.DBSCAN_EPS_MAX),
                        "dbscan_min_samples_min": int(config.DBSCAN_MIN_SAMPLES_MIN),
                        "dbscan_min_samples_max": int(config.DBSCAN_MIN_SAMPLES_MAX),
                        "gmm_n_components_min": int(config.GMM_N_COMPONENTS_MIN),
                        "gmm_n_components_max": int(config.GMM_N_COMPONENTS_MAX),
                        "spectral_n_clusters_min": int(config.SPECTRAL_N_CLUSTERS_MIN),
                        "spectral_n_clusters_max": int(config.SPECTRAL_N_CLUSTERS_MAX),
                        "pca_components_min": int(config.PCA_COMPONENTS_MIN),
                        "pca_components_max": int(config.PCA_COMPONENTS_MAX),
                        "num_clustering_runs": int(config.CLUSTERING_RUNS),
                        "max_songs_per_cluster_val": int(config.MAX_SONGS_PER_CLUSTER),
                        "gmm_n_components_min": int(config.GMM_N_COMPONENTS_MIN),
                        "gmm_n_components_max": int(config.GMM_N_COMPONENTS_MAX),
                        "top_n_playlists_param": int(config.TOP_N_PLAYLISTS),
                        "min_songs_per_genre_for_stratification_param": int(config.MIN_SONGS_PER_GENRE_FOR_STRATIFICATION),
                        "stratified_sampling_target_percentile_param": int(config.STRATIFIED_SAMPLING_TARGET_PERCENTILE),
                        "score_weight_diversity_param": float(config.SCORE_WEIGHT_DIVERSITY),
                        "score_weight_silhouette_param": float(config.SCORE_WEIGHT_SILHOUETTE),
                        "score_weight_davies_bouldin_param": float(config.SCORE_WEIGHT_DAVIES_BOULDIN),
                        "score_weight_calinski_harabasz_param": float(config.SCORE_WEIGHT_CALINSKI_HARABASZ),
                        "score_weight_purity_param": float(config.SCORE_WEIGHT_PURITY),
                        "score_weight_other_feature_diversity_param": float(config.SCORE_WEIGHT_OTHER_FEATURE_DIVERSITY),
                        "score_weight_other_feature_purity_param": float(config.SCORE_WEIGHT_OTHER_FEATURE_PURITY),
                        "ai_model_provider_param": config.AI_MODEL_PROVIDER,
                        "ollama_server_url_param": config.OLLAMA_SERVER_URL,
                        "ollama_model_name_param": config.OLLAMA_MODEL_NAME,
                        "openai_server_url_param": config.OPENAI_SERVER_URL,
                        "openai_model_name_param": config.OPENAI_MODEL_NAME,
                        "openai_api_key_param": config.OPENAI_API_KEY,
                        "gemini_api_key_param": config.GEMINI_API_KEY,
                        "gemini_model_name_param": config.GEMINI_MODEL_NAME,
                        "mistral_api_key_param": config.MISTRAL_API_KEY,
                        "mistral_model_name_param": config.MISTRAL_MODEL_NAME,
                        "top_n_moods_for_clustering_param": int(config.TOP_N_MOODS),
                        "enable_clustering_embeddings_param": bool(config.ENABLE_CLUSTERING_EMBEDDINGS),
                    }
                    rq_queue_high.enqueue('tasks.clustering.run_clustering_task', kwargs=clustering_kwargs, job_id=job_id, description='Cron Clustering', job_timeout=-1)
                    logger.info(f"Cron: enqueued clustering job {job_id}")
                elif task_type == 'sonic_fingerprint':
                    # Run synchronously, not via queue, and create playlist on media server
                    from tasks.sonic_fingerprint_manager import generate_sonic_fingerprint
                    from tasks.voyager_manager import create_playlist_from_ids
                    try:
                        fingerprint_results = generate_sonic_fingerprint()
                        if not fingerprint_results:
                            logger.warning(f"Cron: sonic fingerprint found no results (job_id={job_id})")
                        else:
                            track_ids = [r['item_id'] for r in fingerprint_results if 'item_id' in r]
                            playlist_name = f"Sonic Fingerprint (Cron {time.strftime('%Y-%m-%d')})"
                            try:
                                playlist_id = create_playlist_from_ids(playlist_name, track_ids)
                                logger.info(f"Cron: created sonic fingerprint playlist '{playlist_name}' (playlist_id={playlist_id}, job_id={job_id})")
                            except Exception as e:
                                logger.error(f"Cron: error creating playlist for sonic fingerprint: {e}")
                        logger.info(f"Cron: ran sonic fingerprint synchronously (job_id={job_id})")
                    except Exception as e:
                        logger.error(f"Cron: error running sonic fingerprint: {e}")
                # update last_run
                cur2 = db.cursor()
                cur2.execute("UPDATE cron SET last_run=%s WHERE id=%s", (now_ts, r['id']))
                db.commit()
                cur2.close()
        except Exception as e:
            logger.exception(f"Error processing cron row {r}: {e}")
    cur.close()


"""
NOTE: The cron table is NOT created in this file. It is typically created by a database migration or setup script (e.g., an SQL file or Alembic migration).
Check your deployment or setup scripts for the SQL that creates the 'cron' table.
"""
