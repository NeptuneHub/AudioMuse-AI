# app_waveform.py
from flask import Blueprint, jsonify, request, render_template
import logging
import json
import os
import tempfile
import numpy as np
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from functools import lru_cache

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("librosa not available. Waveform generation will not work. Install with: pip install librosa")

from config import MEDIASERVER_TYPE
from app_helper import get_db

logger = logging.getLogger(__name__)

# Thread pool for parallel processing
# Allow up to 4 concurrent waveform generations
_waveform_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="waveform")

# Create a Blueprint for waveform related routes
waveform_bp = Blueprint('waveform_bp', __name__, template_folder='templates')


@waveform_bp.route('/waveform', methods=['GET'])
def waveform_page():
    """
    Serves the frontend page for waveform visualization.
    ---
    tags:
      - UI
    responses:
      200:
        description: HTML content of the waveform page.
        content:
          text/html:
            schema:
              type: string
    """
    return render_template('waveform.html', title='AudioMuse-AI - Waveform Visualization', active='waveform')


def generate_waveform_peaks(file_path, samples_count=500):
    """
    Generate waveform peaks data for fast visualization.
    Ultra-fast implementation using soundfile + scipy for speed.
    
    Args:
        file_path: Path to the audio file
        samples_count: Number of sample pairs to generate (default 500 = 1000 values for min/max peaks)
    
    Returns:
        List of float32 values representing min/max pairs, normalized to -1.0 to 1.0
        
    Raises:
        RuntimeError: If required libraries are not available or audio processing fails
    """
    if not LIBROSA_AVAILABLE:
        raise RuntimeError("librosa library is not installed. Please install it with: pip install librosa")
    
    try:
        import soundfile as sf
        from scipy import signal
        
        # ULTRA-FAST APPROACH: Use soundfile directly (10x faster than librosa.load)
        # Read audio file
        try:
            y, sr = sf.read(file_path, always_2d=False)
        except Exception as sf_error:
            # If soundfile fails (e.g., .tmp extension), try to detect format from content
            # or rename with proper extension
            logger.warning(f"soundfile failed to read {file_path}: {sf_error}. Trying librosa fallback...")
            raise ImportError("soundfile failed, using librosa fallback")
        
        # Convert stereo to mono if needed (simple average, very fast)
        if y.ndim > 1:
            y = np.mean(y, axis=1)
        
        # Quick downsample if sample rate is high (optional, for even more speed)
        if sr > 16000:
            # Decimate to ~8000 Hz for faster processing
            decimation_factor = sr // 8000
            if decimation_factor > 1:
                y = signal.decimate(y, decimation_factor, ftype='fir', zero_phase=True)
        
        if len(y) == 0:
            return []
        
        # Vectorized peak extraction
        samples_per_peak = max(1, len(y) // samples_count)
        trim_length = samples_count * samples_per_peak
        y_trimmed = y[:trim_length]
        
        # Reshape and get peak energy
        chunks = y_trimmed.reshape(samples_count, samples_per_peak)
        abs_chunks = np.abs(chunks)
        peak_energy = np.max(abs_chunks, axis=1)
        
        # Normalize to -1 to 1 range
        max_val = np.max(peak_energy)
        if max_val > 0:
            peak_energy = peak_energy / max_val
        
        # Create symmetric waveform
        peaks = np.empty(samples_count * 2, dtype=np.float32)
        peaks[0::2] = -peak_energy
        peaks[1::2] = peak_energy
        
        return peaks.tolist()
        
    except ImportError as e:
        # Fallback to librosa if soundfile/scipy not available
        logger.warning(f"soundfile or scipy not available, falling back to slower librosa.load: {e}")
        y, sr = librosa.load(file_path, sr=6000, mono=True, res_type='kaiser_fast')
        
        if len(y) == 0:
            return []
        
        samples_per_peak = max(1, len(y) // samples_count)
        trim_length = samples_count * samples_per_peak
        y_trimmed = y[:trim_length]
        
        chunks = y_trimmed.reshape(samples_count, samples_per_peak)
        abs_chunks = np.abs(chunks)
        peak_energy = np.max(abs_chunks, axis=1)
        
        peaks = np.empty(samples_count * 2, dtype=np.float32)
        peaks[0::2] = -peak_energy
        peaks[1::2] = peak_energy
        peaks = np.clip(peaks, -1.0, 1.0)
        
        return peaks.tolist()
        
    except Exception as e:
        raise RuntimeError(f"Failed to generate waveform: {str(e)}")


@waveform_bp.route('/api/waveform', methods=['GET'])
def get_waveform_endpoint():
    """
    Generate waveform peaks data for a track identified by item_id.
    ---
    tags:
      - Waveform
    parameters:
      - name: item_id
        in: query
        required: true
        description: The media server Item ID of the track.
        schema:
          type: string
    responses:
      200:
        description: Waveform peaks data.
        content:
          application/json:
            schema:
              type: object
              properties:
                peaks:
                  type: array
                  items:
                    type: number
                  description: Array of normalized peak values (min/max pairs)
                duration:
                  type: number
                  description: Track duration in seconds (if available)
                title:
                  type: string
                  description: Track title
                author:
                  type: string
                  description: Track artist
      400:
        description: Bad request, missing item_id
      404:
        description: Track not found
      500:
        description: Server error during waveform generation
    """
    item_id = request.args.get('item_id')
    
    if not item_id:
        return jsonify({"error": "Missing 'item_id' parameter"}), 400
    
    # Get track information from database
    db = get_db()
    cur = db.cursor()
    cur.execute("SELECT title, author FROM score WHERE item_id = %s", (item_id,))
    track_info = cur.fetchone()
    cur.close()
    
    if not track_info:
        return jsonify({"error": f"Track with ID '{item_id}' not found"}), 404
    
    title, author = track_info
    
    # Download the track to a temporary location
    temp_file = None
    try:
        import time
        start_time = time.time()
        
        # Import download_track from the generic mediaserver module which handles all types
        from tasks.mediaserver import download_track
        
        # For better compatibility, we need to fetch the full track details from the media server
        # This ensures we have all the metadata needed for proper file extension detection
        if MEDIASERVER_TYPE == "navidrome":
            # Import Navidrome-specific function to get full song details
            from tasks.mediaserver_navidrome import _navidrome_request
            song_response = _navidrome_request("getSong", {"id": item_id})
            if song_response and "song" in song_response:
                item = song_response["song"]
                # Navidrome may provide 'suffix' field which is the file extension
                if 'suffix' in item and item['suffix']:
                    # Ensure we have the proper extension for soundfile to recognize
                    item['path'] = f"dummy.{item['suffix']}"
                elif 'contentType' in item:
                    # Map MIME type to extension
                    mime_to_ext = {
                        'audio/mpeg': '.mp3',
                        'audio/mp4': '.m4a',
                        'audio/flac': '.flac',
                        'audio/ogg': '.ogg',
                        'audio/opus': '.opus',
                        'audio/x-wav': '.wav',
                        'audio/wav': '.wav',
                    }
                    item['path'] = mime_to_ext.get(item['contentType'], '.mp3')
            else:
                return jsonify({"error": "Failed to fetch track details from Navidrome"}), 404
        else:
            # Create a minimal item dict with the item_id for other media servers
            # The download_track function will handle the specifics for each media server type
            item = {
                'Id': item_id,  # Jellyfin/Emby format
                'id': item_id,  # Navidrome/Lyrion format
                'file': item_id,  # MPD format (uses file path as ID)
                'Name': title,
                'Path': ''  # Will be fetched by download_track if needed
            }
        
        fetch_time = time.time() - start_time
        logger.info(f"⏱️  Fetched track metadata in {fetch_time:.2f}s")
        
        # Create a temporary directory for this download
        temp_dir = tempfile.mkdtemp(prefix='waveform_')
        
        # Download the track
        download_start = time.time()
        temp_file = download_track(temp_dir, item)
        download_time = time.time() - download_start
        logger.info(f"⏱️  Downloaded track in {download_time:.2f}s")
        
        if not temp_file or not os.path.exists(temp_file):
            return jsonify({"error": "Failed to download track from media server"}), 500
        
        # Generate waveform peaks in a thread pool with timeout
        logger.info(f"🌊 Generating waveform with librosa for song={title}, item_id={item_id}")
        
        waveform_start = time.time()
        # Submit to thread pool for parallel execution
        future = _waveform_executor.submit(generate_waveform_peaks, temp_file, 500)
        
        try:
            # Wait up to 15 seconds for waveform generation
            peaks = future.result(timeout=15)
        except FuturesTimeoutError:
            logger.error(f"Waveform generation timed out for {item_id}")
            return jsonify({"error": "Waveform generation timed out (>15s). Try a shorter audio file."}), 500
        
        waveform_time = time.time() - waveform_start
        total_time = time.time() - start_time
        logger.info(f"✅ Generated {len(peaks)} waveform peaks in {waveform_time:.2f}s (total: {total_time:.2f}s)")
        
        response = {
            "peaks": peaks,
            "title": title,
            "author": author
        }
        
        return jsonify(response), 200
        
    except RuntimeError as e:
        logger.error(f"Runtime error generating waveform for {item_id}: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        logger.error(f"Unexpected error generating waveform for {item_id}: {e}", exc_info=True)
        return jsonify({"error": "An unexpected error occurred during waveform generation"}), 500
    finally:
        # Clean up temporary file
        if temp_file and os.path.exists(temp_file):
            try:
                os.remove(temp_file)
                # Try to remove the temp directory if it's empty
                temp_dir = os.path.dirname(temp_file)
                if os.path.exists(temp_dir):
                    os.rmdir(temp_dir)
            except Exception as cleanup_error:
                logger.warning(f"Failed to clean up temporary file {temp_file}: {cleanup_error}")
