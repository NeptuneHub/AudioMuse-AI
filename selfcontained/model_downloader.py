#!/usr/bin/env python3
"""
Model Downloader for AudioMuse-AI Standalone Mode
Auto-downloads required ML models on first run
"""

import os
import sys
import urllib.request
import logging
from pathlib import Path
from typing import List, Dict

logger = logging.getLogger(__name__)

# Model download URLs from GitHub releases
GITHUB_RELEASE_BASE = "https://github.com/NeptuneHub/AudioMuse-AI/releases/download/v3.0.0-model"

REQUIRED_MODELS = {
    # MusicNN / MusiCNN models (required for audio feature extraction)
    'msd-musicnn-1.onnx': {
        'url': f'{GITHUB_RELEASE_BASE}/msd-musicnn-1.onnx',
        'description': 'MusiCNN embedding model (msd-musicnn-1)'
    },
    'msd-msd-musicnn-1.onnx': {
        'url': f'{GITHUB_RELEASE_BASE}/msd-msd-musicnn-1.onnx',
        'description': 'MusiCNN prediction model (msd-msd-musicnn-1)'
    },
    'danceability-msd-musicnn-1.onnx': {
        'url': f'{GITHUB_RELEASE_BASE}/danceability-msd-musicnn-1.onnx',
        'description': 'Danceability classifier (MusicNN)'
    },
    'mood_aggressive-msd-musicnn-1.onnx': {
        'url': f'{GITHUB_RELEASE_BASE}/mood_aggressive-msd-musicnn-1.onnx',
        'description': 'Mood - aggressive (MusicNN)'
    },
    'mood_happy-msd-musicnn-1.onnx': {
        'url': f'{GITHUB_RELEASE_BASE}/mood_happy-msd-musicnn-1.onnx',
        'description': 'Mood - happy (MusicNN)'
    },
    'mood_party-msd-musicnn-1.onnx': {
        'url': f'{GITHUB_RELEASE_BASE}/mood_party-msd-musicnn-1.onnx',
        'description': 'Mood - party (MusicNN)'
    },
    'mood_relaxed-msd-musicnn-1.onnx': {
        'url': f'{GITHUB_RELEASE_BASE}/mood_relaxed-msd-musicnn-1.onnx',
        'description': 'Mood - relaxed (MusicNN)'
    },
    'mood_sad-msd-musicnn-1.onnx': {
        'url': f'{GITHUB_RELEASE_BASE}/mood_sad-msd-musicnn-1.onnx',
        'description': 'Mood - sad (MusicNN)'
    },

    # CLAP models for music search (required)
    'clap_audio_model.onnx': {
        'url': f'{GITHUB_RELEASE_BASE}/clap_audio_model.onnx',
        'size': 268_000_000,  # ~268 MB
        'description': 'CLAP audio encoder for music analysis'
    },
    'clap_text_model.onnx': {
        'url': f'{GITHUB_RELEASE_BASE}/clap_text_model.onnx',
        'size': 478_000_000,  # ~478 MB
        'description': 'CLAP text encoder for semantic search'
    },
    
    # HuggingFace models (BERT, RoBERTa, etc.) - tarball
    'huggingface_models.tar.gz': {
        'url': f'{GITHUB_RELEASE_BASE}/huggingface_models.tar.gz',
        'size': 985_000_000,  # ~985 MB
        'description': 'Text encoders (RoBERTa, BERT, BART, T5)',
        'extract_to': '.cache/huggingface',
        'is_tarball': True
    }
}

OPTIONAL_MODELS = {
    # MuLan models (optional, for advanced music understanding)
    'mulan/mulan_audio_encoder.onnx': {
        'url': f'{GITHUB_RELEASE_BASE}/mulan_audio_encoder.onnx',
        'size': 500_000_000,
        'description': 'MuLan audio encoder (optional)'
    },
    'mulan/mulan_audio_encoder.onnx.data': {
        'url': f'{GITHUB_RELEASE_BASE}/mulan_audio_encoder.onnx.data',
        'size': 1_500_000_000,
        'description': 'MuLan audio encoder weights (optional)'
    },
    'mulan/mulan_text_encoder.onnx': {
        'url': f'{GITHUB_RELEASE_BASE}/mulan_text_encoder.onnx',
        'size': 300_000_000,
        'description': 'MuLan text encoder (optional)'
    },
    'mulan/mulan_text_encoder.onnx.data': {
        'url': f'{GITHUB_RELEASE_BASE}/mulan_text_encoder.onnx.data',
        'size': 200_000_000,
        'description': 'MuLan text encoder weights (optional)'
    },
    'mulan/mulan_tokenizer.tar.gz': {
        'url': f'{GITHUB_RELEASE_BASE}/mulan_tokenizer.tar.gz',
        'size': 5_000_000,
        'description': 'MuLan tokenizer files (optional)',
        'extract_to': 'model/mulan',
        'is_tarball': True
    }
}


def _probe_remote_size(url: str) -> int | None:
    """Try to determine remote Content-Length via HTTP HEAD. Returns bytes or None."""
    try:
        req = urllib.request.Request(url, method='HEAD')
        with urllib.request.urlopen(req, timeout=10) as resp:
            cl = resp.getheader('Content-Length') or resp.getheader('content-length')
            if cl:
                return int(cl)
    except Exception:
        return None


def download_file(url: str, destination: Path, expected_size: int = None) -> bool:
    """
    Download a file with progress reporting and retry logic.

    If `expected_size` is not provided we attempt a HTTP HEAD to learn the
    Content-Length for progress reporting and validation.

    Args:
        url: URL to download from
        destination: Local path to save file
        expected_size: Expected file size in bytes (for validation)

    Returns:
        True if successful, False otherwise
    """
    # Attempt to probe remote size if caller didn't provide it
    if expected_size is None:
        probed = _probe_remote_size(url)
        if probed:
            expected_size = probed
            logger.info(f"Detected remote size for {destination.name}: {expected_size/1_000_000:.1f} MB")

    max_retries = 5

    for attempt in range(max_retries):
        try:
            # Create directory if needed
            destination.parent.mkdir(parents=True, exist_ok=True)

            # Download with progress
            logger.info(f"Downloading {destination.name} (attempt {attempt + 1}/{max_retries})...")

            def progress_hook(block_num, block_size, total_size):
                downloaded = block_num * block_size
                # prefer server-supplied total_size, fall back to expected_size if available
                ts = total_size if total_size and total_size > 0 else (expected_size or 0)
                if ts > 0:
                    percent = min(100, int(downloaded * 100 // ts))
                    mb_downloaded = downloaded / 1_000_000
                    mb_total = ts / 1_000_000
                    if block_num % 100 == 0:  # Update every 100 blocks
                        logger.info(f"  Progress: {percent}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)")

            urllib.request.urlretrieve(url, destination, reporthook=progress_hook)

            # Validate size
            actual_size = destination.stat().st_size
            if expected_size and actual_size < expected_size * 0.95:  # Allow 5% variance
                logger.warning(f"File size mismatch: expected ~{expected_size/1_000_000:.1f}MB, got {actual_size/1_000_000:.1f}MB")
                destination.unlink()
                if attempt < max_retries - 1:
                    import time
                    time.sleep((attempt + 1) ** 2)  # Exponential backoff
                    continue
                return False

            logger.info(f"✓ Downloaded {destination.name} ({actual_size/1_000_000:.1f} MB)")
            return True

        except Exception as e:
            logger.error(f"Download attempt {attempt + 1} failed: {e}")
            if destination.exists():
                destination.unlink()

            if attempt < max_retries - 1:
                import time
                time.sleep((attempt + 1) ** 2)  # Exponential backoff
            else:
                logger.error(f"Failed to download {url} after {max_retries} attempts")
                return False

    return False


def extract_tarball(tarball_path: Path, extract_to: Path) -> bool:
    """Extract a .tar.gz file."""
    try:
        import tarfile
        
        logger.info(f"Extracting {tarball_path.name}...")
        extract_to.mkdir(parents=True, exist_ok=True)
        
        with tarfile.open(tarball_path, 'r:gz') as tar:
            tar.extractall(extract_to)
        
        logger.info(f"✓ Extracted to {extract_to}")
        
        # Remove tarball after extraction
        tarball_path.unlink()
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to extract {tarball_path}: {e}")
        return False


def check_and_download_models(model_dir: Path, cache_dir: Path, include_optional: bool = False) -> bool:
    """
    Check if models exist, download if missing.
    
    Args:
        model_dir: Directory to store ONNX models (e.g., ~/.audiomuse/model)
        cache_dir: Directory for cache files (e.g., ~/.audiomuse/.cache)
        include_optional: Whether to download optional models (MuLan)
    
    Returns:
        True if all required models are available, False otherwise
    """
    model_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    models_to_check = REQUIRED_MODELS.copy()
    if include_optional:
        models_to_check.update(OPTIONAL_MODELS)
    
    missing_models = []
    
    # Check which models are missing
    for model_name, model_info in models_to_check.items():
        if model_info.get('is_tarball'):
            # For tarballs, check if extraction directory exists
            extract_to = cache_dir / model_info['extract_to'] if 'extract_to' in model_info else model_dir / model_info['extract_to']
            if not extract_to.exists() or not any(extract_to.iterdir()):
                missing_models.append((model_name, model_info))
        else:
            # For regular files
            model_path = model_dir / model_name
            if not model_path.exists():
                missing_models.append((model_name, model_info))
    
    if not missing_models:
        logger.info("✓ All models are already downloaded")
        return True

    # For models that don't include a 'size' field, probe the remote URL with
    # an HTTP HEAD to get Content-Length so we can present a realistic total.
    for model_name, model_info in missing_models:
        if 'size' not in model_info:
            probed = _probe_remote_size(model_info['url'])
            if probed:
                model_info['size'] = probed
                logger.info(f"Detected remote size for {model_name}: {probed/1_000_000:.1f} MB")

    # Calculate total download size (missing sizes treated as 0)
    total_size = sum(info.get('size', 0) for _, info in missing_models) / 1_000_000  # MB
    logger.info(f"")
    logger.info(f"╔══════════════════════════════════════════════════════════╗")
    logger.info(f"║  First-time setup: Downloading ML models                ║")
    logger.info(f"╚══════════════════════════════════════════════════════════╝")
    logger.info(f"")
    logger.info(f"  Models to download: {len(missing_models)}")
    logger.info(f"  Total size: ~{total_size:.0f} MB")
    logger.info(f"")
    
    # Download each missing model
    for model_name, model_info in missing_models:
        logger.info(f"Downloading: {model_info['description']}")
        
        # Determine destination
        if model_info.get('is_tarball'):
            destination = cache_dir / model_name if '/' not in model_name else model_dir / model_name
        else:
            destination = model_dir / model_name
        
        # Download
        if not download_file(model_info['url'], destination, model_info.get('size')):
            logger.error(f"Failed to download {model_name}")
            return False
        
        # Extract if tarball
        if model_info.get('is_tarball'):
            extract_to = cache_dir / model_info['extract_to'] if 'extract_to' in model_info else model_dir / model_info['extract_to']
            if not extract_tarball(destination, extract_to):
                return False
    
    logger.info(f"")
    logger.info(f"✓ All models downloaded successfully!")
    logger.info(f"")
    
    return True


def check_system_dependencies():
    """Check if required system dependencies are available."""
    missing = []
    
    # Check for ffmpeg
    import shutil
    if not shutil.which('ffmpeg'):
        missing.append('ffmpeg')
    
    if missing:
        logger.warning(f"")
        logger.warning(f"╔══════════════════════════════════════════════════════════╗")
        logger.warning(f"║  WARNING: Missing system dependencies                    ║")
        logger.warning(f"╚══════════════════════════════════════════════════════════╝")
        logger.warning(f"")
        logger.warning(f"The following system tools are required but not found:")
        for dep in missing:
            logger.warning(f"  - {dep}")
        logger.warning(f"")
        logger.warning(f"Install instructions:")
        logger.warning(f"  macOS:   brew install ffmpeg")
        logger.warning(f"  Ubuntu:  sudo apt install ffmpeg")
        logger.warning(f"  Windows: Download from https://ffmpeg.org/download.html")
        logger.warning(f"")
        
        # Don't fail, just warn
        return True
    
    return True


if __name__ == '__main__':
    # Test model downloader
    logging.basicConfig(level=logging.INFO)
    
    model_dir = Path.home() / '.audiomuse' / 'model'
    cache_dir = Path.home() / '.audiomuse' / '.cache'
    
    check_system_dependencies()
    success = check_and_download_models(model_dir, cache_dir, include_optional=False)
    
    sys.exit(0 if success else 1)
