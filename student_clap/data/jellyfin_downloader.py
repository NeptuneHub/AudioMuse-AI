"""
Jellyfin Audio Downloader with Caching

Downloads audio files from Jellyfin media server with local caching
to avoid repeated downloads during training.
"""

import os
import sys
import logging
import hashlib
import requests
from pathlib import Path
from typing import Optional, Dict
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

logger = logging.getLogger(__name__)


class JellyfinDownloader:
    """Downloads audio files from Jellyfin with caching."""
    
    def __init__(self, config: dict, cache_dir: str = './cache/audio', max_cache_size_gb: float = 2.0):
        """
        Initialize Jellyfin downloader.
        
        Args:
            config: Jellyfin configuration dict with keys:
                - url: Jellyfin server URL
                - user_id: User ID
                - token: API token
            cache_dir: Directory for caching downloaded files
            max_cache_size_gb: Maximum cache size in GB (default: 2GB)
        """
        self.url = config['url'].rstrip('/')
        self.user_id = config['user_id']
        self.token = config['token']
        self.cache_dir = Path(cache_dir)
        self.max_cache_size = max_cache_size_gb * 1024 * 1024 * 1024  # Convert to bytes
        
        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Request headers
        self.headers = {
            'X-Emby-Token': self.token
        }
        
        # Stats
        self.download_count = 0
        self.cache_hit_count = 0
        
    def _get_cache_path(self, item_id: str, extension: str = '.audio') -> Path:
        """
        Get cache file path for an item.
        
        Args:
            item_id: Jellyfin item ID
            extension: File extension
            
        Returns:
            Path to cache file
        """
        # Use item_id as filename (sanitized)
        filename = f"{item_id}{extension}"
        return self.cache_dir / filename
        
    def _get_item_info(self, item_id: str) -> Optional[Dict]:
        """
        Get item information from Jellyfin.
        
        Args:
            item_id: Jellyfin item ID
            
        Returns:
            Item info dict or None if failed
        """
        try:
            url = f"{self.url}/Users/{self.user_id}/Items/{item_id}"
            response = requests.get(url, headers=self.headers, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get item info for {item_id}: {e}")
            return None
    
    def check_item_exists(self, item_id: str) -> bool:
        """
        Check if item's audio file can actually be downloaded (not just metadata exists).
        
        Args:
            item_id: Jellyfin item ID
            
        Returns:
            True if audio file exists and is downloadable
        """
        try:
            # Get download URL
            url = f"{self.url}/Items/{item_id}/Download"
            params = {"api_key": self.token}
            
            # Try to download first 1KB to verify file is accessible
            response = requests.get(url, params=params, headers=self.headers, 
                                  timeout=10, stream=True)
            
            # If we can start downloading, file exists
            if response.status_code == 200:
                # Close connection immediately (don't download full file)
                response.close()
                return True
            return False
        except Exception:
            return False
            
    def _cleanup_cache(self):
        """Remove old files if cache size exceeds limit."""
        current_size = self.get_cache_size()
        if current_size <= self.max_cache_size:
            return
            
        logger.info(f"Cache size {current_size / (1024**3):.1f}GB exceeds limit {self.max_cache_size / (1024**3):.1f}GB, cleaning up...")
        
        # Get all cached files with their access times
        cache_files = []
        for cache_file in self.cache_dir.glob('*'):
            if cache_file.is_file():
                stat = cache_file.stat()
                cache_files.append({
                    'path': cache_file,
                    'size': stat.st_size,
                    'atime': stat.st_atime  # Access time
                })
        
        # Sort by access time (oldest first)
        cache_files.sort(key=lambda x: x['atime'])
        
        # Remove files until we're under the limit
        freed_space = 0
        files_removed = 0
        for file_info in cache_files:
            if current_size - freed_space <= self.max_cache_size:
                break
                
            file_info['path'].unlink()
            freed_space += file_info['size']
            files_removed += 1
            
        logger.info(f"Removed {files_removed} old files, freed {freed_space / (1024**2):.1f}MB")
            
    def download(self, item_id: str, force: bool = False) -> Optional[str]:
        """
        Download audio file from Jellyfin (with caching).
        
        Args:
            item_id: Jellyfin item ID
            force: Force re-download even if cached
            
        Returns:
            Path to downloaded file or None if failed
        """
        # Get item info for extension
        item_info = self._get_item_info(item_id)
        if item_info is None:
            logger.error(f"Could not get info for item {item_id}")
            return None
            
        # Extract file extension
        path = item_info.get('Path', '')
        extension = os.path.splitext(path)[1] or '.tmp'
        
        # Check cache
        cache_path = self._get_cache_path(item_id, extension)
        if cache_path.exists() and not force:
            logger.debug(f"Cache hit for {item_id}: {cache_path}")
            self.cache_hit_count += 1
            # Update access time for LRU cache management
            cache_path.touch()
            return str(cache_path)
        
        # Wait if cache is full (let mel computation cleanup make space)
        import time
        max_retries = 10
        retry_count = 0
        while self.get_cache_size() >= self.max_cache_size and retry_count < max_retries:
            logger.debug(f"Cache full, waiting for cleanup... ({retry_count+1}/{max_retries})")
            time.sleep(1)  # Wait 1 second for mel computation to cleanup
            retry_count += 1
            
        if retry_count >= max_retries:
            logger.error(f"Cache still full after {max_retries} retries, downloading anyway...")
            
        # Download from Jellyfin
        try:
            download_url = f"{self.url}/Items/{item_id}/Download"
            logger.info(f"Downloading {item_info.get('Name', item_id)} from Jellyfin...")
            
            response = requests.get(
                download_url,
                headers=self.headers,
                stream=True,
                timeout=60
            )
            response.raise_for_status()
            
            # Get total size for progress bar
            total_size = int(response.headers.get('content-length', 0))
            
            # Write to cache with progress bar
            with open(cache_path, 'wb') as f:
                if total_size > 0:
                    with tqdm(total=total_size, unit='B', unit_scale=True, 
                             desc=f"Downloading {item_id[:8]}") as pbar:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                            pbar.update(len(chunk))
                else:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        
            logger.info(f"Downloaded and cached: {cache_path}")
            self.download_count += 1
            return str(cache_path)
            
        except Exception as e:
            logger.error(f"Failed to download {item_id}: {e}")
            # Clean up partial download
            if cache_path.exists():
                cache_path.unlink()
            return None
            
    def download_batch(self, item_ids: list, max_workers: int = 4) -> Dict[str, Optional[str]]:
        """
        Download multiple audio files in parallel.
        
        Args:
            item_ids: List of Jellyfin item IDs
            max_workers: Number of parallel downloads
            
        Returns:
            Dict mapping item_id to downloaded file path (or None if failed)
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        results = {}
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all downloads
            future_to_id = {
                executor.submit(self.download, item_id): item_id
                for item_id in item_ids
            }
            
            # Collect results with progress bar
            with tqdm(total=len(item_ids), desc="Downloading audio files") as pbar:
                for future in as_completed(future_to_id):
                    item_id = future_to_id[future]
                    try:
                        path = future.result()
                        results[item_id] = path
                    except Exception as e:
                        logger.error(f"Exception downloading {item_id}: {e}")
                        results[item_id] = None
                    pbar.update(1)
                    
        return results
        
    def clear_cache(self):
        """Clear all cached files."""
        count = 0
        for cache_file in self.cache_dir.glob('*'):
            if cache_file.is_file():
                cache_file.unlink()
                count += 1
        logger.info(f"Cleared {count} files from cache")
        
    def get_cache_size(self) -> int:
        """
        Get total size of cached files in bytes.
        
        Returns:
            Total cache size in bytes
        """
        total_size = 0
        for cache_file in self.cache_dir.glob('*'):
            if cache_file.is_file():
                total_size += cache_file.stat().st_size
        return total_size
        
    def get_stats(self) -> Dict:
        """
        Get downloader statistics.
        
        Returns:
            Dict with download and cache statistics
        """
        cache_files = list(self.cache_dir.glob('*'))
        cache_count = len([f for f in cache_files if f.is_file()])
        cache_size = self.get_cache_size()
        
        return {
            'downloads': self.download_count,
            'cache_hits': self.cache_hit_count,
            'cached_files': cache_count,
            'cache_size_mb': cache_size / (1024 * 1024),
            'cache_dir': str(self.cache_dir)
        }


def test_jellyfin_connection(config: dict) -> bool:
    """
    Test Jellyfin connection.
    
    Args:
        config: Jellyfin configuration dict
        
    Returns:
        True if connection successful
    """
    try:
        url = config['url'].rstrip('/')
        headers = {'X-Emby-Token': config['token']}
        
        # Test system info endpoint
        response = requests.get(f"{url}/System/Info", headers=headers, timeout=10)
        response.raise_for_status()
        
        info = response.json()
        logger.info(f"Connected to Jellyfin: {info.get('ServerName', 'Unknown')} "
                   f"v{info.get('Version', 'Unknown')}")
        return True
        
    except Exception as e:
        logger.error(f"Jellyfin connection failed: {e}")
        return False


if __name__ == '__main__':
    """Test Jellyfin downloader functionality."""
    import yaml
    import argparse
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    parser = argparse.ArgumentParser(description='Test Jellyfin downloader')
    parser.add_argument('--config', type=str, default='../config.yaml',
                        help='Path to config file')
    parser.add_argument('--test', action='store_true',
                        help='Test Jellyfin connection')
    parser.add_argument('--item-id', type=str,
                        help='Test download a specific item')
    parser.add_argument('--clear-cache', action='store_true',
                        help='Clear download cache')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    # Expand environment variables
    jellyfin_config = {}
    for key, value in config['jellyfin'].items():
        if isinstance(value, str) and value.startswith('${') and value.endswith('}'):
            env_var = value[2:-1]
            jellyfin_config[key] = os.environ.get(env_var, value)
        else:
            jellyfin_config[key] = value
    
    if args.test:
        # Test connection
        if test_jellyfin_connection(jellyfin_config):
            print("✓ Jellyfin connection successful")
        else:
            print("✗ Jellyfin connection failed")
            sys.exit(1)
    
    # Create downloader
    downloader = JellyfinDownloader(
        jellyfin_config,
        cache_dir=config['paths']['audio_cache']
    )
    
    if args.clear_cache:
        downloader.clear_cache()
        print("✓ Cache cleared")
        sys.exit(0)
    
    if args.item_id:
        # Test download
        print(f"Downloading item {args.item_id}...")
        path = downloader.download(args.item_id)
        if path:
            print(f"✓ Downloaded to: {path}")
        else:
            print("✗ Download failed")
            sys.exit(1)
    
    # Print stats
    stats = downloader.get_stats()
    print(f"\nDownloader Statistics:")
    print(f"  Downloads: {stats['downloads']}")
    print(f"  Cache hits: {stats['cache_hits']}")
    print(f"  Cached files: {stats['cached_files']}")
    print(f"  Cache size: {stats['cache_size_mb']:.2f} MB")
    print(f"  Cache directory: {stats['cache_dir']}")
