"""
Jellyfin client for downloading audio files.
Uses existing AudioMuse-AI patterns for authentication and streaming.
"""

import logging
import os
import requests
from typing import Dict, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

REQUESTS_TIMEOUT = 300


class JellyfinClient:
    """Client for downloading audio from Jellyfin media server."""
    
    def __init__(self, config: Dict):
        """
        Initialize Jellyfin client.
        
        Args:
            config: Jellyfin configuration dictionary with keys:
                    url, user_id, token
        """
        self.url = config['url']
        self.user_id = config['user_id']
        self.token = config['token']
        self.headers = {"X-Emby-Token": self.token}
        
        logger.info(f"Initialized Jellyfin client for {self.url}")
    
    def get_item_info(self, item_id: str) -> Optional[Dict]:
        """
        Get item information from Jellyfin.
        
        Args:
            item_id: Jellyfin item ID
            
        Returns:
            Dictionary with item information or None if failed
        """
        url = f"{self.url}/Users/{self.user_id}/Items/{item_id}"
        
        try:
            response = requests.get(url, headers=self.headers, timeout=REQUESTS_TIMEOUT)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get item info for {item_id}: {e}")
            return None
    
    def download_audio(self, item_id: str, output_dir: str, force_download: bool = False) -> Optional[str]:
        """
        Download audio file from Jellyfin at highest quality.
        
        Args:
            item_id: Jellyfin item ID
            output_dir: Directory to save downloaded file
            force_download: If True, download even if file exists
            
        Returns:
            Path to downloaded file or None if failed
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Get item info to determine file extension
        item_info = self.get_item_info(item_id)
        if not item_info:
            logger.error(f"Cannot download - failed to get item info for {item_id}")
            return None
        
        # Determine file extension
        file_path = item_info.get('Path', '')
        file_extension = os.path.splitext(file_path)[1] or '.tmp'
        
        # Local filename
        local_filename = os.path.join(output_dir, f"{item_id}{file_extension}")
        
        # Check if file already exists and skip if not forcing download
        if os.path.exists(local_filename) and not force_download:
            logger.info(f"File already exists: {local_filename}")
            return local_filename
        
        # Download from Jellyfin
        download_url = f"{self.url}/Items/{item_id}/Download"
        
        try:
            logger.info(f"Downloading {item_info.get('Name', item_id)} to {local_filename}")
            
            with requests.get(download_url, headers=self.headers, stream=True, timeout=REQUESTS_TIMEOUT) as r:
                r.raise_for_status()
                
                with open(local_filename, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            
            logger.info(f"Successfully downloaded to {local_filename}")
            return local_filename
            
        except Exception as e:
            logger.error(f"Failed to download {item_id}: {e}")
            # Clean up partial file if it exists
            if os.path.exists(local_filename):
                try:
                    os.remove(local_filename)
                except:
                    pass
            return None
    
    def batch_download(self, item_ids: list, output_dir: str, force_download: bool = False) -> Dict[str, str]:
        """
        Download multiple audio files.
        
        Args:
            item_ids: List of Jellyfin item IDs
            output_dir: Directory to save downloaded files
            force_download: If True, download even if files exist
            
        Returns:
            Dictionary mapping item_id to local file path (only successful downloads)
        """
        downloaded_files = {}
        
        for i, item_id in enumerate(item_ids, 1):
            logger.info(f"Downloading {i}/{len(item_ids)}: {item_id}")
            
            local_path = self.download_audio(item_id, output_dir, force_download)
            
            if local_path:
                downloaded_files[item_id] = local_path
            else:
                logger.warning(f"Failed to download {item_id}")
        
        logger.info(f"Successfully downloaded {len(downloaded_files)}/{len(item_ids)} files")
        return downloaded_files
