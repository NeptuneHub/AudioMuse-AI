"""
Configuration manager for standalone mode.
Handles reading/writing media server configuration to ~/.audiomuse/config.ini
"""
import os
import re
import configparser
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def get_config_path():
    """Get path to config.ini file."""
    return Path.home() / '.audiomuse' / 'config.ini'

def read_config():
    """Read configuration from config.ini file."""
    config_path = get_config_path()
    config = configparser.ConfigParser()
    
    if config_path.exists():
        try:
            config.read(config_path)
            return config
        except Exception as e:
            logger.error(f"Error reading config file: {e}")
            return configparser.ConfigParser()
    return configparser.ConfigParser()

def _normalize_base_url(url: str) -> str:
    """Normalize a base URL for mediaserver endpoints.

    - Trim whitespace
    - Ensure scheme (add http:// if missing)
    - Ensure trailing slash
    - Return empty string for falsy input
    """
    if not url:
        return ''
    url = url.strip()
    # Add scheme if missing
    if not re.match(r'^https?://', url, re.I):
        url = 'http://' + url
    # Ensure trailing slash
    if not url.endswith('/'):
        url = url + '/'
    return url


def write_config(config_data):
    """
    Write configuration to config.ini file.
    
    Args:
        config_data: Dictionary with configuration values
        
    Returns:
        True if successful, False otherwise
    """
    config_path = get_config_path()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    config = configparser.ConfigParser()
    
    # Add mediaserver section
    config['mediaserver'] = {
        'type': config_data.get('mediaserver_type', 'jellyfin')
    }
    
    # Add type-specific configuration
    media_type = config_data.get('mediaserver_type', 'jellyfin')
    
    if media_type == 'jellyfin':
        config['jellyfin'] = {
            'url': _normalize_base_url(config_data.get('jellyfin_url', '')),
            'user_id': config_data.get('jellyfin_user_id', ''),
            'token': config_data.get('jellyfin_token', '')
        }
    elif media_type == 'navidrome':
        config['navidrome'] = {
            'url': _normalize_base_url(config_data.get('navidrome_url', '')),
            'user': config_data.get('navidrome_user', ''),
            'password': config_data.get('navidrome_password', '')
        }
    elif media_type == 'lyrion':
        config['lyrion'] = {
            'url': _normalize_base_url(config_data.get('lyrion_url', ''))
        }
    elif media_type == 'emby':
        config['emby'] = {
            'url': _normalize_base_url(config_data.get('emby_url', '')),
            'user_id': config_data.get('emby_user_id', ''),
            'token': config_data.get('emby_token', '')
        }
    
    # Add optional music libraries
    music_libraries = config_data.get('music_libraries', '')
    if music_libraries:
        config['mediaserver']['music_libraries'] = music_libraries
    
    try:
        with open(config_path, 'w') as configfile:
            config.write(configfile)
        logger.info(f"Configuration saved to {config_path}")
        return True
    except Exception as e:
        logger.error(f"Error writing config file: {e}")
        return False

def apply_config_to_environment():
    """
    Read config.ini and set environment variables.
    Called by launcher.py before starting the app.
    
    Returns:
        True if config exists and was applied, False otherwise
    """
    config = read_config()
    
    if not config.sections():
        logger.warning("No configuration found. Please visit http://localhost:8000/setup to configure.")
        return False
    
    # Apply mediaserver type
    if config.has_section('mediaserver'):
        media_type = config.get('mediaserver', 'type', fallback='jellyfin')
        os.environ['MEDIASERVER_TYPE'] = media_type
        logger.info(f"Media server type: {media_type}")
        
        # Apply music libraries if set
        if config.has_option('mediaserver', 'music_libraries'):
            os.environ['MUSIC_LIBRARIES'] = config.get('mediaserver', 'music_libraries')
    
    # Apply type-specific settings
    if config.has_section('jellyfin'):
        os.environ['JELLYFIN_URL'] = _normalize_base_url(config.get('jellyfin', 'url', fallback=''))
        os.environ['JELLYFIN_USER_ID'] = config.get('jellyfin', 'user_id', fallback='')
        os.environ['JELLYFIN_TOKEN'] = config.get('jellyfin', 'token', fallback='')
    
    if config.has_section('navidrome'):
        os.environ['NAVIDROME_URL'] = _normalize_base_url(config.get('navidrome', 'url', fallback=''))
        os.environ['NAVIDROME_USER'] = config.get('navidrome', 'user', fallback='')
        os.environ['NAVIDROME_PASSWORD'] = config.get('navidrome', 'password', fallback='')
    
    if config.has_section('lyrion'):
        os.environ['LYRION_URL'] = _normalize_base_url(config.get('lyrion', 'url', fallback=''))
    
    if config.has_section('emby'):
        os.environ['EMBY_URL'] = _normalize_base_url(config.get('emby', 'url', fallback=''))
        os.environ['EMBY_USER_ID'] = config.get('emby', 'user_id', fallback='')
        os.environ['EMBY_TOKEN'] = config.get('emby', 'token', fallback='')

    # ALSO update the in-memory `config` module so modules that already imported
    # `config` pick up values from the standalone config.ini without requiring a
    # process restart or import-order assumptions.
    try:
        import config as _cfg
        # Update only the mediaserver-related values (minimal surface area)
        _cfg.MEDIASERVER_TYPE = os.environ.get('MEDIASERVER_TYPE', _cfg.MEDIASERVER_TYPE)
        _cfg.MUSIC_LIBRARIES = os.environ.get('MUSIC_LIBRARIES', _cfg.MUSIC_LIBRARIES)

        # Jellyfin
        _cfg.JELLYFIN_URL = os.environ.get('JELLYFIN_URL', getattr(_cfg, 'JELLYFIN_URL', ''))
        _cfg.JELLYFIN_USER_ID = os.environ.get('JELLYFIN_USER_ID', getattr(_cfg, 'JELLYFIN_USER_ID', ''))
        _cfg.JELLYFIN_TOKEN = os.environ.get('JELLYFIN_TOKEN', getattr(_cfg, 'JELLYFIN_TOKEN', ''))

        # Navidrome / Lyrion / Emby
        _cfg.NAVIDROME_URL = os.environ.get('NAVIDROME_URL', getattr(_cfg, 'NAVIDROME_URL', ''))
        _cfg.LYRION_URL = os.environ.get('LYRION_URL', getattr(_cfg, 'LYRION_URL', ''))
        _cfg.EMBY_URL = os.environ.get('EMBY_URL', getattr(_cfg, 'EMBY_URL', ''))
        _cfg.EMBY_USER_ID = os.environ.get('EMBY_USER_ID', getattr(_cfg, 'EMBY_USER_ID', ''))
        _cfg.EMBY_TOKEN = os.environ.get('EMBY_TOKEN', getattr(_cfg, 'EMBY_TOKEN', ''))

        # Recompute `HEADERS` in the in-memory config module so code that uses
        # `config.HEADERS` (set at import-time) immediately sees the updated token.
        try:
            if _cfg.MEDIASERVER_TYPE == 'jellyfin':
                _cfg.HEADERS = {"X-Emby-Token": _cfg.JELLYFIN_TOKEN}
            elif _cfg.MEDIASERVER_TYPE == 'emby':
                _cfg.HEADERS = {"X-Emby-Token": _cfg.EMBY_TOKEN}
            else:
                _cfg.HEADERS = {}
        except Exception:
            # non-critical — continue
            pass

        logger.info("Applied standalone config.ini values to runtime config module")
    except Exception:
        logger.debug("Could not update in-memory config module from config.ini")

    return True

def is_configured():
    """Check if media server is configured."""
    config = read_config()
    if not config.has_section('mediaserver'):
        return False
    
    media_type = config.get('mediaserver', 'type', fallback='')
    
    # Check if type-specific configuration exists
    if media_type == 'jellyfin':
        return config.has_section('jellyfin') and config.get('jellyfin', 'url', fallback='') != ''
    elif media_type == 'navidrome':
        return config.has_section('navidrome') and config.get('navidrome', 'url', fallback='') != ''
    elif media_type == 'lyrion':
        return config.has_section('lyrion') and config.get('lyrion', 'url', fallback='') != ''
    elif media_type == 'emby':
        return config.has_section('emby') and config.get('emby', 'url', fallback='') != ''
    
    return False

def get_current_config():
    """Get current configuration as dictionary for display in UI."""
    config = read_config()
    result = {}
    
    if config.has_section('mediaserver'):
        result['mediaserver_type'] = config.get('mediaserver', 'type', fallback='jellyfin')
        result['music_libraries'] = config.get('mediaserver', 'music_libraries', fallback='')
    
    if config.has_section('jellyfin'):
        result['jellyfin_url'] = _normalize_base_url(config.get('jellyfin', 'url', fallback=''))
        result['jellyfin_user_id'] = config.get('jellyfin', 'user_id', fallback='')
        result['jellyfin_token'] = config.get('jellyfin', 'token', fallback='')
    
    if config.has_section('navidrome'):
        result['navidrome_url'] = _normalize_base_url(config.get('navidrome', 'url', fallback=''))
        result['navidrome_user'] = config.get('navidrome', 'user', fallback='')
        result['navidrome_password'] = config.get('navidrome', 'password', fallback='')
    
    if config.has_section('lyrion'):
        result['lyrion_url'] = _normalize_base_url(config.get('lyrion', 'url', fallback=''))
    
    if config.has_section('emby'):
        result['emby_url'] = _normalize_base_url(config.get('emby', 'url', fallback=''))
        result['emby_user_id'] = config.get('emby', 'user_id', fallback='')
        result['emby_token'] = config.get('emby', 'token', fallback='')
    
    return result
