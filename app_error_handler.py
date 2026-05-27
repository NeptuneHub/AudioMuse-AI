"""Centralized safe error messages for web UI responses."""

SETUP_ERROR_MESSAGES = {
    'MEDIA_SERVER_AUTH_FAILED': 'Media server credentials were rejected. Check the username, password, API token, and user ID.',
    'MEDIA_SERVER_UNREACHABLE': 'AudioMuse-AI could not reach the media server. Check the URL and make sure it is reachable from the AudioMuse-AI container.',
    'MEDIA_SERVER_EMPTY_LIBRARY': 'Connected to the media server, but no top-played songs were returned. Play a few tracks or check that this account can read the library.',
    'MEDIA_SERVER_CONFIG_INCOMPLETE': 'Media server configuration is incomplete. Fill in all required media-server fields and try again.',
    'MEDIA_SERVER_TEST_FAILED': 'Media server connection test failed. Check the configuration and try again.',
    'SETUP_SAVE_FAILED': 'Unable to save configuration. Check the server log for details.',
}


def classify_setup_error(exc):
    """Return a stable setup error code without exposing raw exception details."""
    text = str(exc or '').lower()
    if any(token in text for token in ('unauthorized', 'forbidden', 'invalid credentials', 'authentication', '401', '403')):
        return 'MEDIA_SERVER_AUTH_FAILED'
    if any(token in text for token in ('connection', 'connect', 'timeout', 'timed out', 'resolve', 'refused', 'unreachable')):
        return 'MEDIA_SERVER_UNREACHABLE'
    if any(token in text for token in ('no top-played', 'no top played', 'no items', 'no songs', 'empty')):
        return 'MEDIA_SERVER_EMPTY_LIBRARY'
    if any(token in text for token in ('incomplete', 'missing', 'required')):
        return 'MEDIA_SERVER_CONFIG_INCOMPLETE'
    return 'MEDIA_SERVER_TEST_FAILED'


def setup_error_payload(exc, default_code='SETUP_SAVE_FAILED'):
    """Build a JSON-safe error payload with a stable code and safe message."""
    if default_code == 'MEDIA_SERVER_TEST_FAILED':
        code = classify_setup_error(exc)
    else:
        code = default_code
    return {
        'error': SETUP_ERROR_MESSAGES.get(code, SETUP_ERROR_MESSAGES['SETUP_SAVE_FAILED']),
        'error_code': code,
    }
