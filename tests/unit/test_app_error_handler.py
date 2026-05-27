from app_error_handler import classify_setup_error, setup_error_payload


def test_setup_error_payload_uses_safe_unreachable_message():
    payload = setup_error_payload(
        TimeoutError('Connection timed out to http://internal-host:8096'),
        default_code='MEDIA_SERVER_TEST_FAILED',
    )

    assert payload == {
        'error': 'AudioMuse-AI could not reach the media server. Check the URL and make sure it is reachable from the AudioMuse-AI container.',
        'error_code': 'MEDIA_SERVER_UNREACHABLE',
    }


def test_setup_error_payload_uses_safe_auth_message():
    payload = setup_error_payload(
        PermissionError('401 Unauthorized: invalid credentials'),
        default_code='MEDIA_SERVER_TEST_FAILED',
    )

    assert payload['error_code'] == 'MEDIA_SERVER_AUTH_FAILED'
    assert 'credentials were rejected' in payload['error']


def test_classify_setup_error_handles_empty_library_probe():
    assert classify_setup_error(ValueError('No top-played songs were returned')) == 'MEDIA_SERVER_EMPTY_LIBRARY'
