import socket
from unittest.mock import patch

import pytest

from ssrf_guard import validate_outbound_url


def _addrinfo(ip, port=80):
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, '', (ip, port))]


class TestValidateOutboundUrlSchemes:
    @pytest.mark.parametrize(
        'url',
        [
            'file:///etc/passwd',
            'gopher://10.0.0.1:6379/',
            'ftp://1.2.3.4/',
            'redis://1.2.3.4:6379',
            'ws://1.2.3.4/',
            'not-a-url',
        ],
    )
    def test_rejects_non_http_schemes(self, url):
        ok, reason = validate_outbound_url(url)
        assert ok is False
        assert reason == 'Only http and https URLs are supported'

    @pytest.mark.parametrize('url', ['http://8.8.8.8', 'https://8.8.8.8'])
    def test_accepts_http_and_https(self, url):
        with patch('ssrf_guard.socket.getaddrinfo', return_value=_addrinfo('8.8.8.8')):
            assert validate_outbound_url(url) == (True, None)


class TestValidateOutboundUrlMissingParts:
    @pytest.mark.parametrize('url', ['', None])
    def test_empty_url_rejected(self, url):
        ok, reason = validate_outbound_url(url)
        assert ok is False
        assert reason == 'URL is required'

    @pytest.mark.parametrize('url', ['http://', 'https://'])
    def test_host_less_url_rejected(self, url):
        ok, reason = validate_outbound_url(url)
        assert ok is False
        assert reason == 'URL host is required'


class TestValidateOutboundUrlIpClasses:
    @pytest.mark.parametrize(
        'url,ip',
        [
            ('http://127.0.0.1', '127.0.0.1'),
            ('http://127.0.0.1:8096', '127.0.0.1'),
        ],
    )
    def test_loopback_allowed(self, url, ip):
        with patch('ssrf_guard.socket.getaddrinfo', return_value=_addrinfo(ip)):
            assert validate_outbound_url(url) == (True, None)

    @pytest.mark.parametrize(
        'url,ip',
        [
            ('http://10.0.0.5/rest', '10.0.0.5'),
            ('http://172.16.3.4', '172.16.3.4'),
            ('http://192.168.1.50:8096', '192.168.1.50'),
        ],
    )
    def test_rfc1918_allowed(self, url, ip):
        with patch('ssrf_guard.socket.getaddrinfo', return_value=_addrinfo(ip)):
            assert validate_outbound_url(url) == (True, None)

    @pytest.mark.parametrize(
        'url,ip',
        [
            ('http://8.8.8.8', '8.8.8.8'),
            ('http://1.2.3.4:8096', '1.2.3.4'),
        ],
    )
    def test_public_ip_allowed(self, url, ip):
        with patch('ssrf_guard.socket.getaddrinfo', return_value=_addrinfo(ip)):
            assert validate_outbound_url(url) == (True, None)

    @pytest.mark.parametrize(
        'url,ip',
        [
            ('http://169.254.169.254/latest/meta-data', '169.254.169.254'),
            ('http://169.254.10.20', '169.254.10.20'),
            ('http://224.0.0.1', '224.0.0.1'),
            ('http://0.0.0.0', '0.0.0.0'),
            ('http://240.0.0.1', '240.0.0.1'),
        ],
    )
    def test_dangerous_ip_classes_rejected(self, url, ip):
        with patch('ssrf_guard.socket.getaddrinfo', return_value=_addrinfo(ip)):
            ok, reason = validate_outbound_url(url)
        assert ok is False
        assert reason == 'Target host resolves to a disallowed IP address'

    def test_dns_name_resolving_to_metadata_rejected(self):
        with patch('ssrf_guard.socket.getaddrinfo', return_value=_addrinfo('169.254.169.254')):
            ok, reason = validate_outbound_url('http://metadata.internal/')
        assert ok is False
        assert reason == 'Target host resolves to a disallowed IP address'

    def test_unresolvable_host_rejected(self):
        with patch('ssrf_guard.socket.getaddrinfo', side_effect=socket.gaierror('nope')):
            ok, reason = validate_outbound_url('http://does-not-exist.invalid/')
        assert ok is False
        assert reason == 'Could not resolve host'

    def test_resolved_invalid_ip_rejected(self):
        bad = [(socket.AF_INET, socket.SOCK_STREAM, 6, '', ('not-an-ip', 80))]
        with patch('ssrf_guard.socket.getaddrinfo', return_value=bad):
            ok, reason = validate_outbound_url('http://weird.host/')
        assert ok is False
        assert reason == 'Resolved host to invalid IP'
