import requests as _requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

_CONNECT_RETRY = Retry(
    total=3,
    connect=3,
    read=0,
    status_forcelist=[],
    backoff_factor=1,
)


def _request(verb, *args, **kwargs):
    with _requests.Session() as s:
        adapter = HTTPAdapter(max_retries=_CONNECT_RETRY)
        s.mount("http://", adapter)
        s.mount("https://", adapter)
        return getattr(s, verb)(*args, **kwargs)


def get(*args, **kwargs):
    return _request("get", *args, **kwargs)


def post(*args, **kwargs):
    return _request("post", *args, **kwargs)


def put(*args, **kwargs):
    return _request("put", *args, **kwargs)


def delete(*args, **kwargs):
    return _request("delete", *args, **kwargs)


def head(*args, **kwargs):
    return _request("head", *args, **kwargs)


def patch(*args, **kwargs):
    return _request("patch", *args, **kwargs)


def request(*args, **kwargs):
    return _request("request", *args, **kwargs)


def __getattr__(name):
    return getattr(_requests, name)
