import requests
import logging
import json
import time
import urllib.parse
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

class PocketBaseClient:
    """A client for interacting with the PocketBase API."""

    def __init__(self, base_url, email=None, password=None, token=None, log_prefix="[PocketBaseClient]"):
        self.base_url = base_url.rstrip('/')
        self.email = email
        self.password = password
        self.token = token
        self.log_prefix = log_prefix
        self.session = requests.Session()
        
        retry_strategy = Retry(
            total=3,
            status_forcelist=[429, 500, 502, 503, 504],
            backoff_factor=2  # e.g., 2s, 4s, 8s
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
        
        self.session.headers.update({"Content-Type": "application/json"})
        self.timeout = (15, 90)

        if self.token:
            logger.info(f"{self.log_prefix} Initialized with an existing token.")
            self.session.headers.update({"Authorization": self.token})

    def authenticate(self):
        """Authenticates with the PocketBase server and stores the token."""
        if not self.email or not self.password:
            raise ValueError("Authentication requires email and password.")

        auth_url = f"{self.base_url}/api/collections/users/auth-with-password"
        payload = {"identity": self.email, "password": self.password}
        
        try:
            logger.info(f"{self.log_prefix} Attempting authentication...")
            response = self.session.post(auth_url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            self.token = response.json().get('token')
            if not self.token:
                raise ConnectionError("Authentication successful but no token received.")
            self.session.headers.update({"Authorization": self.token})
            logger.info(f"{self.log_prefix} Successfully authenticated.")
            return True
        except requests.exceptions.RequestException as e:
            error_body = ""
            if hasattr(e, 'response') and e.response and hasattr(e.response, 'text'):
                try:
                    error_json = e.response.json()
                    error_body = json.dumps(error_json)
                except json.JSONDecodeError:
                    error_body = e.response.text
            else:
                error_body = "No response body"

            error_message = f"Failed to authenticate: {e} | Body: {error_body}"
            logger.error(f"{self.log_prefix} {error_message}")
            raise ConnectionError(error_message) from e

    def _make_request(self, method, endpoint, **kwargs):
        """Makes a request, handling re-authentication for password-based clients if necessary."""
        # MODIFIED: Only try to authenticate if not already authenticated via token.
        if 'Authorization' not in self.session.headers:
            if self.email and self.password:
                self.authenticate()
            else:
                raise ConnectionError("PocketBase client is not authenticated. Please provide a token or credentials.")

        kwargs.setdefault('timeout', self.timeout)
        url = f"{self.base_url}{endpoint}"

        try:
            response = self.session.request(method, url, **kwargs)
            
            # MODIFIED: Only re-authenticate if it's a password-based client
            if response.status_code == 401 and self.email and self.password:
                logger.warning(f"{self.log_prefix} Token may have expired. Re-authenticating with credentials...")
                self.authenticate()
                response = self.session.request(method, url, **kwargs)

            response.raise_for_status()
            return response.json() if response.content else {}
        except requests.exceptions.RequestException as e:
            logger.error(f"{self.log_prefix} API request failed for {method} {endpoint}: {e}")
            if hasattr(e, 'response') and e.response is not None:
                response_text = e.response.text
                if "already existing" in response_text:
                     logger.warning(f"{self.log_prefix} Batch request failed because one or more items already exist.")
                else:
                    logger.error(f"{self.log_prefix} Response status: {e.response.status_code}, body: {response_text}")
            raise

    def _sanitize_for_filter(self, value):
        return value.replace('\\', '\\\\').replace('"', '\\"')

    def get_records_by_artists(self, artists, collection):
        """Fetches all records for a list of artists using a GET request with a shorter filter."""
        if not artists:
            return []
        
        endpoint = f"/api/collections/{collection}/records"
        ARTIST_CHUNK_SIZE = 5
        all_records = []
        artist_chunks = [artists[i:i + ARTIST_CHUNK_SIZE] for i in range(0, len(artists), ARTIST_CHUNK_SIZE)]

        for chunk in artist_chunks:
            filter_parts = [f'artist="{self._sanitize_for_filter(artist)}"' for artist in chunk]
            filter_query = " || ".join(filter_parts)
            params = { 'filter': filter_query, 'perPage': 500 }
            
            try:
                response_data = self._make_request('GET', endpoint, params=params)
                all_records.extend(response_data.get('items', []))
            except requests.exceptions.RequestException as e:
                logger.error(f"{self.log_prefix} FATAL: Could not retrieve records from '{collection}' for artists. Reason: {e}")
                raise e
        
        return all_records

    def submit_batch_request(self, requests_payload):
        """
        Submits a generic batch request. This is transactional on the PocketBase side.
        If any request in the batch fails, the entire transaction is rolled back.
        """
        if not requests_payload:
            return True

        batch_endpoint = "/api/batch"
        payload = {"requests": requests_payload}
        
        try:
            self._make_request('POST', batch_endpoint, json=payload)
            logger.info(f"{self.log_prefix} Successfully submitted batch request for {len(requests_payload)} operations.")
        except requests.exceptions.RequestException:
            logger.error(f"{self.log_prefix} The entire batch request failed and was rolled back.")
            raise
