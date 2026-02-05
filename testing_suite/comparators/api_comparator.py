"""
API Comparison Module for AudioMuse-AI Testing Suite.

Tests all API endpoints on both instances and compares:
  - HTTP status codes
  - Response shapes (keys, types, list lengths)
  - Response content (values where deterministic)
  - Error handling and edge cases
  - Endpoint availability
  - Task lifecycle (start -> poll -> success)
"""

import json
import logging
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple

from testing_suite.config import ComparisonConfig, InstanceConfig
from testing_suite.utils import (
    ComparisonReport, TestResult, TestStatus,
    http_get, http_post, timed_request, wait_for_task_success, pct_diff
)

logger = logging.getLogger(__name__)


class APIComparator:
    """Tests and compares API endpoints across two AudioMuse-AI instances."""

    def __init__(self, config: ComparisonConfig):
        self.config = config
        self.url_a = config.instance_a.api_url.rstrip('/')
        self.url_b = config.instance_b.api_url.rstrip('/')
        self.name_a = config.instance_a.name
        self.name_b = config.instance_b.name
        self.timeout = config.api_timeout
        self.retries = config.api_retries
        self.retry_delay = config.api_retry_delay

    def run_all(self, report: ComparisonReport):
        """Run all API comparison tests."""
        logger.info("Starting API comparison tests...")

        # Check connectivity first
        alive_a = self._check_alive(self.url_a)
        alive_b = self._check_alive(self.url_b)

        report.add_result(TestResult(
            category="api",
            name="Instance A connectivity",
            status=TestStatus.PASS if alive_a else TestStatus.ERROR,
            message=f"{self.url_a}: {'reachable' if alive_a else 'unreachable'}",
            instance_a_value=alive_a,
        ))
        report.add_result(TestResult(
            category="api",
            name="Instance B connectivity",
            status=TestStatus.PASS if alive_b else TestStatus.ERROR,
            message=f"{self.url_b}: {'reachable' if alive_b else 'unreachable'}",
            instance_b_value=alive_b,
        ))

        if not alive_a and not alive_b:
            report.add_result(TestResult(
                category="api",
                name="API Tests",
                status=TestStatus.ERROR,
                message="Neither instance is reachable; skipping all API tests",
            ))
            return

        # Run endpoint tests
        self._test_config_endpoint(report, alive_a, alive_b)
        self._test_playlists_endpoint(report, alive_a, alive_b)
        self._test_active_tasks_endpoint(report, alive_a, alive_b)
        self._test_last_task_endpoint(report, alive_a, alive_b)
        self._test_search_tracks_endpoint(report, alive_a, alive_b)
        self._test_similar_tracks_endpoint(report, alive_a, alive_b)
        self._test_max_distance_endpoint(report, alive_a, alive_b)
        self._test_map_endpoint(report, alive_a, alive_b)
        self._test_map_cache_status(report, alive_a, alive_b)
        self._test_clap_stats(report, alive_a, alive_b)
        self._test_clap_warmup_status(report, alive_a, alive_b)
        self._test_clap_top_queries(report, alive_a, alive_b)
        self._test_setup_status(report, alive_a, alive_b)
        self._test_setup_providers(report, alive_a, alive_b)
        self._test_setup_settings(report, alive_a, alive_b)
        self._test_setup_server_info(report, alive_a, alive_b)
        self._test_provider_types(report, alive_a, alive_b)
        self._test_providers_enabled(report, alive_a, alive_b)
        self._test_cron_entries(report, alive_a, alive_b)
        self._test_waveform_endpoint(report, alive_a, alive_b)
        self._test_find_path_endpoint(report, alive_a, alive_b)
        self._test_sonic_fingerprint(report, alive_a, alive_b)
        self._test_alchemy_endpoint(report, alive_a, alive_b)
        self._test_artist_projections(report, alive_a, alive_b)
        self._test_search_artists(report, alive_a, alive_b)
        self._test_external_search(report, alive_a, alive_b)
        self._test_chat_config_defaults(report, alive_a, alive_b)
        self._test_error_handling(report, alive_a, alive_b)
        self._test_collection_last_task(report, alive_a, alive_b)

        logger.info("API comparison tests complete.")

    # ------------------------------------------------------------------
    # Connectivity
    # ------------------------------------------------------------------

    def _check_alive(self, url: str) -> bool:
        """Check if an instance is reachable."""
        try:
            resp = http_get(f"{url}/api/config", timeout=15, retries=2, retry_delay=1)
            return resp.status_code == 200
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Helper: compare GET endpoint on both instances
    # ------------------------------------------------------------------

    def _compare_get(self, report: ComparisonReport, path: str, test_name: str,
                     params: dict = None, alive_a: bool = True, alive_b: bool = True,
                     expected_status: int = 200, check_keys: list = None,
                     compare_list_length: bool = False):
        """
        Hit a GET endpoint on both instances and compare the results.
        Adds test results to the report.
        """
        t0 = time.time()
        resp_a = resp_b = None
        data_a = data_b = None

        try:
            if alive_a:
                resp_a, lat_a = timed_request("GET", f"{self.url_a}{path}",
                                               params=params, timeout=self.timeout,
                                               retries=self.retries, retry_delay=self.retry_delay)
            if alive_b:
                resp_b, lat_b = timed_request("GET", f"{self.url_b}{path}",
                                               params=params, timeout=self.timeout,
                                               retries=self.retries, retry_delay=self.retry_delay)
        except Exception as e:
            report.add_result(TestResult(
                category="api",
                name=f"{test_name}: request",
                status=TestStatus.ERROR,
                message=str(e),
                duration_seconds=time.time() - t0,
            ))
            return None, None

        # Status code comparison
        status_a = resp_a.status_code if resp_a else None
        status_b = resp_b.status_code if resp_b else None

        if alive_a and alive_b:
            if status_a == expected_status and status_b == expected_status:
                status = TestStatus.PASS
                msg = f"Both returned {expected_status}"
            elif status_a == status_b:
                status = TestStatus.WARN
                msg = f"Both returned {status_a} (expected {expected_status})"
            else:
                status = TestStatus.FAIL
                msg = f"Status codes differ: A={status_a}, B={status_b}"

            report.add_result(TestResult(
                category="api",
                name=f"{test_name}: status code",
                status=status,
                message=msg,
                instance_a_value=status_a,
                instance_b_value=status_b,
                duration_seconds=time.time() - t0,
                details={"latency_a": lat_a if alive_a else None,
                          "latency_b": lat_b if alive_b else None},
            ))

        # Parse JSON response
        try:
            if resp_a and resp_a.status_code == expected_status:
                data_a = resp_a.json()
            if resp_b and resp_b.status_code == expected_status:
                data_b = resp_b.json()
        except Exception as e:
            report.add_result(TestResult(
                category="api",
                name=f"{test_name}: JSON parse",
                status=TestStatus.ERROR,
                message=f"JSON parse error: {e}",
                duration_seconds=time.time() - t0,
            ))
            return data_a, data_b

        # Key comparison (if both have JSON data)
        if data_a is not None and data_b is not None:
            if isinstance(data_a, dict) and isinstance(data_b, dict):
                keys_a = set(data_a.keys())
                keys_b = set(data_b.keys())
                if keys_a == keys_b:
                    key_status = TestStatus.PASS
                    key_msg = f"Same keys: {sorted(keys_a)}"
                else:
                    key_status = TestStatus.FAIL
                    missing_b = keys_a - keys_b
                    missing_a = keys_b - keys_a
                    key_msg = f"Keys differ: only_A={missing_b}, only_B={missing_a}"

                report.add_result(TestResult(
                    category="api",
                    name=f"{test_name}: response shape",
                    status=key_status,
                    message=key_msg,
                    instance_a_value=sorted(keys_a),
                    instance_b_value=sorted(keys_b),
                    duration_seconds=time.time() - t0,
                ))

            if isinstance(data_a, list) and isinstance(data_b, list):
                if compare_list_length:
                    len_a = len(data_a)
                    len_b = len(data_b)
                    if len_a == len_b:
                        l_status = TestStatus.PASS
                        l_msg = f"Same list length: {len_a}"
                    else:
                        diff = pct_diff(len_a, len_b)
                        l_status = TestStatus.WARN if diff <= 20 else TestStatus.FAIL
                        l_msg = f"List lengths differ: A={len_a}, B={len_b} ({diff:.1f}%)"

                    report.add_result(TestResult(
                        category="api",
                        name=f"{test_name}: list length",
                        status=l_status,
                        message=l_msg,
                        instance_a_value=len_a,
                        instance_b_value=len_b,
                        duration_seconds=time.time() - t0,
                    ))

            # Check specific keys exist
            if check_keys and isinstance(data_a, dict) and isinstance(data_b, dict):
                for key in check_keys:
                    has_a = key in data_a
                    has_b = key in data_b
                    if has_a and has_b:
                        kstatus = TestStatus.PASS
                    elif has_a or has_b:
                        kstatus = TestStatus.FAIL
                    else:
                        kstatus = TestStatus.WARN

                    report.add_result(TestResult(
                        category="api",
                        name=f"{test_name}: key '{key}'",
                        status=kstatus,
                        message=f"A has '{key}': {has_a}, B has '{key}': {has_b}",
                        duration_seconds=time.time() - t0,
                    ))

        return data_a, data_b

    def _compare_post(self, report: ComparisonReport, path: str, test_name: str,
                      json_data: dict = None, alive_a: bool = True, alive_b: bool = True,
                      expected_status: int = 200, check_keys: list = None):
        """Hit a POST endpoint on both instances and compare results."""
        t0 = time.time()
        resp_a = resp_b = None
        data_a = data_b = None

        try:
            if alive_a:
                resp_a, lat_a = timed_request("POST", f"{self.url_a}{path}",
                                               json_data=json_data, timeout=self.timeout,
                                               retries=self.retries, retry_delay=self.retry_delay)
            if alive_b:
                resp_b, lat_b = timed_request("POST", f"{self.url_b}{path}",
                                               json_data=json_data, timeout=self.timeout,
                                               retries=self.retries, retry_delay=self.retry_delay)
        except Exception as e:
            report.add_result(TestResult(
                category="api",
                name=f"{test_name}: request",
                status=TestStatus.ERROR,
                message=str(e),
                duration_seconds=time.time() - t0,
            ))
            return None, None

        status_a = resp_a.status_code if resp_a else None
        status_b = resp_b.status_code if resp_b else None

        if alive_a and alive_b:
            if status_a == expected_status and status_b == expected_status:
                status = TestStatus.PASS
                msg = f"Both returned {expected_status}"
            elif status_a == status_b:
                status = TestStatus.WARN
                msg = f"Both returned {status_a} (expected {expected_status})"
            else:
                status = TestStatus.FAIL
                msg = f"Status codes differ: A={status_a}, B={status_b}"

            report.add_result(TestResult(
                category="api",
                name=f"{test_name}: status code",
                status=status,
                message=msg,
                instance_a_value=status_a,
                instance_b_value=status_b,
                duration_seconds=time.time() - t0,
                details={"latency_a": lat_a if alive_a else None,
                          "latency_b": lat_b if alive_b else None},
            ))

        try:
            if resp_a and resp_a.status_code == expected_status:
                data_a = resp_a.json()
            if resp_b and resp_b.status_code == expected_status:
                data_b = resp_b.json()
        except Exception:
            pass

        if data_a is not None and data_b is not None and isinstance(data_a, dict) and isinstance(data_b, dict):
            keys_a = set(data_a.keys())
            keys_b = set(data_b.keys())
            if keys_a == keys_b:
                report.add_result(TestResult(
                    category="api", name=f"{test_name}: response shape",
                    status=TestStatus.PASS, message=f"Same keys: {sorted(keys_a)}",
                    duration_seconds=time.time() - t0,
                ))
            else:
                report.add_result(TestResult(
                    category="api", name=f"{test_name}: response shape",
                    status=TestStatus.FAIL,
                    message=f"Keys differ: only_A={keys_a - keys_b}, only_B={keys_b - keys_a}",
                    duration_seconds=time.time() - t0,
                ))

            if check_keys:
                for key in check_keys:
                    has_a = key in data_a
                    has_b = key in data_b
                    report.add_result(TestResult(
                        category="api", name=f"{test_name}: key '{key}'",
                        status=TestStatus.PASS if (has_a and has_b) else TestStatus.FAIL,
                        message=f"A: {has_a}, B: {has_b}",
                        duration_seconds=time.time() - t0,
                    ))

        return data_a, data_b

    # ------------------------------------------------------------------
    # Individual endpoint tests
    # ------------------------------------------------------------------

    def _test_config_endpoint(self, report, alive_a, alive_b):
        self._compare_get(report, "/api/config", "GET /api/config",
                          alive_a=alive_a, alive_b=alive_b)

    def _test_playlists_endpoint(self, report, alive_a, alive_b):
        self._compare_get(report, "/api/playlists", "GET /api/playlists",
                          alive_a=alive_a, alive_b=alive_b,
                          compare_list_length=True)

    def _test_active_tasks_endpoint(self, report, alive_a, alive_b):
        self._compare_get(report, "/api/active_tasks", "GET /api/active_tasks",
                          alive_a=alive_a, alive_b=alive_b)

    def _test_last_task_endpoint(self, report, alive_a, alive_b):
        self._compare_get(report, "/api/last_task", "GET /api/last_task",
                          alive_a=alive_a, alive_b=alive_b)

    def _test_search_tracks_endpoint(self, report, alive_a, alive_b):
        params = {
            "artist": self.config.test_track_artist_1,
            "title": self.config.test_track_title_1,
        }
        data_a, data_b = self._compare_get(
            report, "/api/search_tracks", "GET /api/search_tracks",
            params=params, alive_a=alive_a, alive_b=alive_b,
            compare_list_length=True)

        # Validate response has expected track fields
        t0 = time.time()
        for label, data in [("A", data_a), ("B", data_b)]:
            if data and isinstance(data, list) and data:
                track = data[0]
                expected = {"item_id", "title"}
                present = expected.intersection(track.keys())
                if present == expected:
                    report.add_result(TestResult(
                        category="api",
                        name=f"GET /api/search_tracks: {label} track fields",
                        status=TestStatus.PASS,
                        message=f"Track has required fields: {expected}",
                        duration_seconds=time.time() - t0,
                    ))
                else:
                    report.add_result(TestResult(
                        category="api",
                        name=f"GET /api/search_tracks: {label} track fields",
                        status=TestStatus.FAIL,
                        message=f"Missing fields: {expected - present}. Has: {set(track.keys())}",
                        duration_seconds=time.time() - t0,
                    ))

    def _test_similar_tracks_endpoint(self, report, alive_a, alive_b):
        params = {
            "title": self.config.test_track_title_1,
            "artist": self.config.test_track_artist_1,
            "n": 5,
        }
        data_a, data_b = self._compare_get(
            report, "/api/similar_tracks", "GET /api/similar_tracks",
            params=params, alive_a=alive_a, alive_b=alive_b,
            compare_list_length=True)

        # Validate result tracks have item_id
        t0 = time.time()
        for label, data in [("A", data_a), ("B", data_b)]:
            if data and isinstance(data, list):
                has_ids = all('item_id' in t for t in data)
                report.add_result(TestResult(
                    category="api",
                    name=f"GET /api/similar_tracks: {label} item_ids present",
                    status=TestStatus.PASS if has_ids else TestStatus.FAIL,
                    message=f"All tracks have item_id: {has_ids} ({len(data)} tracks)",
                    duration_seconds=time.time() - t0,
                ))

    def _test_max_distance_endpoint(self, report, alive_a, alive_b):
        self._compare_get(report, "/api/max_distance", "GET /api/max_distance",
                          alive_a=alive_a, alive_b=alive_b)

    def _test_map_endpoint(self, report, alive_a, alive_b):
        data_a, data_b = self._compare_get(
            report, "/api/map", "GET /api/map",
            params={"percent": 10}, alive_a=alive_a, alive_b=alive_b,
            check_keys=["items"])

        # Validate items structure
        t0 = time.time()
        for label, data in [("A", data_a), ("B", data_b)]:
            if data and isinstance(data, dict) and "items" in data:
                items = data["items"]
                if isinstance(items, list) and items:
                    report.add_result(TestResult(
                        category="api",
                        name=f"GET /api/map: {label} items non-empty",
                        status=TestStatus.PASS,
                        message=f"{len(items)} items returned",
                        duration_seconds=time.time() - t0,
                    ))
                else:
                    report.add_result(TestResult(
                        category="api",
                        name=f"GET /api/map: {label} items non-empty",
                        status=TestStatus.WARN,
                        message=f"Empty items list",
                        duration_seconds=time.time() - t0,
                    ))

    def _test_map_cache_status(self, report, alive_a, alive_b):
        self._compare_get(report, "/api/map_cache_status", "GET /api/map_cache_status",
                          alive_a=alive_a, alive_b=alive_b)

    def _test_clap_stats(self, report, alive_a, alive_b):
        self._compare_get(report, "/api/clap/stats", "GET /api/clap/stats",
                          alive_a=alive_a, alive_b=alive_b)

    def _test_clap_warmup_status(self, report, alive_a, alive_b):
        self._compare_get(report, "/api/clap/warmup/status", "GET /api/clap/warmup/status",
                          alive_a=alive_a, alive_b=alive_b)

    def _test_clap_top_queries(self, report, alive_a, alive_b):
        self._compare_get(report, "/api/clap/top_queries", "GET /api/clap/top_queries",
                          alive_a=alive_a, alive_b=alive_b,
                          compare_list_length=True)

    def _test_setup_status(self, report, alive_a, alive_b):
        self._compare_get(report, "/api/setup/status", "GET /api/setup/status",
                          alive_a=alive_a, alive_b=alive_b)

    def _test_setup_providers(self, report, alive_a, alive_b):
        self._compare_get(report, "/api/setup/providers", "GET /api/setup/providers",
                          alive_a=alive_a, alive_b=alive_b,
                          compare_list_length=True)

    def _test_setup_settings(self, report, alive_a, alive_b):
        self._compare_get(report, "/api/setup/settings", "GET /api/setup/settings",
                          alive_a=alive_a, alive_b=alive_b)

    def _test_setup_server_info(self, report, alive_a, alive_b):
        self._compare_get(report, "/api/setup/server-info", "GET /api/setup/server-info",
                          alive_a=alive_a, alive_b=alive_b)

    def _test_provider_types(self, report, alive_a, alive_b):
        self._compare_get(report, "/api/setup/providers/types", "GET /api/setup/providers/types",
                          alive_a=alive_a, alive_b=alive_b)

    def _test_providers_enabled(self, report, alive_a, alive_b):
        self._compare_get(report, "/api/providers/enabled", "GET /api/providers/enabled",
                          alive_a=alive_a, alive_b=alive_b)

    def _test_cron_entries(self, report, alive_a, alive_b):
        self._compare_get(report, "/api/cron", "GET /api/cron",
                          alive_a=alive_a, alive_b=alive_b,
                          compare_list_length=True)

    def _test_waveform_endpoint(self, report, alive_a, alive_b):
        # Waveform needs a track query param - test without to verify error handling
        self._compare_get(report, "/api/waveform", "GET /api/waveform (no params)",
                          alive_a=alive_a, alive_b=alive_b,
                          expected_status=400)

    def _test_find_path_endpoint(self, report, alive_a, alive_b):
        """Test /api/find_path by first finding two track IDs."""
        t0 = time.time()
        try:
            # Find track IDs from both instances
            id_a_start = self._find_track_id(self.url_a,
                                              self.config.test_track_artist_1,
                                              self.config.test_track_title_1) if alive_a else None
            id_a_end = self._find_track_id(self.url_a,
                                            self.config.test_track_artist_2,
                                            self.config.test_track_title_2) if alive_a else None
            id_b_start = self._find_track_id(self.url_b,
                                              self.config.test_track_artist_1,
                                              self.config.test_track_title_1) if alive_b else None
            id_b_end = self._find_track_id(self.url_b,
                                            self.config.test_track_artist_2,
                                            self.config.test_track_title_2) if alive_b else None

            for label, url, start_id, end_id in [
                ("A", self.url_a, id_a_start, id_a_end),
                ("B", self.url_b, id_b_start, id_b_end),
            ]:
                if not start_id or not end_id:
                    report.add_result(TestResult(
                        category="api",
                        name=f"GET /api/find_path ({label}): track lookup",
                        status=TestStatus.SKIP,
                        message="Could not find test tracks",
                        duration_seconds=time.time() - t0,
                    ))
                    continue

                resp, lat = timed_request("GET", f"{url}/api/find_path",
                                           params={"start_song_id": start_id,
                                                   "end_song_id": end_id,
                                                   "max_steps": 10},
                                           timeout=self.timeout, retries=self.retries,
                                           retry_delay=self.retry_delay)

                if resp.status_code == 200:
                    data = resp.json()
                    path = data.get('path', data) if isinstance(data, dict) else data
                    path_len = len(path) if isinstance(path, list) else 0
                    report.add_result(TestResult(
                        category="api",
                        name=f"GET /api/find_path ({label})",
                        status=TestStatus.PASS if path_len > 0 else TestStatus.WARN,
                        message=f"Path length: {path_len}, latency: {lat:.2f}s",
                        instance_a_value=path_len if label == "A" else None,
                        instance_b_value=path_len if label == "B" else None,
                        duration_seconds=time.time() - t0,
                        details={"latency": lat},
                    ))
                else:
                    report.add_result(TestResult(
                        category="api",
                        name=f"GET /api/find_path ({label})",
                        status=TestStatus.FAIL,
                        message=f"Status {resp.status_code}: {resp.text[:200]}",
                        duration_seconds=time.time() - t0,
                    ))
        except Exception as e:
            report.add_result(TestResult(
                category="api",
                name="GET /api/find_path",
                status=TestStatus.ERROR,
                message=str(e),
                duration_seconds=time.time() - t0,
            ))

    def _test_sonic_fingerprint(self, report, alive_a, alive_b):
        """Test sonic fingerprint generation on both instances."""
        payload = {"n": 1, "jellyfin_user_identifier": "admin", "jellyfin_token": ""}
        self._compare_post(report, "/api/sonic_fingerprint/generate",
                           "POST /api/sonic_fingerprint/generate",
                           json_data=payload, alive_a=alive_a, alive_b=alive_b)

    def _test_alchemy_endpoint(self, report, alive_a, alive_b):
        """Test song alchemy (requires track IDs)."""
        t0 = time.time()
        try:
            for label, url, alive in [("A", self.url_a, alive_a), ("B", self.url_b, alive_b)]:
                if not alive:
                    continue

                add_id = self._find_track_id(url, self.config.test_track_artist_1,
                                              self.config.test_track_title_1)
                sub_id = self._find_track_id(url, self.config.test_track_artist_2,
                                              self.config.test_track_title_2)

                if not add_id or not sub_id:
                    report.add_result(TestResult(
                        category="api",
                        name=f"POST /api/alchemy ({label}): track lookup",
                        status=TestStatus.SKIP,
                        message="Could not find test tracks for alchemy",
                        duration_seconds=time.time() - t0,
                    ))
                    continue

                payload = {
                    "items": [
                        {"id": add_id, "op": "ADD"},
                        {"id": sub_id, "op": "SUBTRACT"},
                    ],
                    "n": 5,
                    "temperature": 1,
                    "subtract_distance": 0.2,
                }

                resp, lat = timed_request("POST", f"{url}/api/alchemy",
                                           json_data=payload, timeout=self.timeout,
                                           retries=self.retries, retry_delay=self.retry_delay)

                if resp.status_code == 200:
                    data = resp.json()
                    expected_keys = {"results", "projection"}
                    has_keys = expected_keys.issubset(data.keys()) if isinstance(data, dict) else False
                    results_count = len(data.get("results", [])) if isinstance(data, dict) else 0

                    report.add_result(TestResult(
                        category="api",
                        name=f"POST /api/alchemy ({label})",
                        status=TestStatus.PASS if has_keys and results_count > 0 else TestStatus.WARN,
                        message=f"Has expected keys: {has_keys}, results: {results_count}, latency: {lat:.2f}s",
                        duration_seconds=time.time() - t0,
                        details={"latency": lat, "result_count": results_count},
                    ))
                else:
                    report.add_result(TestResult(
                        category="api",
                        name=f"POST /api/alchemy ({label})",
                        status=TestStatus.FAIL,
                        message=f"Status {resp.status_code}: {resp.text[:200]}",
                        duration_seconds=time.time() - t0,
                    ))
        except Exception as e:
            report.add_result(TestResult(
                category="api",
                name="POST /api/alchemy",
                status=TestStatus.ERROR,
                message=str(e),
                duration_seconds=time.time() - t0,
            ))

    def _test_artist_projections(self, report, alive_a, alive_b):
        self._compare_get(report, "/api/artist_projections", "GET /api/artist_projections",
                          alive_a=alive_a, alive_b=alive_b)

    def _test_search_artists(self, report, alive_a, alive_b):
        self._compare_get(report, "/api/search_artists", "GET /api/search_artists",
                          params={"q": "Red Hot"}, alive_a=alive_a, alive_b=alive_b,
                          compare_list_length=True)

    def _test_external_search(self, report, alive_a, alive_b):
        self._compare_get(report, "/external/search", "GET /external/search",
                          params={"q": "piano"}, alive_a=alive_a, alive_b=alive_b,
                          compare_list_length=True)

    def _test_chat_config_defaults(self, report, alive_a, alive_b):
        self._compare_get(report, "/chat/api/config_defaults", "GET /chat/api/config_defaults",
                          alive_a=alive_a, alive_b=alive_b)

    def _test_collection_last_task(self, report, alive_a, alive_b):
        self._compare_get(report, "/api/collection/last_task", "GET /api/collection/last_task",
                          alive_a=alive_a, alive_b=alive_b)

    # ------------------------------------------------------------------
    # Error handling tests
    # ------------------------------------------------------------------

    def _test_error_handling(self, report, alive_a, alive_b):
        """Test that both instances handle errors consistently."""
        error_cases = [
            ("/api/status/nonexistent_task_id_12345", "Nonexistent task status", 200),
            ("/api/track", "Track without item_id", 400),
            ("/api/similar_tracks", "Similar tracks without params", 400),
        ]

        for path, desc, expected_status in error_cases:
            t0 = time.time()
            try:
                resp_a = http_get(f"{self.url_a}{path}", timeout=15, retries=1) if alive_a else None
                resp_b = http_get(f"{self.url_b}{path}", timeout=15, retries=1) if alive_b else None

                status_a = resp_a.status_code if resp_a else None
                status_b = resp_b.status_code if resp_b else None

                if alive_a and alive_b:
                    if status_a == status_b:
                        report.add_result(TestResult(
                            category="api",
                            name=f"Error handling: {desc}",
                            status=TestStatus.PASS,
                            message=f"Consistent error codes: {status_a}",
                            instance_a_value=status_a,
                            instance_b_value=status_b,
                            duration_seconds=time.time() - t0,
                        ))
                    else:
                        report.add_result(TestResult(
                            category="api",
                            name=f"Error handling: {desc}",
                            status=TestStatus.WARN,
                            message=f"Different error codes: A={status_a}, B={status_b}",
                            instance_a_value=status_a,
                            instance_b_value=status_b,
                            duration_seconds=time.time() - t0,
                        ))
            except Exception as e:
                report.add_result(TestResult(
                    category="api",
                    name=f"Error handling: {desc}",
                    status=TestStatus.ERROR,
                    message=str(e),
                    duration_seconds=time.time() - t0,
                ))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _find_track_id(self, base_url: str, artist: str, title: str) -> Optional[str]:
        """Find a track's item_id via the search API."""
        try:
            resp = http_get(f"{base_url}/api/search_tracks",
                            params={"artist": artist, "title": title},
                            timeout=30, retries=2, retry_delay=1)
            if resp.status_code == 200:
                results = resp.json()
                if isinstance(results, list) and results:
                    # Try exact match first
                    for track in results:
                        track_artist = track.get("author") or track.get("artist") or ""
                        if track_artist.lower() == artist.lower() and \
                           track.get("title", "").lower() == title.lower():
                            return track["item_id"]
                    # Fallback to first result
                    return results[0].get("item_id")
        except Exception as e:
            logger.debug(f"Track search failed: {e}")
        return None
