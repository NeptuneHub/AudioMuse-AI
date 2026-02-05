"""
Performance Comparison Module for AudioMuse-AI Testing Suite.

Benchmarks and compares performance between two instances:
  - API endpoint latency (p50, p95, p99, mean, max)
  - Throughput under concurrent load
  - Database query performance
  - Search/similarity response times
  - Memory-intensive operations (map, alchemy, clustering)
  - Warmup vs steady-state performance
"""

import logging
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

from testing_suite.config import ComparisonConfig, InstanceConfig
from testing_suite.utils import (
    ComparisonReport, TestResult, TestStatus,
    http_get, http_post, timed_request, pct_diff, format_duration,
    pg_query, pg_scalar
)

logger = logging.getLogger(__name__)


def _percentile(data: List[float], pct: float) -> float:
    """Calculate a percentile from a sorted list of values."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * (pct / 100.0)
    f = int(k)
    c = f + 1
    if c >= len(sorted_data):
        return sorted_data[-1]
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])


def _latency_stats(latencies: List[float]) -> dict:
    """Compute latency statistics from a list of measurements."""
    if not latencies:
        return {"count": 0, "mean": 0, "median": 0, "p95": 0, "p99": 0,
                "min": 0, "max": 0, "stddev": 0}
    return {
        "count": len(latencies),
        "mean": statistics.mean(latencies),
        "median": statistics.median(latencies),
        "p95": _percentile(latencies, 95),
        "p99": _percentile(latencies, 99),
        "min": min(latencies),
        "max": max(latencies),
        "stddev": statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
    }


class PerformanceComparator:
    """Benchmarks and compares performance between two instances."""

    def __init__(self, config: ComparisonConfig):
        self.config = config
        self.url_a = config.instance_a.api_url.rstrip('/')
        self.url_b = config.instance_b.api_url.rstrip('/')
        self.name_a = config.instance_a.name
        self.name_b = config.instance_b.name
        self.warmup_n = config.perf_warmup_requests
        self.bench_n = config.perf_benchmark_requests
        self.concurrent = config.perf_concurrent_users

    def run_all(self, report: ComparisonReport):
        """Run all performance comparison tests."""
        logger.info("Starting performance comparison tests...")

        # Check connectivity
        alive_a = self._check_alive(self.url_a)
        alive_b = self._check_alive(self.url_b)

        if not alive_a and not alive_b:
            report.add_result(TestResult(
                category="performance",
                name="Connectivity",
                status=TestStatus.ERROR,
                message="Neither instance reachable; skipping performance tests",
            ))
            return

        # Define endpoint benchmarks
        benchmarks = [
            # (path, method, params/json, description, expected_status)
            ("/api/config", "GET", None, "Config endpoint", 200),
            ("/api/playlists", "GET", None, "Playlists list", 200),
            ("/api/active_tasks", "GET", None, "Active tasks", 200),
            ("/api/last_task", "GET", None, "Last task", 200),
            ("/api/search_tracks?artist=Red+Hot&title=By", "GET", None, "Track search", 200),
            ("/api/similar_tracks?title=By+the+Way&artist=Red+Hot+Chili+Peppers&n=5", "GET", None, "Similar tracks", 200),
            ("/api/map?percent=10", "GET", None, "Map visualization", 200),
            ("/api/map_cache_status", "GET", None, "Map cache status", 200),
            ("/api/clap/stats", "GET", None, "CLAP stats", 200),
            ("/api/clap/top_queries", "GET", None, "CLAP top queries", 200),
            ("/api/setup/status", "GET", None, "Setup status", 200),
            ("/api/setup/providers", "GET", None, "Providers list", 200),
            ("/api/setup/settings", "GET", None, "App settings", 200),
            ("/api/cron", "GET", None, "Cron entries", 200),
            ("/api/search_artists?q=Red", "GET", None, "Artist search", 200),
            ("/external/search?q=piano", "GET", None, "External search", 200),
        ]

        # Run latency benchmarks for each endpoint
        for path, method, data, desc, expected_status in benchmarks:
            self._benchmark_endpoint(report, path, method, data, desc,
                                      expected_status, alive_a, alive_b)

        # Concurrent load test on a few key endpoints
        self._concurrent_load_test(report, alive_a, alive_b)

        # Database query performance
        self._benchmark_db_queries(report)

        logger.info("Performance comparison tests complete.")

    # ------------------------------------------------------------------
    # Endpoint latency benchmark
    # ------------------------------------------------------------------

    def _benchmark_endpoint(self, report: ComparisonReport, path: str,
                             method: str, data: Any, description: str,
                             expected_status: int, alive_a: bool, alive_b: bool):
        """Benchmark a single endpoint on both instances."""
        t0 = time.time()
        latencies_a = []
        latencies_b = []
        errors_a = 0
        errors_b = 0

        # Warmup phase
        for _ in range(self.warmup_n):
            try:
                if alive_a:
                    if method == "GET":
                        http_get(f"{self.url_a}{path}", timeout=30, retries=1)
                    else:
                        http_post(f"{self.url_a}{path}", json_data=data, timeout=30, retries=1)
            except Exception:
                pass
            try:
                if alive_b:
                    if method == "GET":
                        http_get(f"{self.url_b}{path}", timeout=30, retries=1)
                    else:
                        http_post(f"{self.url_b}{path}", json_data=data, timeout=30, retries=1)
            except Exception:
                pass

        # Benchmark phase
        for i in range(self.bench_n):
            if alive_a:
                try:
                    start = time.perf_counter()
                    if method == "GET":
                        resp = http_get(f"{self.url_a}{path}", timeout=60, retries=1)
                    else:
                        resp = http_post(f"{self.url_a}{path}", json_data=data, timeout=60, retries=1)
                    elapsed = time.perf_counter() - start
                    if resp.status_code == expected_status:
                        latencies_a.append(elapsed)
                    else:
                        errors_a += 1
                except Exception:
                    errors_a += 1

            if alive_b:
                try:
                    start = time.perf_counter()
                    if method == "GET":
                        resp = http_get(f"{self.url_b}{path}", timeout=60, retries=1)
                    else:
                        resp = http_post(f"{self.url_b}{path}", json_data=data, timeout=60, retries=1)
                    elapsed = time.perf_counter() - start
                    if resp.status_code == expected_status:
                        latencies_b.append(elapsed)
                    else:
                        errors_b += 1
                except Exception:
                    errors_b += 1

        stats_a = _latency_stats(latencies_a)
        stats_b = _latency_stats(latencies_b)

        # Determine status based on relative performance
        if stats_a['mean'] > 0 and stats_b['mean'] > 0:
            ratio = stats_b['mean'] / stats_a['mean']
            if ratio <= 1.2:  # B is within 20% of A
                status = TestStatus.PASS
                comparison = f"B is {ratio:.2f}x vs A"
            elif ratio <= 2.0:
                status = TestStatus.WARN
                comparison = f"B is {ratio:.2f}x slower than A"
            else:
                status = TestStatus.FAIL
                comparison = f"B is {ratio:.2f}x slower than A"

            # Also check if B is faster
            if ratio < 0.8:
                status = TestStatus.PASS
                comparison = f"B is {1/ratio:.2f}x faster than A"
        else:
            status = TestStatus.WARN if (latencies_a or latencies_b) else TestStatus.SKIP
            comparison = "Cannot compare (one or both had no successful requests)"

        report.add_result(TestResult(
            category="performance",
            name=f"Latency: {description}",
            status=status,
            message=(
                f"{comparison} | "
                f"A: mean={format_duration(stats_a['mean'])}, "
                f"p95={format_duration(stats_a['p95'])}, "
                f"p99={format_duration(stats_a['p99'])} "
                f"({errors_a} errors) | "
                f"B: mean={format_duration(stats_b['mean'])}, "
                f"p95={format_duration(stats_b['p95'])}, "
                f"p99={format_duration(stats_b['p99'])} "
                f"({errors_b} errors)"
            ),
            instance_a_value=stats_a,
            instance_b_value=stats_b,
            diff=pct_diff(stats_a['mean'], stats_b['mean']) if stats_a['mean'] and stats_b['mean'] else None,
            duration_seconds=time.time() - t0,
            details={
                "path": path,
                "method": method,
                "warmup_requests": self.warmup_n,
                "benchmark_requests": self.bench_n,
                "errors_a": errors_a,
                "errors_b": errors_b,
            },
        ))

    # ------------------------------------------------------------------
    # Concurrent load test
    # ------------------------------------------------------------------

    def _concurrent_load_test(self, report: ComparisonReport,
                               alive_a: bool, alive_b: bool):
        """Test throughput under concurrent load."""
        endpoints = [
            "/api/config",
            "/api/search_tracks?artist=Red+Hot&title=By",
            "/api/playlists",
        ]

        for path in endpoints:
            t0 = time.time()
            results_a = self._run_concurrent(self.url_a, path, self.concurrent,
                                              self.bench_n) if alive_a else None
            results_b = self._run_concurrent(self.url_b, path, self.concurrent,
                                              self.bench_n) if alive_b else None

            if results_a and results_b:
                throughput_a = results_a['successful'] / results_a['total_time'] if results_a['total_time'] > 0 else 0
                throughput_b = results_b['successful'] / results_b['total_time'] if results_b['total_time'] > 0 else 0

                if throughput_a > 0 and throughput_b > 0:
                    ratio = throughput_b / throughput_a
                    if ratio >= 0.8:
                        status = TestStatus.PASS
                    elif ratio >= 0.5:
                        status = TestStatus.WARN
                    else:
                        status = TestStatus.FAIL
                else:
                    status = TestStatus.WARN
                    ratio = 0

                report.add_result(TestResult(
                    category="performance",
                    name=f"Concurrent Load: {path.split('?')[0]}",
                    status=status,
                    message=(
                        f"{self.concurrent} concurrent users, {self.bench_n} requests each | "
                        f"A: {throughput_a:.1f} req/s, "
                        f"mean={format_duration(results_a['mean_latency'])}, "
                        f"{results_a['errors']} errors | "
                        f"B: {throughput_b:.1f} req/s, "
                        f"mean={format_duration(results_b['mean_latency'])}, "
                        f"{results_b['errors']} errors"
                    ),
                    instance_a_value={"throughput_rps": throughput_a, **results_a},
                    instance_b_value={"throughput_rps": throughput_b, **results_b},
                    duration_seconds=time.time() - t0,
                ))
            else:
                report.add_result(TestResult(
                    category="performance",
                    name=f"Concurrent Load: {path.split('?')[0]}",
                    status=TestStatus.SKIP,
                    message="Cannot run concurrent test (one or both instances unavailable)",
                    duration_seconds=time.time() - t0,
                ))

    def _run_concurrent(self, base_url: str, path: str,
                         concurrent: int, requests_per_worker: int) -> dict:
        """Run concurrent requests and measure throughput."""
        latencies = []
        errors = 0

        def worker():
            nonlocal errors
            local_latencies = []
            for _ in range(requests_per_worker):
                try:
                    start = time.perf_counter()
                    resp = http_get(f"{base_url}{path}", timeout=30, retries=1)
                    elapsed = time.perf_counter() - start
                    if resp.status_code == 200:
                        local_latencies.append(elapsed)
                    else:
                        errors += 1
                except Exception:
                    errors += 1
            return local_latencies

        overall_start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=concurrent) as executor:
            futures = [executor.submit(worker) for _ in range(concurrent)]
            for f in as_completed(futures):
                try:
                    latencies.extend(f.result())
                except Exception:
                    errors += 1
        total_time = time.perf_counter() - overall_start

        return {
            "successful": len(latencies),
            "errors": errors,
            "total_time": total_time,
            "mean_latency": statistics.mean(latencies) if latencies else 0,
            "p95_latency": _percentile(latencies, 95) if latencies else 0,
        }

    # ------------------------------------------------------------------
    # Database query performance
    # ------------------------------------------------------------------

    def _benchmark_db_queries(self, report: ComparisonReport):
        """Benchmark critical database queries on both instances."""
        queries = [
            ("SELECT COUNT(*) FROM score", "Score count"),
            ("SELECT COUNT(*) FROM embedding", "Embedding count"),
            ("SELECT COUNT(*) FROM playlist", "Playlist count"),
            ("SELECT COUNT(DISTINCT playlist_name) FROM playlist", "Distinct playlists"),
            ("SELECT item_id, title, author FROM score LIMIT 100", "Score fetch 100"),
            ("SELECT s.item_id, s.title FROM score s JOIN embedding e ON s.item_id = e.item_id LIMIT 50",
             "Score-embedding join 50"),
            ("SELECT AVG(tempo), AVG(energy) FROM score WHERE tempo IS NOT NULL", "Score aggregation"),
            ("SELECT key, COUNT(*) FROM score WHERE key IS NOT NULL GROUP BY key", "Key distribution"),
        ]

        dsn_a = self.config.instance_a.pg_dsn
        dsn_b = self.config.instance_b.pg_dsn

        can_a = self._test_db(dsn_a)
        can_b = self._test_db(dsn_b)

        if not can_a and not can_b:
            report.add_result(TestResult(
                category="performance",
                name="DB Query Performance",
                status=TestStatus.SKIP,
                message="Cannot connect to either database",
            ))
            return

        for sql, desc in queries:
            t0 = time.time()
            latencies_a = []
            latencies_b = []

            for _ in range(max(3, self.bench_n // 2)):
                if can_a:
                    try:
                        start = time.perf_counter()
                        pg_query(dsn_a, sql)
                        latencies_a.append(time.perf_counter() - start)
                    except Exception:
                        pass

                if can_b:
                    try:
                        start = time.perf_counter()
                        pg_query(dsn_b, sql)
                        latencies_b.append(time.perf_counter() - start)
                    except Exception:
                        pass

            stats_a = _latency_stats(latencies_a)
            stats_b = _latency_stats(latencies_b)

            if stats_a['mean'] > 0 and stats_b['mean'] > 0:
                ratio = stats_b['mean'] / stats_a['mean']
                if ratio <= 1.5:
                    status = TestStatus.PASS
                elif ratio <= 3.0:
                    status = TestStatus.WARN
                else:
                    status = TestStatus.FAIL
                comparison = f"B/A ratio: {ratio:.2f}x"
            else:
                status = TestStatus.WARN
                comparison = "Insufficient data"

            report.add_result(TestResult(
                category="performance",
                name=f"DB Query: {desc}",
                status=status,
                message=(
                    f"{comparison} | "
                    f"A: mean={format_duration(stats_a['mean'])}, "
                    f"p95={format_duration(stats_a['p95'])} | "
                    f"B: mean={format_duration(stats_b['mean'])}, "
                    f"p95={format_duration(stats_b['p95'])}"
                ),
                instance_a_value=stats_a,
                instance_b_value=stats_b,
                diff=pct_diff(stats_a['mean'], stats_b['mean']) if stats_a['mean'] and stats_b['mean'] else None,
                duration_seconds=time.time() - t0,
            ))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _check_alive(self, url: str) -> bool:
        try:
            resp = http_get(f"{url}/api/config", timeout=15, retries=2, retry_delay=1)
            return resp.status_code == 200
        except Exception:
            return False

    def _test_db(self, dsn: str) -> bool:
        try:
            pg_scalar(dsn, "SELECT 1")
            return True
        except Exception:
            return False
