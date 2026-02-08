"""
Shared utilities for the AudioMuse-AI Testing & Comparison Suite.

Provides HTTP helpers, database connectors, Docker log fetchers,
timing utilities, and result aggregation primitives.
"""

import json
import logging
import subprocess
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

class TestStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    WARN = "WARN"
    SKIP = "SKIP"
    ERROR = "ERROR"


@dataclass
class TestResult:
    """A single test result entry."""
    category: str       # e.g. "api", "database", "docker", "performance"
    name: str           # descriptive test name
    status: TestStatus
    message: str = ""
    instance_a_value: Any = None
    instance_b_value: Any = None
    diff: Any = None    # computed difference
    duration_seconds: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d['status'] = self.status.value
        # Ensure JSON-serializable
        for k in ('instance_a_value', 'instance_b_value', 'diff', 'details'):
            try:
                json.dumps(d[k])
            except (TypeError, ValueError):
                d[k] = str(d[k])
        return d


@dataclass
class CategorySummary:
    """Summary for a test category."""
    category: str
    total: int = 0
    passed: int = 0
    failed: int = 0
    warned: int = 0
    skipped: int = 0
    errors: int = 0
    results: List[TestResult] = field(default_factory=list)

    def add(self, result: TestResult):
        self.results.append(result)
        self.total += 1
        if result.status == TestStatus.PASS:
            self.passed += 1
        elif result.status == TestStatus.FAIL:
            self.failed += 1
        elif result.status == TestStatus.WARN:
            self.warned += 1
        elif result.status == TestStatus.SKIP:
            self.skipped += 1
        elif result.status == TestStatus.ERROR:
            self.errors += 1


@dataclass
class ComparisonReport:
    """Full comparison report across all categories."""
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    instance_a_name: str = ""
    instance_b_name: str = ""
    instance_a_branch: str = ""
    instance_b_branch: str = ""
    categories: Dict[str, CategorySummary] = field(default_factory=dict)
    config_snapshot: Dict[str, Any] = field(default_factory=dict)

    def add_result(self, result: TestResult):
        cat = result.category
        if cat not in self.categories:
            self.categories[cat] = CategorySummary(category=cat)
        self.categories[cat].add(result)

    @property
    def total_tests(self) -> int:
        return sum(c.total for c in self.categories.values())

    @property
    def total_passed(self) -> int:
        return sum(c.passed for c in self.categories.values())

    @property
    def total_failed(self) -> int:
        return sum(c.failed for c in self.categories.values())

    @property
    def total_errors(self) -> int:
        return sum(c.errors for c in self.categories.values())

    @property
    def overall_status(self) -> TestStatus:
        if self.total_failed > 0 or self.total_errors > 0:
            return TestStatus.FAIL
        return TestStatus.PASS

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "instance_a": {"name": self.instance_a_name, "branch": self.instance_a_branch},
            "instance_b": {"name": self.instance_b_name, "branch": self.instance_b_branch},
            "overall_status": self.overall_status.value,
            "summary": {
                "total": self.total_tests,
                "passed": self.total_passed,
                "failed": self.total_failed,
                "errors": self.total_errors,
            },
            "categories": {
                name: {
                    "total": cat.total,
                    "passed": cat.passed,
                    "failed": cat.failed,
                    "warned": cat.warned,
                    "skipped": cat.skipped,
                    "errors": cat.errors,
                    "results": [r.to_dict() for r in cat.results],
                }
                for name, cat in self.categories.items()
            },
            "config": self.config_snapshot,
        }


# ---------------------------------------------------------------------------
# HTTP Helpers
# ---------------------------------------------------------------------------

def http_get(url: str, params: dict = None, timeout: int = 120,
             retries: int = 3, retry_delay: float = 2.0) -> requests.Response:
    """HTTP GET with retries on connection errors."""
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            return resp
        except requests.RequestException as e:
            last_exc = e
            if attempt < retries:
                time.sleep(retry_delay)
                logger.debug(f"Retry {attempt}/{retries} for GET {url}: {e}")
    raise last_exc


def http_post(url: str, json_data: dict = None, timeout: int = 120,
              retries: int = 3, retry_delay: float = 2.0) -> requests.Response:
    """HTTP POST with retries on connection errors."""
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(url, json=json_data, timeout=timeout)
            return resp
        except requests.RequestException as e:
            last_exc = e
            if attempt < retries:
                time.sleep(retry_delay)
                logger.debug(f"Retry {attempt}/{retries} for POST {url}: {e}")
    raise last_exc


def timed_request(method: str, url: str, **kwargs) -> Tuple[requests.Response, float]:
    """Execute an HTTP request and return (response, elapsed_seconds)."""
    start = time.perf_counter()
    if method.upper() == "GET":
        resp = http_get(url, **kwargs)
    else:
        resp = http_post(url, **kwargs)
    elapsed = time.perf_counter() - start
    return resp, elapsed


def wait_for_task_success(base_url: str, task_id: str, timeout: int = 1200,
                          retries: int = 3, retry_delay: float = 2.0) -> dict:
    """Poll active_tasks until task completes, then verify success via last_task."""
    start = time.time()
    while time.time() - start < timeout:
        act_resp = http_get(f'{base_url}/api/active_tasks', retries=retries,
                            retry_delay=retry_delay)
        act_resp.raise_for_status()
        active = act_resp.json()

        if active and active.get('task_id') == task_id:
            time.sleep(2)
            continue

        last_resp = http_get(f'{base_url}/api/last_task', retries=retries,
                             retry_delay=retry_delay)
        last_resp.raise_for_status()
        final = last_resp.json()
        final_id = final.get('task_id')
        final_state = (final.get('status') or final.get('state') or '').upper()

        if final_id == task_id:
            return final
        # Task might have been superseded; keep polling briefly
        time.sleep(2)

    return {"status": "TIMEOUT", "task_id": task_id}


# ---------------------------------------------------------------------------
# Database Helpers
# ---------------------------------------------------------------------------

def get_pg_connection(dsn: str):
    """Create a psycopg2 connection from DSN."""
    import psycopg2
    return psycopg2.connect(dsn, connect_timeout=30,
                            options='-c statement_timeout=120000')


def pg_query(dsn: str, sql: str, params: tuple = None) -> List[tuple]:
    """Execute a read-only query and return all rows."""
    import psycopg2
    conn = psycopg2.connect(dsn, connect_timeout=30,
                            options='-c statement_timeout=120000')
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall()
    finally:
        conn.close()


def pg_query_dict(dsn: str, sql: str, params: tuple = None) -> List[dict]:
    """Execute a query and return rows as dicts."""
    import psycopg2
    from psycopg2.extras import RealDictCursor
    conn = psycopg2.connect(dsn, connect_timeout=30,
                            options='-c statement_timeout=120000')
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, params)
            return [dict(row) for row in cur.fetchall()]
    finally:
        conn.close()


def pg_scalar(dsn: str, sql: str, params: tuple = None):
    """Execute a query that returns a single scalar value."""
    rows = pg_query(dsn, sql, params)
    if rows and rows[0]:
        return rows[0][0]
    return None


# ---------------------------------------------------------------------------
# Docker Helpers
# ---------------------------------------------------------------------------

def docker_exec(container: str, command: str, ssh_host: str = "",
                ssh_user: str = "", ssh_key: str = "",
                ssh_port: int = 22, timeout: int = 30) -> Tuple[str, str, int]:
    """
    Run a command inside a Docker container (locally or via SSH).
    Returns (stdout, stderr, returncode).
    """
    if ssh_host:
        ssh_cmd = ["ssh", "-o", "StrictHostKeyChecking=no",
                   "-p", str(ssh_port)]
        if ssh_key:
            ssh_cmd += ["-i", ssh_key]
        ssh_cmd.append(f"{ssh_user}@{ssh_host}" if ssh_user else ssh_host)
        ssh_cmd.append(f"docker exec {container} {command}")
        full_cmd = ssh_cmd
    else:
        full_cmd = ["docker", "exec", container] + command.split()

    try:
        proc = subprocess.run(full_cmd, capture_output=True, text=True, timeout=timeout)
        return proc.stdout, proc.stderr, proc.returncode
    except subprocess.TimeoutExpired:
        return "", "Command timed out", -1
    except FileNotFoundError:
        return "", "docker or ssh command not found", -2


def docker_logs(container: str, tail: int = 500, since: str = "",
                ssh_host: str = "", ssh_user: str = "", ssh_key: str = "",
                ssh_port: int = 22, timeout: int = 30) -> Tuple[str, str, int]:
    """
    Fetch Docker container logs (locally or via SSH).
    Returns (stdout, stderr, returncode).
    """
    cmd_parts = ["docker", "logs", f"--tail={tail}"]
    if since:
        cmd_parts += [f"--since={since}"]
    cmd_parts.append(container)

    if ssh_host:
        ssh_cmd = ["ssh", "-o", "StrictHostKeyChecking=no",
                   "-p", str(ssh_port)]
        if ssh_key:
            ssh_cmd += ["-i", ssh_key]
        ssh_cmd.append(f"{ssh_user}@{ssh_host}" if ssh_user else ssh_host)
        ssh_cmd.append(" ".join(cmd_parts))
        full_cmd = ssh_cmd
    else:
        full_cmd = cmd_parts

    try:
        proc = subprocess.run(full_cmd, capture_output=True, text=True, timeout=timeout)
        return proc.stdout, proc.stderr, proc.returncode
    except subprocess.TimeoutExpired:
        return "", "Logs fetch timed out", -1
    except FileNotFoundError:
        return "", "docker or ssh command not found", -2


def docker_inspect(container: str, ssh_host: str = "", ssh_user: str = "",
                   ssh_key: str = "", ssh_port: int = 22,
                   timeout: int = 15) -> Optional[dict]:
    """
    Run docker inspect on a container and return the parsed JSON.
    Returns None on failure.
    """
    cmd_parts = ["docker", "inspect", container]

    if ssh_host:
        ssh_cmd = ["ssh", "-o", "StrictHostKeyChecking=no",
                   "-p", str(ssh_port)]
        if ssh_key:
            ssh_cmd += ["-i", ssh_key]
        ssh_cmd.append(f"{ssh_user}@{ssh_host}" if ssh_user else ssh_host)
        ssh_cmd.append(" ".join(cmd_parts))
        full_cmd = ssh_cmd
    else:
        full_cmd = cmd_parts

    try:
        proc = subprocess.run(full_cmd, capture_output=True, text=True, timeout=timeout)
        if proc.returncode == 0:
            data = json.loads(proc.stdout)
            return data[0] if isinstance(data, list) and data else data
    except Exception as e:
        logger.debug(f"docker inspect failed for {container}: {e}")
    return None


# ---------------------------------------------------------------------------
# Comparison Helpers
# ---------------------------------------------------------------------------

def compare_values(a, b, tolerance_pct: float = 0.0) -> Tuple[bool, str]:
    """
    Compare two values. For numeric types, allow a percentage tolerance.
    Returns (is_equal, description).
    """
    if a is None and b is None:
        return True, "Both None"
    if a is None or b is None:
        return False, f"One is None: A={a}, B={b}"

    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if a == 0 and b == 0:
            return True, "Both zero"
        if a == 0 or b == 0:
            return False, f"A={a}, B={b}"
        pct_diff = abs(a - b) / max(abs(a), abs(b)) * 100
        if pct_diff <= tolerance_pct:
            return True, f"Within tolerance ({pct_diff:.2f}% <= {tolerance_pct}%)"
        return False, f"Difference {pct_diff:.2f}% exceeds tolerance {tolerance_pct}%"

    if isinstance(a, str) and isinstance(b, str):
        if a == b:
            return True, "Strings match"
        return False, f"Strings differ: '{a[:100]}' vs '{b[:100]}'"

    if isinstance(a, dict) and isinstance(b, dict):
        keys_a = set(a.keys())
        keys_b = set(b.keys())
        if keys_a != keys_b:
            missing_in_b = keys_a - keys_b
            missing_in_a = keys_b - keys_a
            return False, f"Key mismatch: missing_in_B={missing_in_b}, missing_in_A={missing_in_a}"
        return True, "Dict keys match"

    if isinstance(a, list) and isinstance(b, list):
        if len(a) == len(b):
            return True, f"Lists same length ({len(a)})"
        return False, f"List length differs: {len(a)} vs {len(b)}"

    # Fallback
    if a == b:
        return True, "Values equal"
    return False, f"Values differ: {a} vs {b}"


def pct_diff(a: float, b: float) -> float:
    """Calculate percentage difference between two values."""
    if a == 0 and b == 0:
        return 0.0
    if a == 0 or b == 0:
        return 100.0
    return abs(a - b) / max(abs(a), abs(b)) * 100


def format_duration(seconds: float) -> str:
    """Format seconds into a human-readable string."""
    if seconds < 1:
        return f"{seconds*1000:.1f}ms"
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes}m {secs:.1f}s"
