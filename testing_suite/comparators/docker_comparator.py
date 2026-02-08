"""
Docker Comparison Module for AudioMuse-AI Testing Suite.

Compares two Docker deployments across:
  - Container health and status
  - Resource usage (memory, CPU)
  - Log analysis (error rates, warning patterns)
  - Service connectivity (Redis, PostgreSQL, Flask, Worker)
  - Container uptime and restart counts
  - Log-based error pattern detection
"""

import json
import logging
import re
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from testing_suite.config import ComparisonConfig, InstanceConfig
from testing_suite.utils import (
    ComparisonReport, TestResult, TestStatus,
    docker_exec, docker_logs, docker_inspect, pct_diff
)

logger = logging.getLogger(__name__)

# Log patterns to search for
ERROR_PATTERNS = [
    (r"(?i)traceback \(most recent call last\)", "Python Traceback"),
    (r"(?i)error|exception", "Error/Exception"),
    (r"(?i)out of memory|oom|killed", "OOM/Memory Kill"),
    (r"(?i)connection refused|connection reset|broken pipe", "Connection Error"),
    (r"(?i)timeout|timed out", "Timeout"),
    (r"(?i)permission denied|access denied", "Permission Error"),
    (r"(?i)disk full|no space left", "Disk Space"),
    (r"(?i)segmentation fault|segfault|core dump", "Crash/Segfault"),
    (r"(?i)worker .* died|worker .* killed", "Worker Death"),
    (r"(?i)database .* error|psycopg2\..*error", "Database Error"),
    (r"(?i)redis\..*error|redis connection", "Redis Error"),
]

WARNING_PATTERNS = [
    (r"(?i)deprecat", "Deprecation Warning"),
    (r"(?i)warning", "Warning"),
    (r"(?i)retry|retrying", "Retry Attempt"),
    (r"(?i)slow query|slow request", "Slow Operation"),
    (r"(?i)memory usage|memory pressure", "Memory Pressure"),
]


def _get_ssh_params(instance: InstanceConfig) -> dict:
    """Extract SSH parameters from instance config."""
    return {
        "ssh_host": instance.ssh_host,
        "ssh_user": instance.ssh_user,
        "ssh_key": instance.ssh_key,
        "ssh_port": instance.ssh_port,
    }


class DockerComparator:
    """Compares Docker deployment health across two AudioMuse-AI instances."""

    def __init__(self, config: ComparisonConfig):
        self.config = config
        self.inst_a = config.instance_a
        self.inst_b = config.instance_b

    def run_all(self, report: ComparisonReport):
        """Run all Docker comparison tests."""
        logger.info("Starting Docker comparison tests...")

        # Test each container type on both instances
        containers = [
            ("flask", "docker_flask_container", "Flask App Server"),
            ("worker", "docker_worker_container", "RQ Worker"),
            ("postgres", "docker_postgres_container", "PostgreSQL"),
            ("redis", "docker_redis_container", "Redis"),
        ]

        for container_key, attr_name, description in containers:
            name_a = getattr(self.inst_a, attr_name)
            name_b = getattr(self.inst_b, attr_name)

            self._test_container_health(report, name_a, name_b, description,
                                        self.inst_a, self.inst_b)
            self._test_container_resource_usage(report, name_a, name_b, description,
                                                 self.inst_a, self.inst_b)

        # Log analysis for flask and worker containers
        for attr_name, description in [
            ("docker_flask_container", "Flask"),
            ("docker_worker_container", "Worker"),
        ]:
            name_a = getattr(self.inst_a, attr_name)
            name_b = getattr(self.inst_b, attr_name)
            self._test_log_error_analysis(report, name_a, name_b, description,
                                           self.inst_a, self.inst_b)

        # Service connectivity tests
        self._test_redis_connectivity(report)
        self._test_postgres_connectivity(report)

        logger.info("Docker comparison tests complete.")

    # ------------------------------------------------------------------
    # Container health
    # ------------------------------------------------------------------

    def _test_container_health(self, report: ComparisonReport,
                                name_a: str, name_b: str, description: str,
                                inst_a: InstanceConfig, inst_b: InstanceConfig):
        """Check container status, uptime, and restart count."""
        t0 = time.time()

        info_a = docker_inspect(name_a, **_get_ssh_params(inst_a))
        info_b = docker_inspect(name_b, **_get_ssh_params(inst_b))

        # Container running status
        running_a = self._is_running(info_a)
        running_b = self._is_running(info_b)

        if running_a is None and running_b is None:
            report.add_result(TestResult(
                category="docker",
                name=f"{description}: container status",
                status=TestStatus.SKIP,
                message="Cannot inspect containers (Docker not available or containers not found)",
                duration_seconds=time.time() - t0,
            ))
            return

        if running_a and running_b:
            status = TestStatus.PASS
            msg = "Both containers running"
        elif running_a or running_b:
            status = TestStatus.FAIL
            msg = f"Only {'A' if running_a else 'B'} is running"
        else:
            status = TestStatus.FAIL
            msg = "Neither container is running"

        report.add_result(TestResult(
            category="docker",
            name=f"{description}: container status",
            status=status,
            message=msg,
            instance_a_value=f"running={running_a}",
            instance_b_value=f"running={running_b}",
            duration_seconds=time.time() - t0,
        ))

        # Restart count
        restarts_a = self._get_restart_count(info_a)
        restarts_b = self._get_restart_count(info_b)

        if restarts_a is not None or restarts_b is not None:
            if (restarts_a or 0) == 0 and (restarts_b or 0) == 0:
                r_status = TestStatus.PASS
                r_msg = "No restarts on either instance"
            elif (restarts_a or 0) > 5 or (restarts_b or 0) > 5:
                r_status = TestStatus.FAIL
                r_msg = f"High restart count: A={restarts_a}, B={restarts_b}"
            else:
                r_status = TestStatus.WARN
                r_msg = f"Restarts: A={restarts_a}, B={restarts_b}"

            report.add_result(TestResult(
                category="docker",
                name=f"{description}: restart count",
                status=r_status,
                message=r_msg,
                instance_a_value=restarts_a,
                instance_b_value=restarts_b,
                duration_seconds=time.time() - t0,
            ))

        # Health check status
        health_a = self._get_health_status(info_a)
        health_b = self._get_health_status(info_b)

        if health_a or health_b:
            if health_a == health_b:
                h_status = TestStatus.PASS
                h_msg = f"Same health status: {health_a}"
            else:
                h_status = TestStatus.WARN
                h_msg = f"Health differs: A={health_a}, B={health_b}"

            report.add_result(TestResult(
                category="docker",
                name=f"{description}: health check",
                status=h_status,
                message=h_msg,
                instance_a_value=health_a,
                instance_b_value=health_b,
                duration_seconds=time.time() - t0,
            ))

    # ------------------------------------------------------------------
    # Resource usage
    # ------------------------------------------------------------------

    def _test_container_resource_usage(self, report: ComparisonReport,
                                        name_a: str, name_b: str, description: str,
                                        inst_a: InstanceConfig, inst_b: InstanceConfig):
        """Compare memory and CPU usage between containers."""
        t0 = time.time()

        stats_a = self._get_container_stats(name_a, inst_a)
        stats_b = self._get_container_stats(name_b, inst_b)

        if not stats_a and not stats_b:
            report.add_result(TestResult(
                category="docker",
                name=f"{description}: resource usage",
                status=TestStatus.SKIP,
                message="Cannot get container stats",
                duration_seconds=time.time() - t0,
            ))
            return

        # Memory usage
        mem_a = stats_a.get('memory_mb') if stats_a else None
        mem_b = stats_b.get('memory_mb') if stats_b else None

        if mem_a is not None and mem_b is not None:
            diff = pct_diff(mem_a, mem_b)
            if diff <= 20:
                status = TestStatus.PASS
            elif diff <= 50:
                status = TestStatus.WARN
            else:
                status = TestStatus.FAIL

            report.add_result(TestResult(
                category="docker",
                name=f"{description}: memory usage",
                status=status,
                message=f"A={mem_a:.1f}MB, B={mem_b:.1f}MB (diff {diff:.1f}%)",
                instance_a_value=mem_a,
                instance_b_value=mem_b,
                diff=diff,
                duration_seconds=time.time() - t0,
            ))

        # CPU usage
        cpu_a = stats_a.get('cpu_pct') if stats_a else None
        cpu_b = stats_b.get('cpu_pct') if stats_b else None

        if cpu_a is not None and cpu_b is not None:
            report.add_result(TestResult(
                category="docker",
                name=f"{description}: CPU usage",
                status=TestStatus.PASS,
                message=f"A={cpu_a:.1f}%, B={cpu_b:.1f}%",
                instance_a_value=cpu_a,
                instance_b_value=cpu_b,
                duration_seconds=time.time() - t0,
            ))

    # ------------------------------------------------------------------
    # Log error analysis
    # ------------------------------------------------------------------

    def _test_log_error_analysis(self, report: ComparisonReport,
                                  name_a: str, name_b: str, description: str,
                                  inst_a: InstanceConfig, inst_b: InstanceConfig):
        """Analyze container logs for error patterns."""
        t0 = time.time()

        # Fetch logs
        logs_a_stdout, logs_a_stderr, rc_a = docker_logs(
            name_a, tail=2000, **_get_ssh_params(inst_a), timeout=30)
        logs_b_stdout, logs_b_stderr, rc_b = docker_logs(
            name_b, tail=2000, **_get_ssh_params(inst_b), timeout=30)

        if rc_a != 0 and rc_b != 0:
            report.add_result(TestResult(
                category="docker",
                name=f"{description}: log analysis",
                status=TestStatus.SKIP,
                message="Cannot fetch logs from either container",
                duration_seconds=time.time() - t0,
            ))
            return

        # Combine stdout + stderr for analysis
        logs_a = (logs_a_stdout or '') + '\n' + (logs_a_stderr or '')
        logs_b = (logs_b_stdout or '') + '\n' + (logs_b_stderr or '')

        # Error pattern matching
        errors_a = self._count_patterns(logs_a, ERROR_PATTERNS)
        errors_b = self._count_patterns(logs_b, ERROR_PATTERNS)

        total_errors_a = sum(errors_a.values())
        total_errors_b = sum(errors_b.values())

        if total_errors_a == 0 and total_errors_b == 0:
            report.add_result(TestResult(
                category="docker",
                name=f"{description}: error patterns",
                status=TestStatus.PASS,
                message="No error patterns detected in recent logs",
                duration_seconds=time.time() - t0,
            ))
        else:
            # Compare error counts
            if total_errors_a <= total_errors_b:
                status = TestStatus.WARN
            else:
                status = TestStatus.WARN

            if total_errors_a > 50 or total_errors_b > 50:
                status = TestStatus.FAIL

            report.add_result(TestResult(
                category="docker",
                name=f"{description}: error count",
                status=status,
                message=f"Errors in last 2000 log lines: A={total_errors_a}, B={total_errors_b}",
                instance_a_value=total_errors_a,
                instance_b_value=total_errors_b,
                duration_seconds=time.time() - t0,
                details={"errors_a": dict(errors_a), "errors_b": dict(errors_b)},
            ))

            # Detailed per-pattern breakdown
            all_patterns = set(errors_a.keys()) | set(errors_b.keys())
            for pattern_name in sorted(all_patterns):
                cnt_a = errors_a.get(pattern_name, 0)
                cnt_b = errors_b.get(pattern_name, 0)

                if cnt_a == 0 and cnt_b == 0:
                    continue

                if cnt_a == cnt_b:
                    p_status = TestStatus.WARN
                elif cnt_a > cnt_b * 2 or cnt_b > cnt_a * 2:
                    p_status = TestStatus.FAIL
                else:
                    p_status = TestStatus.WARN

                report.add_result(TestResult(
                    category="docker",
                    name=f"{description}: {pattern_name}",
                    status=p_status,
                    message=f"A={cnt_a}, B={cnt_b}",
                    instance_a_value=cnt_a,
                    instance_b_value=cnt_b,
                    duration_seconds=time.time() - t0,
                ))

        # Warning pattern matching
        warnings_a = self._count_patterns(logs_a, WARNING_PATTERNS)
        warnings_b = self._count_patterns(logs_b, WARNING_PATTERNS)

        total_warnings_a = sum(warnings_a.values())
        total_warnings_b = sum(warnings_b.values())

        report.add_result(TestResult(
            category="docker",
            name=f"{description}: warning count",
            status=TestStatus.PASS if total_warnings_a < 100 and total_warnings_b < 100 else TestStatus.WARN,
            message=f"Warnings: A={total_warnings_a}, B={total_warnings_b}",
            instance_a_value=total_warnings_a,
            instance_b_value=total_warnings_b,
            duration_seconds=time.time() - t0,
            details={"warnings_a": dict(warnings_a), "warnings_b": dict(warnings_b)},
        ))

        # Check for Python tracebacks specifically (important indicator)
        tb_count_a = logs_a.count("Traceback (most recent call last)")
        tb_count_b = logs_b.count("Traceback (most recent call last)")

        if tb_count_a == 0 and tb_count_b == 0:
            tb_status = TestStatus.PASS
            tb_msg = "No Python tracebacks in recent logs"
        elif tb_count_a > 10 or tb_count_b > 10:
            tb_status = TestStatus.FAIL
            tb_msg = f"Tracebacks: A={tb_count_a}, B={tb_count_b}"
        else:
            tb_status = TestStatus.WARN
            tb_msg = f"Tracebacks: A={tb_count_a}, B={tb_count_b}"

        report.add_result(TestResult(
            category="docker",
            name=f"{description}: Python tracebacks",
            status=tb_status,
            message=tb_msg,
            instance_a_value=tb_count_a,
            instance_b_value=tb_count_b,
            duration_seconds=time.time() - t0,
        ))

    # ------------------------------------------------------------------
    # Service connectivity
    # ------------------------------------------------------------------

    def _test_redis_connectivity(self, report: ComparisonReport):
        """Test Redis connectivity from within the Flask container."""
        t0 = time.time()
        for label, inst in [("A", self.inst_a), ("B", self.inst_b)]:
            container = inst.docker_flask_container
            stdout, stderr, rc = docker_exec(
                container, "python -c \"from redis import Redis; r=Redis.from_url('" +
                inst.redis_url + "'); print(r.ping())\"",
                **_get_ssh_params(inst), timeout=15,
            )
            if rc == 0 and "True" in stdout:
                status = TestStatus.PASS
                msg = "Redis ping successful"
            elif rc == -2:
                status = TestStatus.SKIP
                msg = "Docker not available"
            else:
                status = TestStatus.WARN
                msg = f"Redis ping failed: {stderr[:200] if stderr else stdout[:200]}"

            report.add_result(TestResult(
                category="docker",
                name=f"Redis connectivity ({label})",
                status=status,
                message=msg,
                duration_seconds=time.time() - t0,
            ))

    def _test_postgres_connectivity(self, report: ComparisonReport):
        """Test PostgreSQL connectivity from within the Flask container."""
        t0 = time.time()
        for label, inst in [("A", self.inst_a), ("B", self.inst_b)]:
            container = inst.docker_flask_container
            stdout, stderr, rc = docker_exec(
                container,
                f"python -c \"import psycopg2; c=psycopg2.connect('{inst.pg_dsn}', connect_timeout=5); "
                f"cur=c.cursor(); cur.execute('SELECT 1'); print(cur.fetchone()[0]); c.close()\"",
                **_get_ssh_params(inst), timeout=15,
            )
            if rc == 0 and "1" in stdout:
                status = TestStatus.PASS
                msg = "PostgreSQL SELECT 1 successful"
            elif rc == -2:
                status = TestStatus.SKIP
                msg = "Docker not available"
            else:
                status = TestStatus.WARN
                msg = f"PostgreSQL test failed: {stderr[:200] if stderr else stdout[:200]}"

            report.add_result(TestResult(
                category="docker",
                name=f"PostgreSQL connectivity ({label})",
                status=status,
                message=msg,
                duration_seconds=time.time() - t0,
            ))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _is_running(self, info: Optional[dict]) -> Optional[bool]:
        """Check if container is running from inspect data."""
        if not info:
            return None
        state = info.get("State", {})
        return state.get("Running", False)

    def _get_restart_count(self, info: Optional[dict]) -> Optional[int]:
        """Get container restart count from inspect data."""
        if not info:
            return None
        return info.get("RestartCount", 0)

    def _get_health_status(self, info: Optional[dict]) -> Optional[str]:
        """Get container health status from inspect data."""
        if not info:
            return None
        state = info.get("State", {})
        health = state.get("Health", {})
        return health.get("Status") if health else None

    def _get_container_stats(self, name: str, inst: InstanceConfig) -> Optional[dict]:
        """Get container resource stats via docker stats --no-stream."""
        ssh_params = _get_ssh_params(inst)
        cmd_parts = ["docker", "stats", "--no-stream", "--format",
                      "{{.MemUsage}}|||{{.CPUPerc}}", name]

        if ssh_params.get("ssh_host"):
            import subprocess
            ssh_cmd = ["ssh", "-o", "StrictHostKeyChecking=no",
                       "-p", str(ssh_params.get("ssh_port", 22))]
            if ssh_params.get("ssh_key"):
                ssh_cmd += ["-i", ssh_params["ssh_key"]]
            host = f"{ssh_params['ssh_user']}@{ssh_params['ssh_host']}" \
                if ssh_params.get("ssh_user") else ssh_params["ssh_host"]
            ssh_cmd.append(host)
            ssh_cmd.append(" ".join(cmd_parts))
            full_cmd = ssh_cmd
        else:
            full_cmd = cmd_parts

        try:
            import subprocess
            proc = subprocess.run(full_cmd, capture_output=True, text=True, timeout=15)
            if proc.returncode == 0 and proc.stdout.strip():
                parts = proc.stdout.strip().split("|||")
                if len(parts) == 2:
                    mem_str = parts[0].strip()
                    cpu_str = parts[1].strip().rstrip('%')

                    # Parse memory (e.g., "512MiB / 16GiB")
                    mem_match = re.search(r'([\d.]+)(Ki|Mi|Gi|B)', mem_str)
                    memory_mb = 0.0
                    if mem_match:
                        val = float(mem_match.group(1))
                        unit = mem_match.group(2)
                        if unit == "Gi":
                            memory_mb = val * 1024
                        elif unit == "Mi":
                            memory_mb = val
                        elif unit == "Ki":
                            memory_mb = val / 1024
                        else:
                            memory_mb = val / (1024 * 1024)

                    cpu_pct = float(cpu_str) if cpu_str else 0.0

                    return {"memory_mb": memory_mb, "cpu_pct": cpu_pct}
        except Exception as e:
            logger.debug(f"Stats fetch failed for {name}: {e}")
        return None

    def _count_patterns(self, log_text: str, patterns: list) -> Counter:
        """Count occurrences of each pattern in log text."""
        counts = Counter()
        for pattern, name in patterns:
            matches = re.findall(pattern, log_text)
            if matches:
                counts[name] = len(matches)
        return counts
