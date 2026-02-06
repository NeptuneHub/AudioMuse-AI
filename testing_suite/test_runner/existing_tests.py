"""
Existing Test Integration Module for AudioMuse-AI Testing Suite.

Discovers and runs existing unit and integration tests from the codebase
against both instances, collecting and comparing results.

Integrates:
  - tests/unit/ (17 unit test modules via pytest)
  - test/test.py (E2E API integration tests)
  - test/test_analysis_integration.py (ONNX model integration)
  - test/test_clap_analysis_integration.py (CLAP model integration)
"""

import json
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from testing_suite.config import ComparisonConfig, InstanceConfig
from testing_suite.utils import (
    ComparisonReport, TestResult, TestStatus, pct_diff
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Known unit test files
UNIT_TEST_DIR = PROJECT_ROOT / "tests" / "unit"
UNIT_TEST_FILES = [
    "test_ai.py",
    "test_analysis.py",
    "test_app_analysis.py",
    "test_artist_gmm_manager.py",
    "test_clap_text_search.py",
    "test_clustering.py",
    "test_clustering_helper.py",
    "test_clustering_postprocessing.py",
    "test_commons.py",
    "test_mediaserver.py",
    "test_memory_cleanup.py",
    "test_memory_utils.py",
    "test_path_manager.py",
    "test_song_alchemy.py",
    "test_sonic_fingerprint_manager.py",
    "test_string_sanitization.py",
    "test_voyager_manager.py",
]

# Integration test files
INTEGRATION_TEST_DIR = PROJECT_ROOT / "test"
INTEGRATION_TEST_FILES = [
    "test_analysis_integration.py",
    "test_clap_analysis_integration.py",
]

# E2E API test (requires a running instance)
E2E_TEST_FILE = PROJECT_ROOT / "test" / "test.py"

# Individual E2E test names (from test/test.py)
E2E_TEST_NAMES = [
    "test_analysis_smoke_flow",
    "test_instant_playlist_functionality",
    "test_sonic_fingerprint_and_playlist",
    "test_song_alchemy_and_playlist",
    "test_map_visualization",
    "test_annoy_similarity_and_playlist",
    "test_song_path_and_playlist",
    "test_clustering_smoke_flow",
]


def _parse_pytest_json(json_path: str) -> dict:
    """Parse pytest JSON report."""
    try:
        with open(json_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Could not parse pytest JSON report: {e}")
        return {}


def _run_pytest(test_path: str, extra_args: list = None,
                env_override: dict = None, timeout: int = 600,
                json_report: bool = True) -> Tuple[dict, str, int]:
    """
    Run pytest and capture results.
    Returns (parsed_json_result, stdout, returncode).
    """
    cmd = ["python", "-m", "pytest", "-v", "--tb=short"]

    json_path = None
    if json_report:
        json_path = f"/tmp/pytest_report_{int(time.time() * 1000)}.json"
        cmd += [f"--json-report", f"--json-report-file={json_path}"]

    cmd.append(str(test_path))

    if extra_args:
        cmd.extend(extra_args)

    env = os.environ.copy()
    if env_override:
        env.update(env_override)

    # Ensure project root is in PYTHONPATH
    python_path = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{PROJECT_ROOT}:{python_path}" if python_path else str(PROJECT_ROOT)

    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
            cwd=str(PROJECT_ROOT), env=env,
        )
        stdout = proc.stdout + proc.stderr
        returncode = proc.returncode

        # Parse JSON report if available
        result = {}
        if json_path and os.path.exists(json_path):
            result = _parse_pytest_json(json_path)
            os.unlink(json_path)

        return result, stdout, returncode

    except subprocess.TimeoutExpired:
        return {}, f"pytest timed out after {timeout}s", -1
    except Exception as e:
        return {}, str(e), -2


def _parse_stdout_results(stdout: str) -> dict:
    """
    Parse pytest stdout for test results when JSON report is not available.
    Returns dict with passed, failed, error, skipped counts and test names.
    """
    import re

    results = {
        "passed": 0,
        "failed": 0,
        "errors": 0,
        "skipped": 0,
        "tests": [],
    }

    # Parse individual test results: PASSED, FAILED, ERROR, SKIPPED
    for line in stdout.split('\n'):
        line = line.strip()
        if " PASSED" in line:
            results["passed"] += 1
            results["tests"].append({"name": line.split(" PASSED")[0].strip(), "status": "passed"})
        elif " FAILED" in line:
            results["failed"] += 1
            results["tests"].append({"name": line.split(" FAILED")[0].strip(), "status": "failed"})
        elif " ERROR" in line:
            results["errors"] += 1
            results["tests"].append({"name": line.split(" ERROR")[0].strip(), "status": "error"})
        elif " SKIPPED" in line:
            results["skipped"] += 1
            results["tests"].append({"name": line.split(" SKIPPED")[0].strip(), "status": "skipped"})

    # Try to parse summary line like "5 passed, 1 failed, 2 skipped"
    summary_match = re.search(
        r'(\d+)\s+passed.*?(?:(\d+)\s+failed)?.*?(?:(\d+)\s+skipped)?.*?(?:(\d+)\s+error)?',
        stdout
    )
    if summary_match:
        if summary_match.group(1):
            results["passed"] = max(results["passed"], int(summary_match.group(1)))
        if summary_match.group(2):
            results["failed"] = max(results["failed"], int(summary_match.group(2)))
        if summary_match.group(3):
            results["skipped"] = max(results["skipped"], int(summary_match.group(3)))
        if summary_match.group(4):
            results["errors"] = max(results["errors"], int(summary_match.group(4)))

    return results


class ExistingTestRunner:
    """Runs existing tests and integrates results into the comparison report."""

    def __init__(self, config: ComparisonConfig):
        self.config = config

    def run_all(self, report: ComparisonReport):
        """Run all existing test suites."""
        logger.info("Starting existing test integration...")

        if self.config.run_existing_unit_tests:
            self._run_unit_tests(report)

        if self.config.run_existing_integration_tests:
            self._run_integration_tests(report)
            self._run_e2e_tests(report)

        logger.info("Existing test integration complete.")

    # ------------------------------------------------------------------
    # Unit tests
    # ------------------------------------------------------------------

    def _run_unit_tests(self, report: ComparisonReport):
        """Run unit tests from tests/unit/ directory."""
        t0 = time.time()

        if not UNIT_TEST_DIR.exists():
            report.add_result(TestResult(
                category="existing_tests",
                name="Unit Tests: directory check",
                status=TestStatus.ERROR,
                message=f"Unit test directory not found: {UNIT_TEST_DIR}",
                duration_seconds=time.time() - t0,
            ))
            return

        # Run entire unit test suite
        logger.info("Running unit test suite...")
        result, stdout, rc = _run_pytest(
            str(UNIT_TEST_DIR),
            extra_args=["-x", "--timeout=120"],
            timeout=600,
            json_report=True,
        )

        # Parse results
        if result and "summary" in result:
            summary = result["summary"]
            passed = summary.get("passed", 0)
            failed = summary.get("failed", 0)
            errors = summary.get("error", 0)
            skipped = summary.get("skipped", 0)
            total = summary.get("total", passed + failed + errors + skipped)
        else:
            # Fallback to stdout parsing
            parsed = _parse_stdout_results(stdout)
            passed = parsed["passed"]
            failed = parsed["failed"]
            errors = parsed["errors"]
            skipped = parsed["skipped"]
            total = passed + failed + errors + skipped

        if failed == 0 and errors == 0:
            status = TestStatus.PASS
        elif errors > 0:
            status = TestStatus.ERROR
        else:
            status = TestStatus.FAIL

        report.add_result(TestResult(
            category="existing_tests",
            name="Unit Tests: overall",
            status=status,
            message=(
                f"Total={total}, Passed={passed}, Failed={failed}, "
                f"Errors={errors}, Skipped={skipped} | "
                f"Return code: {rc}"
            ),
            instance_a_value={
                "total": total, "passed": passed, "failed": failed,
                "errors": errors, "skipped": skipped,
            },
            duration_seconds=time.time() - t0,
            details={"returncode": rc, "stdout_tail": stdout[-2000:] if stdout else ""},
        ))

        # Report individual test file results
        for test_file in UNIT_TEST_FILES:
            test_path = UNIT_TEST_DIR / test_file
            if not test_path.exists():
                report.add_result(TestResult(
                    category="existing_tests",
                    name=f"Unit: {test_file}",
                    status=TestStatus.SKIP,
                    message=f"File not found: {test_path}",
                ))
                continue

            tf0 = time.time()
            file_result, file_stdout, file_rc = _run_pytest(
                str(test_path),
                extra_args=["--timeout=60"],
                timeout=120,
                json_report=False,
            )

            parsed = _parse_stdout_results(file_stdout)

            if file_rc == 0:
                file_status = TestStatus.PASS
            elif file_rc == 1:
                file_status = TestStatus.FAIL
            elif file_rc == 5:
                file_status = TestStatus.SKIP  # No tests collected
            else:
                file_status = TestStatus.ERROR

            report.add_result(TestResult(
                category="existing_tests",
                name=f"Unit: {test_file}",
                status=file_status,
                message=(
                    f"Passed={parsed['passed']}, Failed={parsed['failed']}, "
                    f"Errors={parsed['errors']}, Skipped={parsed['skipped']}"
                ),
                instance_a_value=parsed,
                duration_seconds=time.time() - tf0,
                details={"returncode": file_rc},
            ))

    # ------------------------------------------------------------------
    # Integration tests
    # ------------------------------------------------------------------

    def _run_integration_tests(self, report: ComparisonReport):
        """Run integration tests from test/ directory."""
        for test_file in INTEGRATION_TEST_FILES:
            test_path = INTEGRATION_TEST_DIR / test_file
            if not test_path.exists():
                report.add_result(TestResult(
                    category="existing_tests",
                    name=f"Integration: {test_file}",
                    status=TestStatus.SKIP,
                    message=f"File not found: {test_path}",
                ))
                continue

            t0 = time.time()
            result, stdout, rc = _run_pytest(
                str(test_path),
                extra_args=["-m", "integration", "--timeout=300"],
                timeout=600,
                json_report=False,
            )

            parsed = _parse_stdout_results(stdout)

            if rc == 0:
                status = TestStatus.PASS
            elif rc == 5:
                status = TestStatus.SKIP
            elif rc == 1:
                status = TestStatus.FAIL
            else:
                status = TestStatus.ERROR

            report.add_result(TestResult(
                category="existing_tests",
                name=f"Integration: {test_file}",
                status=status,
                message=(
                    f"Passed={parsed['passed']}, Failed={parsed['failed']}, "
                    f"Errors={parsed['errors']}, Skipped={parsed['skipped']}"
                ),
                instance_a_value=parsed,
                duration_seconds=time.time() - t0,
                details={"returncode": rc, "stdout_tail": stdout[-1000:] if stdout else ""},
            ))

    # ------------------------------------------------------------------
    # E2E API tests (against both instances)
    # ------------------------------------------------------------------

    def _run_e2e_tests(self, report: ComparisonReport):
        """Run E2E API tests from test/test.py against both instances."""
        if not E2E_TEST_FILE.exists():
            report.add_result(TestResult(
                category="existing_tests",
                name="E2E Tests: file check",
                status=TestStatus.SKIP,
                message=f"E2E test file not found: {E2E_TEST_FILE}",
            ))
            return

        instances = []
        if self.config.instance_a.api_url:
            instances.append(("A", self.config.instance_a))
        if self.config.instance_b.api_url:
            instances.append(("B", self.config.instance_b))

        for label, instance in instances:
            # Run non-destructive E2E tests (skip analysis and clustering which modify state)
            safe_tests = [
                "test_map_visualization",
                "test_annoy_similarity_and_playlist",
            ]

            for test_name in safe_tests:
                t0 = time.time()
                result, stdout, rc = _run_pytest(
                    str(E2E_TEST_FILE),
                    extra_args=["-k", test_name, "--timeout=300"],
                    env_override={"BASE_URL": instance.api_url},
                    timeout=600,
                    json_report=False,
                )

                parsed = _parse_stdout_results(stdout)

                if rc == 0:
                    status = TestStatus.PASS
                elif rc == 5:
                    status = TestStatus.SKIP
                elif rc == 1:
                    status = TestStatus.FAIL
                else:
                    status = TestStatus.ERROR

                report.add_result(TestResult(
                    category="existing_tests",
                    name=f"E2E ({label}): {test_name}",
                    status=status,
                    message=(
                        f"Instance {label} ({instance.api_url}): "
                        f"Passed={parsed['passed']}, Failed={parsed['failed']}"
                    ),
                    instance_a_value=parsed if label == "A" else None,
                    instance_b_value=parsed if label == "B" else None,
                    duration_seconds=time.time() - t0,
                    details={"returncode": rc, "instance": label,
                             "api_url": instance.api_url,
                             "stdout_tail": stdout[-500:] if stdout else ""},
                ))

    # ------------------------------------------------------------------
    # Discovery: list all available tests
    # ------------------------------------------------------------------

    @staticmethod
    def discover_tests() -> dict:
        """Discover all available tests and return a structured summary."""
        discovery = {
            "unit_tests": [],
            "integration_tests": [],
            "e2e_tests": [],
        }

        # Unit tests
        if UNIT_TEST_DIR.exists():
            for f in sorted(UNIT_TEST_DIR.glob("test_*.py")):
                discovery["unit_tests"].append({
                    "file": str(f.relative_to(PROJECT_ROOT)),
                    "name": f.stem,
                    "exists": True,
                })

        # Integration tests
        for f in INTEGRATION_TEST_FILES:
            path = INTEGRATION_TEST_DIR / f
            discovery["integration_tests"].append({
                "file": str(path.relative_to(PROJECT_ROOT)),
                "name": Path(f).stem,
                "exists": path.exists(),
            })

        # E2E tests
        if E2E_TEST_FILE.exists():
            for name in E2E_TEST_NAMES:
                discovery["e2e_tests"].append({
                    "file": str(E2E_TEST_FILE.relative_to(PROJECT_ROOT)),
                    "name": name,
                    "exists": True,
                })

        return discovery
