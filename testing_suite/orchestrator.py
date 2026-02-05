"""
Main Orchestrator for the AudioMuse-AI Testing & Comparison Suite.

Coordinates all comparison modules and generates the final report.
"""

import json
import logging
import os
import time
from datetime import datetime

from testing_suite.config import ComparisonConfig
from testing_suite.utils import ComparisonReport
from testing_suite.comparators.api_comparator import APIComparator
from testing_suite.comparators.db_comparator import DatabaseComparator
from testing_suite.comparators.docker_comparator import DockerComparator
from testing_suite.comparators.performance_comparator import PerformanceComparator
from testing_suite.test_runner.existing_tests import ExistingTestRunner
from testing_suite.reports.html_report import generate_html_report

logger = logging.getLogger(__name__)


class ComparisonOrchestrator:
    """Orchestrates all comparison modules and produces the final report."""

    def __init__(self, config: ComparisonConfig):
        self.config = config
        self.report = ComparisonReport(
            instance_a_name=config.instance_a.name,
            instance_b_name=config.instance_b.name,
            instance_a_branch=config.instance_a.branch,
            instance_b_branch=config.instance_b.branch,
            config_snapshot=config.to_dict(),
        )

    def run(self) -> ComparisonReport:
        """Run all configured comparison modules and return the report."""
        overall_start = time.time()

        logger.info("=" * 70)
        logger.info("AudioMuse-AI Testing & Comparison Suite")
        logger.info("=" * 70)
        logger.info(f"Instance A: {self.config.instance_a.name} "
                     f"({self.config.instance_a.branch}) "
                     f"@ {self.config.instance_a.api_url}")
        logger.info(f"Instance B: {self.config.instance_b.name} "
                     f"({self.config.instance_b.branch}) "
                     f"@ {self.config.instance_b.api_url}")
        logger.info("-" * 70)

        # Run each module based on config flags
        modules = []

        if self.config.run_db_tests:
            modules.append(("Database Comparison", DatabaseComparator(self.config)))

        if self.config.run_api_tests:
            modules.append(("API Comparison", APIComparator(self.config)))

        if self.config.run_docker_tests:
            modules.append(("Docker Comparison", DockerComparator(self.config)))

        if self.config.run_performance_tests:
            modules.append(("Performance Benchmark", PerformanceComparator(self.config)))

        if self.config.run_existing_unit_tests or self.config.run_existing_integration_tests:
            modules.append(("Existing Tests", ExistingTestRunner(self.config)))

        for name, module in modules:
            logger.info(f"\n{'='*50}")
            logger.info(f"Running: {name}")
            logger.info(f"{'='*50}")
            try:
                module_start = time.time()
                module.run_all(self.report)
                module_duration = time.time() - module_start
                logger.info(f"{name} completed in {module_duration:.1f}s")
            except Exception as e:
                logger.error(f"{name} failed with error: {e}", exc_info=True)
                from testing_suite.utils import TestResult, TestStatus
                self.report.add_result(TestResult(
                    category=name.lower().replace(" ", "_"),
                    name=f"{name}: Module Error",
                    status=TestStatus.ERROR,
                    message=f"Module failed: {str(e)}",
                ))

        # Generate reports
        self._generate_reports()

        overall_duration = time.time() - overall_start
        logger.info(f"\n{'='*70}")
        logger.info(f"Testing complete in {overall_duration:.1f}s")
        logger.info(f"Overall status: {self.report.overall_status.value}")
        logger.info(f"Total: {self.report.total_tests} tests, "
                     f"{self.report.total_passed} passed, "
                     f"{self.report.total_failed} failed, "
                     f"{self.report.total_errors} errors")
        logger.info(f"{'='*70}")

        return self.report

    def _generate_reports(self):
        """Generate output reports in configured formats."""
        os.makedirs(self.config.output_dir, exist_ok=True)
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

        # JSON report (always generated)
        json_path = os.path.join(self.config.output_dir, f"comparison_{timestamp}.json")
        with open(json_path, 'w') as f:
            json.dump(self.report.to_dict(), f, indent=2, default=str)
        logger.info(f"JSON report saved: {json_path}")

        # HTML report
        if self.config.report_format in ("html", "both"):
            html_path = os.path.join(self.config.output_dir, f"comparison_{timestamp}.html")
            generate_html_report(self.report, html_path)
            logger.info(f"HTML report saved: {html_path}")

        # Also save a latest symlink/copy
        for ext in ["json", "html"]:
            src = os.path.join(self.config.output_dir, f"comparison_{timestamp}.{ext}")
            dst = os.path.join(self.config.output_dir, f"comparison_latest.{ext}")
            if os.path.exists(src):
                try:
                    if os.path.exists(dst) or os.path.islink(dst):
                        os.unlink(dst)
                    os.symlink(os.path.basename(src), dst)
                except OSError:
                    # Symlinks may not work on all systems; copy instead
                    import shutil
                    shutil.copy2(src, dst)

    def print_summary(self):
        """Print a concise summary to stdout."""
        print(f"\n{'='*60}")
        print(f"  COMPARISON REPORT SUMMARY")
        print(f"{'='*60}")
        print(f"  Instance A: {self.report.instance_a_name} ({self.report.instance_a_branch})")
        print(f"  Instance B: {self.report.instance_b_name} ({self.report.instance_b_branch})")
        print(f"  Overall: {self.report.overall_status.value}")
        print(f"  Total: {self.report.total_tests} | "
              f"Pass: {self.report.total_passed} | "
              f"Fail: {self.report.total_failed} | "
              f"Error: {self.report.total_errors}")
        print(f"{'='*60}")

        for cat_name, cat in self.report.categories.items():
            indicator = "PASS" if cat.failed == 0 and cat.errors == 0 else "FAIL"
            print(f"  [{indicator:4s}] {cat_name:25s}  "
                  f"P:{cat.passed:3d}  F:{cat.failed:3d}  "
                  f"W:{cat.warned:3d}  S:{cat.skipped:3d}  E:{cat.errors:3d}")

            # Show failed tests
            for r in cat.results:
                if r.status.value in ("FAIL", "ERROR"):
                    print(f"         X {r.name}: {r.message[:80]}")

        print(f"{'='*60}")
        print(f"  Reports: {self.config.output_dir}/comparison_latest.*")
        print(f"{'='*60}\n")
