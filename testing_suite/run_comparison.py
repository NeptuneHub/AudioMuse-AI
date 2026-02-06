#!/usr/bin/env python3
"""
AudioMuse-AI Testing & Comparison Suite - CLI Entry Point

Comprehensive tool to test all features, database quality, API results,
and performance between two AudioMuse-AI instances (e.g., main branch vs feature branch).

Usage:
  # Quick comparison with two API URLs (minimal config):
  python -m testing_suite.run_comparison \
    --url-a http://main-instance:8000 \
    --url-b http://feature-instance:8000

  # Full comparison with database and Docker access:
  python -m testing_suite.run_comparison \
    --url-a http://main:8000 --url-b http://feature:8000 \
    --pg-host-a main-db-host --pg-host-b feature-db-host \
    --flask-container-a audiomuse-main-flask --flask-container-b audiomuse-feature-flask

  # From YAML config file:
  python -m testing_suite.run_comparison --config comparison_config.yaml

  # Only run specific test categories:
  python -m testing_suite.run_comparison \
    --url-a http://main:8000 --url-b http://feature:8000 \
    --only api,performance

  # Skip slow tests:
  python -m testing_suite.run_comparison \
    --url-a http://main:8000 --url-b http://feature:8000 \
    --skip docker,existing_tests

  # Discover available tests:
  python -m testing_suite.run_comparison --discover

  # Verbose output:
  python -m testing_suite.run_comparison \
    --url-a http://main:8000 --url-b http://feature:8000 -v
"""

import argparse
import logging
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from testing_suite.config import (
    ComparisonConfig, InstanceConfig,
    load_config_from_yaml, load_config_from_env,
)
from testing_suite.orchestrator import ComparisonOrchestrator
from testing_suite.test_runner.existing_tests import ExistingTestRunner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="AudioMuse-AI Testing & Comparison Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Config source
    parser.add_argument("--config", "-c", type=str, default="",
                        help="Path to YAML config file")

    # Discovery mode
    parser.add_argument("--discover", action="store_true",
                        help="Discover and list all available tests, then exit")

    # Instance A
    grp_a = parser.add_argument_group("Instance A (main/baseline)")
    grp_a.add_argument("--url-a", type=str, default="",
                       help="API URL for instance A (e.g., http://localhost:8000)")
    grp_a.add_argument("--name-a", type=str, default="main",
                       help="Name for instance A (default: main)")
    grp_a.add_argument("--branch-a", type=str, default="main",
                       help="Branch name for instance A")
    grp_a.add_argument("--pg-host-a", type=str, default="",
                       help="PostgreSQL host for instance A")
    grp_a.add_argument("--pg-port-a", type=int, default=5432,
                       help="PostgreSQL port for instance A")
    grp_a.add_argument("--pg-user-a", type=str, default="audiomuse",
                       help="PostgreSQL user for instance A")
    grp_a.add_argument("--pg-pass-a", type=str, default="audiomusepassword",
                       help="PostgreSQL password for instance A")
    grp_a.add_argument("--pg-db-a", type=str, default="audiomusedb",
                       help="PostgreSQL database for instance A")
    grp_a.add_argument("--redis-a", type=str, default="",
                       help="Redis URL for instance A")
    grp_a.add_argument("--flask-container-a", type=str, default="audiomuse-ai-flask-app",
                       help="Docker flask container name for A")
    grp_a.add_argument("--worker-container-a", type=str, default="audiomuse-ai-worker-instance",
                       help="Docker worker container name for A")
    grp_a.add_argument("--ssh-host-a", type=str, default="",
                       help="SSH host for remote Docker access (instance A)")
    grp_a.add_argument("--ssh-user-a", type=str, default="",
                       help="SSH user for remote Docker access (instance A)")
    grp_a.add_argument("--ssh-key-a", type=str, default="",
                       help="SSH key file for remote Docker access (instance A)")

    # Instance B
    grp_b = parser.add_argument_group("Instance B (feature/test)")
    grp_b.add_argument("--url-b", type=str, default="",
                       help="API URL for instance B (e.g., http://localhost:8001)")
    grp_b.add_argument("--name-b", type=str, default="feature",
                       help="Name for instance B (default: feature)")
    grp_b.add_argument("--branch-b", type=str, default="feature",
                       help="Branch name for instance B")
    grp_b.add_argument("--pg-host-b", type=str, default="",
                       help="PostgreSQL host for instance B")
    grp_b.add_argument("--pg-port-b", type=int, default=5432,
                       help="PostgreSQL port for instance B")
    grp_b.add_argument("--pg-user-b", type=str, default="audiomuse",
                       help="PostgreSQL user for instance B")
    grp_b.add_argument("--pg-pass-b", type=str, default="audiomusepassword",
                       help="PostgreSQL password for instance B")
    grp_b.add_argument("--pg-db-b", type=str, default="audiomusedb",
                       help="PostgreSQL database for instance B")
    grp_b.add_argument("--redis-b", type=str, default="",
                       help="Redis URL for instance B")
    grp_b.add_argument("--flask-container-b", type=str, default="audiomuse-ai-flask-app",
                       help="Docker flask container name for B")
    grp_b.add_argument("--worker-container-b", type=str, default="audiomuse-ai-worker-instance",
                       help="Docker worker container name for B")
    grp_b.add_argument("--ssh-host-b", type=str, default="",
                       help="SSH host for remote Docker access (instance B)")
    grp_b.add_argument("--ssh-user-b", type=str, default="",
                       help="SSH user for remote Docker access (instance B)")
    grp_b.add_argument("--ssh-key-b", type=str, default="",
                       help="SSH key file for remote Docker access (instance B)")

    # Test selection
    grp_t = parser.add_argument_group("Test Selection")
    grp_t.add_argument("--only", type=str, default="",
                       help="Only run these categories (comma-separated: api,db,docker,performance,existing_tests)")
    grp_t.add_argument("--skip", type=str, default="",
                       help="Skip these categories (comma-separated)")
    grp_t.add_argument("--skip-setup-crud", action="store_true",
                       help="Skip setup wizard CRUD tests (provider create/update/delete)")
    grp_t.add_argument("--enable-task-starts", action="store_true",
                       help="Enable task start smoke tests (analysis, clustering, cleaning)")

    # Performance settings
    grp_p = parser.add_argument_group("Performance Settings")
    grp_p.add_argument("--warmup", type=int, default=3,
                       help="Warmup requests before benchmarking (default: 3)")
    grp_p.add_argument("--bench-requests", type=int, default=10,
                       help="Benchmark requests per endpoint (default: 10)")
    grp_p.add_argument("--concurrent", type=int, default=5,
                       help="Concurrent users for load test (default: 5)")

    # Output
    grp_o = parser.add_argument_group("Output")
    grp_o.add_argument("--output-dir", "-o", type=str, default="testing_suite/reports/output",
                       help="Output directory for reports")
    grp_o.add_argument("--format", type=str, default="both", choices=["html", "json", "both"],
                       help="Report format (default: both)")
    grp_o.add_argument("-v", "--verbose", action="store_true",
                       help="Verbose output")

    return parser


def build_config(args) -> ComparisonConfig:
    """Build ComparisonConfig from CLI arguments."""
    # Start with YAML file or env if specified
    if args.config:
        config = load_config_from_yaml(args.config)
    else:
        config = load_config_from_env()

    # CLI overrides
    a = config.instance_a
    b = config.instance_b

    if args.url_a:
        a.api_url = args.url_a
    if args.name_a:
        a.name = args.name_a
    if args.branch_a:
        a.branch = args.branch_a
    if args.pg_host_a:
        a.pg_host = args.pg_host_a
    a.pg_port = args.pg_port_a
    if args.pg_user_a:
        a.pg_user = args.pg_user_a
    if args.pg_pass_a:
        a.pg_password = args.pg_pass_a
    if args.pg_db_a:
        a.pg_database = args.pg_db_a
    if args.redis_a:
        a.redis_url = args.redis_a
    if args.flask_container_a:
        a.docker_flask_container = args.flask_container_a
    if args.worker_container_a:
        a.docker_worker_container = args.worker_container_a
    if args.ssh_host_a:
        a.ssh_host = args.ssh_host_a
    if args.ssh_user_a:
        a.ssh_user = args.ssh_user_a
    if args.ssh_key_a:
        a.ssh_key = args.ssh_key_a

    if args.url_b:
        b.api_url = args.url_b
    if args.name_b:
        b.name = args.name_b
    if args.branch_b:
        b.branch = args.branch_b
    if args.pg_host_b:
        b.pg_host = args.pg_host_b
    b.pg_port = args.pg_port_b
    if args.pg_user_b:
        b.pg_user = args.pg_user_b
    if args.pg_pass_b:
        b.pg_password = args.pg_pass_b
    if args.pg_db_b:
        b.pg_database = args.pg_db_b
    if args.redis_b:
        b.redis_url = args.redis_b
    if args.flask_container_b:
        b.docker_flask_container = args.flask_container_b
    if args.worker_container_b:
        b.docker_worker_container = args.worker_container_b
    if args.ssh_host_b:
        b.ssh_host = args.ssh_host_b
    if args.ssh_user_b:
        b.ssh_user = args.ssh_user_b
    if args.ssh_key_b:
        b.ssh_key = args.ssh_key_b

    # Performance settings
    config.perf_warmup_requests = args.warmup
    config.perf_benchmark_requests = args.bench_requests
    config.perf_concurrent_users = args.concurrent

    # Output settings
    config.output_dir = args.output_dir
    config.report_format = args.format
    config.verbose = args.verbose

    # Test selection
    if args.only:
        categories = set(args.only.split(","))
        config.run_api_tests = "api" in categories
        config.run_db_tests = "db" in categories or "database" in categories
        config.run_docker_tests = "docker" in categories
        config.run_performance_tests = "performance" in categories or "perf" in categories
        config.run_existing_unit_tests = "existing_tests" in categories or "unit" in categories
        config.run_existing_integration_tests = "existing_tests" in categories or "integration" in categories

    if args.skip:
        skip = set(args.skip.split(","))
        if "api" in skip:
            config.run_api_tests = False
        if "db" in skip or "database" in skip:
            config.run_db_tests = False
        if "docker" in skip:
            config.run_docker_tests = False
        if "performance" in skip or "perf" in skip:
            config.run_performance_tests = False
        if "existing_tests" in skip or "unit" in skip:
            config.run_existing_unit_tests = False
        if "existing_tests" in skip or "integration" in skip:
            config.run_existing_integration_tests = False

    # Advanced test group flags
    if args.skip_setup_crud:
        config.run_setup_crud_tests = False
    if args.enable_task_starts:
        config.run_task_start_tests = True

    return config


def main():
    parser = build_parser()
    args = parser.parse_args()

    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Discovery mode
    if args.discover:
        discovery = ExistingTestRunner.discover_tests()
        print("\n=== AudioMuse-AI Test Discovery ===\n")

        print(f"Unit Tests ({len(discovery['unit_tests'])} files):")
        for t in discovery["unit_tests"]:
            status = "OK" if t["exists"] else "MISSING"
            print(f"  [{status}] {t['file']}")

        print(f"\nIntegration Tests ({len(discovery['integration_tests'])} files):")
        for t in discovery["integration_tests"]:
            status = "OK" if t["exists"] else "MISSING"
            print(f"  [{status}] {t['file']}")

        print(f"\nE2E API Tests ({len(discovery['e2e_tests'])} tests):")
        for t in discovery["e2e_tests"]:
            print(f"  [OK] {t['name']} ({t['file']})")

        total = (len(discovery["unit_tests"]) +
                 len(discovery["integration_tests"]) +
                 len(discovery["e2e_tests"]))
        print(f"\nTotal: {total} test files/entries discovered.\n")
        return 0

    # Validate minimum config
    config = build_config(args)

    if not config.instance_a.api_url or not config.instance_b.api_url:
        if not args.config:
            parser.error("At least --url-a and --url-b are required "
                         "(or use --config for YAML config)")

    # Run comparison
    orchestrator = ComparisonOrchestrator(config)
    report = orchestrator.run()
    orchestrator.print_summary()

    # Exit code based on results
    if report.total_failed > 0 or report.total_errors > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
