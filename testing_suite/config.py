"""
Configuration for the AudioMuse-AI Testing & Comparison Suite.

Defines connection parameters for two instances (Instance A / Instance B)
which typically correspond to main branch and feature branch deployments.

Configuration can be provided via:
  1. Environment variables (INSTANCE_A_*, INSTANCE_B_*)
  2. A YAML config file (--config flag)
  3. CLI arguments
"""

import os
import json
import logging
from dataclasses import dataclass, field, asdict
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class InstanceConfig:
    """Connection configuration for a single AudioMuse-AI instance."""

    # Identity
    name: str = "instance"
    branch: str = "unknown"

    # API connection
    api_url: str = "http://localhost:8000"
    api_timeout: int = 120

    # PostgreSQL connection
    pg_host: str = "localhost"
    pg_port: int = 5432
    pg_user: str = "audiomuse"
    pg_password: str = "audiomusepassword"
    pg_database: str = "audiomusedb"

    # Redis connection
    redis_url: str = "redis://localhost:6379/0"

    # Docker container names (for log collection)
    docker_flask_container: str = "audiomuse-ai-flask-app"
    docker_worker_container: str = "audiomuse-ai-worker-instance"
    docker_postgres_container: str = "audiomuse-postgres"
    docker_redis_container: str = "audiomuse-redis"

    # Docker compose file (optional, for status checks)
    docker_compose_file: str = ""

    # SSH details if instances are remote
    ssh_host: str = ""
    ssh_user: str = ""
    ssh_key: str = ""
    ssh_port: int = 22

    @property
    def pg_dsn(self) -> str:
        """Construct PostgreSQL DSN from components."""
        from urllib.parse import quote
        user = quote(self.pg_user, safe='')
        password = quote(self.pg_password, safe='')
        return f"postgresql://{user}:{password}@{self.pg_host}:{self.pg_port}/{self.pg_database}"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ComparisonConfig:
    """Top-level configuration for the comparison suite."""

    # Instance configurations
    instance_a: InstanceConfig = field(default_factory=lambda: InstanceConfig(
        name="main", branch="main"
    ))
    instance_b: InstanceConfig = field(default_factory=lambda: InstanceConfig(
        name="feature", branch="feature"
    ))

    # Test control flags
    run_api_tests: bool = True
    run_db_tests: bool = True
    run_docker_tests: bool = True
    run_performance_tests: bool = True
    run_existing_unit_tests: bool = True
    run_existing_integration_tests: bool = True

    # Performance test settings
    perf_warmup_requests: int = 3
    perf_benchmark_requests: int = 10
    perf_concurrent_users: int = 5

    # API test settings
    api_retries: int = 3
    api_retry_delay: float = 2.0
    api_task_timeout: int = 1200  # 20 minutes for long-running tasks

    # Advanced test groups
    run_setup_crud_tests: bool = True     # Setup wizard provider CRUD (feature-only, creates/deletes test data)
    run_task_start_tests: bool = False    # Task start smoke tests (triggers analysis/clustering work)

    # Database comparison thresholds
    db_row_count_tolerance_pct: float = 5.0  # % difference allowed in row counts
    db_embedding_dimension_expected: int = 200
    db_clap_dimension_expected: int = 512
    db_score_null_threshold_pct: float = 10.0  # Max % of NULL values in critical columns

    # Reporting
    output_dir: str = "testing_suite/reports/output"
    report_format: str = "html"  # html, json, or both
    verbose: bool = False

    # Test track references for functional tests
    test_track_artist_1: str = "Red Hot Chili Peppers"
    test_track_title_1: str = "By the Way"
    test_track_artist_2: str = "System of a Down"
    test_track_title_2: str = "Attack"

    def to_dict(self) -> dict:
        return {
            "instance_a": self.instance_a.to_dict(),
            "instance_b": self.instance_b.to_dict(),
            "run_api_tests": self.run_api_tests,
            "run_db_tests": self.run_db_tests,
            "run_docker_tests": self.run_docker_tests,
            "run_performance_tests": self.run_performance_tests,
            "run_existing_unit_tests": self.run_existing_unit_tests,
            "run_existing_integration_tests": self.run_existing_integration_tests,
            "perf_warmup_requests": self.perf_warmup_requests,
            "perf_benchmark_requests": self.perf_benchmark_requests,
            "perf_concurrent_users": self.perf_concurrent_users,
            "output_dir": self.output_dir,
            "report_format": self.report_format,
        }


def load_config_from_yaml(path: str) -> ComparisonConfig:
    """Load comparison config from a YAML file."""
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML is required for YAML config files: pip install pyyaml")

    with open(path, 'r') as f:
        data = yaml.safe_load(f)

    config = ComparisonConfig()

    if 'instance_a' in data:
        for k, v in data['instance_a'].items():
            if hasattr(config.instance_a, k):
                setattr(config.instance_a, k, v)

    if 'instance_b' in data:
        for k, v in data['instance_b'].items():
            if hasattr(config.instance_b, k):
                setattr(config.instance_b, k, v)

    # Top-level settings
    for k, v in data.items():
        if k not in ('instance_a', 'instance_b') and hasattr(config, k):
            setattr(config, k, v)

    return config


def load_config_from_env() -> ComparisonConfig:
    """Load comparison config from environment variables."""
    config = ComparisonConfig()

    # Instance A
    a = config.instance_a
    a.name = os.getenv("INSTANCE_A_NAME", a.name)
    a.branch = os.getenv("INSTANCE_A_BRANCH", a.branch)
    a.api_url = os.getenv("INSTANCE_A_API_URL", a.api_url)
    a.pg_host = os.getenv("INSTANCE_A_PG_HOST", a.pg_host)
    a.pg_port = int(os.getenv("INSTANCE_A_PG_PORT", str(a.pg_port)))
    a.pg_user = os.getenv("INSTANCE_A_PG_USER", a.pg_user)
    a.pg_password = os.getenv("INSTANCE_A_PG_PASSWORD", a.pg_password)
    a.pg_database = os.getenv("INSTANCE_A_PG_DATABASE", a.pg_database)
    a.redis_url = os.getenv("INSTANCE_A_REDIS_URL", a.redis_url)
    a.docker_flask_container = os.getenv("INSTANCE_A_FLASK_CONTAINER", a.docker_flask_container)
    a.docker_worker_container = os.getenv("INSTANCE_A_WORKER_CONTAINER", a.docker_worker_container)
    a.docker_postgres_container = os.getenv("INSTANCE_A_PG_CONTAINER", a.docker_postgres_container)
    a.ssh_host = os.getenv("INSTANCE_A_SSH_HOST", a.ssh_host)
    a.ssh_user = os.getenv("INSTANCE_A_SSH_USER", a.ssh_user)
    a.ssh_key = os.getenv("INSTANCE_A_SSH_KEY", a.ssh_key)

    # Instance B
    b = config.instance_b
    b.name = os.getenv("INSTANCE_B_NAME", b.name)
    b.branch = os.getenv("INSTANCE_B_BRANCH", b.branch)
    b.api_url = os.getenv("INSTANCE_B_API_URL", b.api_url)
    b.pg_host = os.getenv("INSTANCE_B_PG_HOST", b.pg_host)
    b.pg_port = int(os.getenv("INSTANCE_B_PG_PORT", str(b.pg_port)))
    b.pg_user = os.getenv("INSTANCE_B_PG_USER", b.pg_user)
    b.pg_password = os.getenv("INSTANCE_B_PG_PASSWORD", b.pg_password)
    b.pg_database = os.getenv("INSTANCE_B_PG_DATABASE", b.pg_database)
    b.redis_url = os.getenv("INSTANCE_B_REDIS_URL", b.redis_url)
    b.docker_flask_container = os.getenv("INSTANCE_B_FLASK_CONTAINER", b.docker_flask_container)
    b.docker_worker_container = os.getenv("INSTANCE_B_WORKER_CONTAINER", b.docker_worker_container)
    b.docker_postgres_container = os.getenv("INSTANCE_B_PG_CONTAINER", b.docker_postgres_container)
    b.ssh_host = os.getenv("INSTANCE_B_SSH_HOST", b.ssh_host)
    b.ssh_user = os.getenv("INSTANCE_B_SSH_USER", b.ssh_user)
    b.ssh_key = os.getenv("INSTANCE_B_SSH_KEY", b.ssh_key)

    # Global settings
    config.verbose = os.getenv("COMPARISON_VERBOSE", "false").lower() == "true"
    config.output_dir = os.getenv("COMPARISON_OUTPUT_DIR", config.output_dir)
    config.report_format = os.getenv("COMPARISON_REPORT_FORMAT", config.report_format)

    return config
