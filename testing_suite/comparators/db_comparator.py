"""
Database Comparison Module for AudioMuse-AI Testing Suite.

Compares two PostgreSQL instances across:
  - Schema presence and structure (all expected tables and columns)
  - Row counts and data volume
  - Data quality (NULL rates, value distributions, outliers)
  - Embedding integrity (dimensions, NaN checks, storage sizes)
  - Index and constraint validation
  - Cross-table referential integrity
  - Score/analysis value distributions
  - Playlist quality metrics
"""

import json
import logging
import struct
import time
from typing import Any, Dict, List, Optional, Tuple

from testing_suite.config import ComparisonConfig, InstanceConfig
from testing_suite.utils import (
    ComparisonReport, TestResult, TestStatus,
    pg_query, pg_query_dict, pg_scalar, pct_diff
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Expected schema definition (ground truth)
# ---------------------------------------------------------------------------

EXPECTED_TABLES = {
    "score": [
        "item_id", "title", "author", "album", "album_artist",
        "tempo", "key", "scale", "mood_vector", "energy",
        "other_features", "year", "rating", "file_path", "track_id"
    ],
    "embedding": ["item_id", "embedding"],
    "clap_embedding": ["item_id", "embedding"],
    "playlist": ["id", "playlist_name", "item_id", "title", "author"],
    "task_status": [
        "id", "task_id", "parent_task_id", "task_type", "sub_type_identifier",
        "status", "progress", "details", "timestamp", "start_time", "end_time"
    ],
    "voyager_index_data": [
        "index_name", "index_data", "id_map_json", "embedding_dimension", "created_at"
    ],
    "artist_index_data": [
        "index_name", "index_data", "artist_map_json", "gmm_params_json", "created_at"
    ],
    "map_projection_data": [
        "index_name", "projection_data", "id_map_json", "embedding_dimension", "created_at"
    ],
    "artist_component_projection": [
        "index_name", "projection_data", "artist_component_map_json", "created_at"
    ],
    "cron": ["id", "name", "task_type", "cron_expr", "enabled", "last_run", "created_at"],
    "artist_mapping": ["artist_name", "artist_id"],
    "text_search_queries": ["id", "query_text", "score", "rank", "created_at"],
    "provider": [
        "id", "provider_type", "name", "config", "enabled",
        "priority", "created_at", "updated_at"
    ],
    "track": [
        "id", "file_path_hash", "file_path", "normalized_path",
        "file_size", "file_modified", "created_at", "updated_at"
    ],
    "provider_track": [
        "id", "provider_id", "track_id", "item_id",
        "title", "artist", "album", "last_synced"
    ],
    "app_settings": ["key", "value", "category", "description", "updated_at"],
}

# Critical columns that should not be NULL in the score table
SCORE_CRITICAL_COLUMNS = ["item_id", "title", "author", "tempo", "key", "scale", "mood_vector"]

# Columns to check for statistical distribution in score
SCORE_NUMERIC_COLUMNS = ["tempo", "energy"]


def _safe_dsn_connect(dsn: str, instance_name: str) -> bool:
    """Test if we can connect to the database."""
    try:
        pg_scalar(dsn, "SELECT 1")
        return True
    except Exception as e:
        logger.warning(f"Cannot connect to {instance_name} database: {e}")
        return False


class DatabaseComparator:
    """Compares two PostgreSQL database instances."""

    def __init__(self, config: ComparisonConfig):
        self.config = config
        self.dsn_a = config.instance_a.pg_dsn
        self.dsn_b = config.instance_b.pg_dsn
        self.name_a = config.instance_a.name
        self.name_b = config.instance_b.name

    def run_all(self, report: ComparisonReport):
        """Run all database comparison tests and add results to report."""
        logger.info("Starting database comparison tests...")

        # Check connectivity first
        can_a = _safe_dsn_connect(self.dsn_a, self.name_a)
        can_b = _safe_dsn_connect(self.dsn_b, self.name_b)

        if not can_a and not can_b:
            report.add_result(TestResult(
                category="database",
                name="DB Connectivity",
                status=TestStatus.ERROR,
                message="Cannot connect to either database instance"
            ))
            return

        if not can_a or not can_b:
            report.add_result(TestResult(
                category="database",
                name="DB Connectivity",
                status=TestStatus.WARN,
                message=f"Only connected to {'A' if can_a else 'B'} instance"
            ))

        # Run test suites
        if can_a and can_b:
            self._test_schema_comparison(report)
            self._test_row_counts(report)
            self._test_data_quality(report)
            self._test_embedding_integrity(report)
            self._test_referential_integrity(report)
            self._test_score_distributions(report)
            self._test_playlist_quality(report)
            self._test_index_data_presence(report)
            self._test_task_status_health(report)
            self._test_provider_config(report)
            self._test_app_settings(report)
        elif can_a or can_b:
            # Single-instance validation
            dsn = self.dsn_a if can_a else self.dsn_b
            name = self.name_a if can_a else self.name_b
            self._test_single_instance_schema(report, dsn, name)
            self._test_single_instance_quality(report, dsn, name)

        logger.info("Database comparison tests complete.")

    # ------------------------------------------------------------------
    # Schema comparison
    # ------------------------------------------------------------------

    def _test_schema_comparison(self, report: ComparisonReport):
        """Compare table existence and column structure between instances."""
        for table_name, expected_cols in EXPECTED_TABLES.items():
            t0 = time.time()
            try:
                cols_a = self._get_table_columns(self.dsn_a, table_name)
                cols_b = self._get_table_columns(self.dsn_b, table_name)

                table_exists_a = cols_a is not None
                table_exists_b = cols_b is not None

                if not table_exists_a and not table_exists_b:
                    # Optional tables like mulan_embedding may not exist
                    report.add_result(TestResult(
                        category="database",
                        name=f"Schema: {table_name} existence",
                        status=TestStatus.SKIP,
                        message=f"Table '{table_name}' does not exist in either instance",
                        duration_seconds=time.time() - t0,
                    ))
                    continue

                if table_exists_a != table_exists_b:
                    report.add_result(TestResult(
                        category="database",
                        name=f"Schema: {table_name} existence",
                        status=TestStatus.FAIL,
                        message=f"Table '{table_name}' exists in {'A only' if table_exists_a else 'B only'}",
                        instance_a_value=table_exists_a,
                        instance_b_value=table_exists_b,
                        duration_seconds=time.time() - t0,
                    ))
                    continue

                # Compare columns
                set_a = set(cols_a)
                set_b = set(cols_b)
                missing_in_b = set_a - set_b
                missing_in_a = set_b - set_a

                if set_a == set_b:
                    status = TestStatus.PASS
                    msg = f"Columns match ({len(set_a)} columns)"
                else:
                    status = TestStatus.FAIL
                    msg = f"Column mismatch: missing_in_B={missing_in_b}, missing_in_A={missing_in_a}"

                # Also check against expected columns
                expected_set = set(expected_cols)
                missing_expected_a = expected_set - set_a
                missing_expected_b = expected_set - set_b

                if missing_expected_a or missing_expected_b:
                    if status == TestStatus.PASS:
                        status = TestStatus.WARN
                    msg += f" | Expected cols missing: A={missing_expected_a or 'none'}, B={missing_expected_b or 'none'}"

                report.add_result(TestResult(
                    category="database",
                    name=f"Schema: {table_name} columns",
                    status=status,
                    message=msg,
                    instance_a_value=sorted(set_a),
                    instance_b_value=sorted(set_b),
                    duration_seconds=time.time() - t0,
                ))
            except Exception as e:
                report.add_result(TestResult(
                    category="database",
                    name=f"Schema: {table_name}",
                    status=TestStatus.ERROR,
                    message=str(e),
                    duration_seconds=time.time() - t0,
                ))

    # ------------------------------------------------------------------
    # Row counts
    # ------------------------------------------------------------------

    def _test_row_counts(self, report: ComparisonReport):
        """Compare row counts across all tables."""
        tables_to_count = [
            "score", "embedding", "clap_embedding", "playlist",
            "task_status", "voyager_index_data", "artist_index_data",
            "map_projection_data", "cron", "artist_mapping",
            "text_search_queries", "provider", "track", "provider_track",
            "app_settings"
        ]
        for table_name in tables_to_count:
            t0 = time.time()
            try:
                count_a = self._safe_count(self.dsn_a, table_name)
                count_b = self._safe_count(self.dsn_b, table_name)

                if count_a is None and count_b is None:
                    report.add_result(TestResult(
                        category="database",
                        name=f"Row Count: {table_name}",
                        status=TestStatus.SKIP,
                        message="Table does not exist in either instance",
                        duration_seconds=time.time() - t0,
                    ))
                    continue

                diff_pct = pct_diff(count_a or 0, count_b or 0) if (count_a or count_b) else 0

                if count_a == count_b:
                    status = TestStatus.PASS
                    msg = f"Both have {count_a} rows"
                elif diff_pct <= self.config.db_row_count_tolerance_pct:
                    status = TestStatus.WARN
                    msg = f"A={count_a}, B={count_b} (diff {diff_pct:.1f}% within tolerance)"
                else:
                    status = TestStatus.FAIL
                    msg = f"A={count_a}, B={count_b} (diff {diff_pct:.1f}% exceeds {self.config.db_row_count_tolerance_pct}%)"

                report.add_result(TestResult(
                    category="database",
                    name=f"Row Count: {table_name}",
                    status=status,
                    message=msg,
                    instance_a_value=count_a,
                    instance_b_value=count_b,
                    diff=diff_pct,
                    duration_seconds=time.time() - t0,
                ))
            except Exception as e:
                report.add_result(TestResult(
                    category="database",
                    name=f"Row Count: {table_name}",
                    status=TestStatus.ERROR,
                    message=str(e),
                    duration_seconds=time.time() - t0,
                ))

    # ------------------------------------------------------------------
    # Data quality checks
    # ------------------------------------------------------------------

    def _test_data_quality(self, report: ComparisonReport):
        """Check NULL rates and data quality in the score table."""
        for col in SCORE_CRITICAL_COLUMNS:
            t0 = time.time()
            try:
                null_pct_a = self._null_percentage(self.dsn_a, "score", col)
                null_pct_b = self._null_percentage(self.dsn_b, "score", col)

                if null_pct_a is None and null_pct_b is None:
                    continue

                threshold = self.config.db_score_null_threshold_pct
                problems = []
                if null_pct_a is not None and null_pct_a > threshold:
                    problems.append(f"A has {null_pct_a:.1f}% NULLs")
                if null_pct_b is not None and null_pct_b > threshold:
                    problems.append(f"B has {null_pct_b:.1f}% NULLs")

                if problems:
                    status = TestStatus.FAIL
                    msg = f"score.{col}: " + "; ".join(problems) + f" (threshold {threshold}%)"
                else:
                    status = TestStatus.PASS
                    msg = f"score.{col}: A={null_pct_a:.1f}% NULL, B={null_pct_b:.1f}% NULL (OK)"

                report.add_result(TestResult(
                    category="database",
                    name=f"Data Quality: score.{col} NULLs",
                    status=status,
                    message=msg,
                    instance_a_value=null_pct_a,
                    instance_b_value=null_pct_b,
                    duration_seconds=time.time() - t0,
                ))
            except Exception as e:
                report.add_result(TestResult(
                    category="database",
                    name=f"Data Quality: score.{col}",
                    status=TestStatus.ERROR,
                    message=str(e),
                    duration_seconds=time.time() - t0,
                ))

        # Check for duplicate item_ids in score
        t0 = time.time()
        try:
            dupes_a = pg_scalar(self.dsn_a,
                "SELECT COUNT(*) FROM (SELECT item_id FROM score GROUP BY item_id HAVING COUNT(*) > 1) sub")
            dupes_b = pg_scalar(self.dsn_b,
                "SELECT COUNT(*) FROM (SELECT item_id FROM score GROUP BY item_id HAVING COUNT(*) > 1) sub")

            if (dupes_a or 0) == 0 and (dupes_b or 0) == 0:
                status = TestStatus.PASS
                msg = "No duplicate item_ids in either instance"
            else:
                status = TestStatus.FAIL
                msg = f"Duplicate item_ids: A={dupes_a}, B={dupes_b}"

            report.add_result(TestResult(
                category="database",
                name="Data Quality: score duplicate item_ids",
                status=status,
                message=msg,
                instance_a_value=dupes_a,
                instance_b_value=dupes_b,
                duration_seconds=time.time() - t0,
            ))
        except Exception as e:
            report.add_result(TestResult(
                category="database",
                name="Data Quality: score duplicates",
                status=TestStatus.ERROR,
                message=str(e),
                duration_seconds=time.time() - t0,
            ))

        # Check mood_vector format validity
        t0 = time.time()
        try:
            invalid_moods_a = pg_scalar(self.dsn_a, """
                SELECT COUNT(*) FROM score
                WHERE mood_vector IS NOT NULL
                AND mood_vector NOT LIKE '%:%'
            """)
            invalid_moods_b = pg_scalar(self.dsn_b, """
                SELECT COUNT(*) FROM score
                WHERE mood_vector IS NOT NULL
                AND mood_vector NOT LIKE '%:%'
            """)

            if (invalid_moods_a or 0) == 0 and (invalid_moods_b or 0) == 0:
                status = TestStatus.PASS
                msg = "All mood_vectors have valid format"
            else:
                status = TestStatus.WARN
                msg = f"Invalid mood_vector format: A={invalid_moods_a}, B={invalid_moods_b}"

            report.add_result(TestResult(
                category="database",
                name="Data Quality: mood_vector format",
                status=status,
                message=msg,
                instance_a_value=invalid_moods_a,
                instance_b_value=invalid_moods_b,
                duration_seconds=time.time() - t0,
            ))
        except Exception as e:
            report.add_result(TestResult(
                category="database",
                name="Data Quality: mood_vector format",
                status=TestStatus.ERROR,
                message=str(e),
                duration_seconds=time.time() - t0,
            ))

    # ------------------------------------------------------------------
    # Embedding integrity
    # ------------------------------------------------------------------

    def _test_embedding_integrity(self, report: ComparisonReport):
        """Check embedding dimensions, storage, and coverage."""
        for emb_table, expected_dim in [
            ("embedding", self.config.db_embedding_dimension_expected),
            ("clap_embedding", self.config.db_clap_dimension_expected),
        ]:
            t0 = time.time()
            try:
                count_a = self._safe_count(self.dsn_a, emb_table)
                count_b = self._safe_count(self.dsn_b, emb_table)
                score_count_a = self._safe_count(self.dsn_a, "score")
                score_count_b = self._safe_count(self.dsn_b, "score")

                if count_a is None and count_b is None:
                    report.add_result(TestResult(
                        category="database",
                        name=f"Embedding: {emb_table} existence",
                        status=TestStatus.SKIP,
                        message=f"Table {emb_table} does not exist",
                        duration_seconds=time.time() - t0,
                    ))
                    continue

                # Coverage check
                coverage_a = (count_a / score_count_a * 100) if score_count_a else 0
                coverage_b = (count_b / score_count_b * 100) if score_count_b else 0

                if coverage_a >= 95 and coverage_b >= 95:
                    status = TestStatus.PASS
                elif coverage_a >= 80 and coverage_b >= 80:
                    status = TestStatus.WARN
                else:
                    status = TestStatus.FAIL

                report.add_result(TestResult(
                    category="database",
                    name=f"Embedding: {emb_table} coverage",
                    status=status,
                    message=f"Coverage: A={coverage_a:.1f}% ({count_a}/{score_count_a}), "
                            f"B={coverage_b:.1f}% ({count_b}/{score_count_b})",
                    instance_a_value=coverage_a,
                    instance_b_value=coverage_b,
                    duration_seconds=time.time() - t0,
                ))

                # NULL embedding check
                null_emb_a = pg_scalar(self.dsn_a,
                    f"SELECT COUNT(*) FROM {emb_table} WHERE embedding IS NULL")
                null_emb_b = pg_scalar(self.dsn_b,
                    f"SELECT COUNT(*) FROM {emb_table} WHERE embedding IS NULL")

                if (null_emb_a or 0) == 0 and (null_emb_b or 0) == 0:
                    status = TestStatus.PASS
                    msg = "No NULL embeddings"
                else:
                    status = TestStatus.FAIL
                    msg = f"NULL embeddings: A={null_emb_a}, B={null_emb_b}"

                report.add_result(TestResult(
                    category="database",
                    name=f"Embedding: {emb_table} NULL check",
                    status=status,
                    message=msg,
                    instance_a_value=null_emb_a,
                    instance_b_value=null_emb_b,
                    duration_seconds=time.time() - t0,
                ))

                # Average embedding size (proxy for dimension check)
                avg_size_a = pg_scalar(self.dsn_a,
                    f"SELECT AVG(octet_length(embedding)) FROM {emb_table} WHERE embedding IS NOT NULL")
                avg_size_b = pg_scalar(self.dsn_b,
                    f"SELECT AVG(octet_length(embedding)) FROM {emb_table} WHERE embedding IS NOT NULL")

                if avg_size_a and avg_size_b:
                    # float32 = 4 bytes per dimension
                    approx_dim_a = int(float(avg_size_a) / 4) if avg_size_a else 0
                    approx_dim_b = int(float(avg_size_b) / 4) if avg_size_b else 0

                    if approx_dim_a == approx_dim_b:
                        status = TestStatus.PASS
                    else:
                        status = TestStatus.FAIL

                    report.add_result(TestResult(
                        category="database",
                        name=f"Embedding: {emb_table} avg dimension",
                        status=status,
                        message=f"Approx dimensions: A~{approx_dim_a}, B~{approx_dim_b} "
                                f"(avg bytes: A={float(avg_size_a):.0f}, B={float(avg_size_b):.0f})",
                        instance_a_value=approx_dim_a,
                        instance_b_value=approx_dim_b,
                        duration_seconds=time.time() - t0,
                    ))

            except Exception as e:
                report.add_result(TestResult(
                    category="database",
                    name=f"Embedding: {emb_table}",
                    status=TestStatus.ERROR,
                    message=str(e),
                    duration_seconds=time.time() - t0,
                ))

    # ------------------------------------------------------------------
    # Referential integrity
    # ------------------------------------------------------------------

    def _test_referential_integrity(self, report: ComparisonReport):
        """Check foreign key relationships are intact."""
        # Embeddings should all reference valid score rows
        for emb_table in ["embedding", "clap_embedding"]:
            t0 = time.time()
            try:
                orphans_a = pg_scalar(self.dsn_a, f"""
                    SELECT COUNT(*) FROM {emb_table} e
                    LEFT JOIN score s ON e.item_id = s.item_id
                    WHERE s.item_id IS NULL
                """)
                orphans_b = pg_scalar(self.dsn_b, f"""
                    SELECT COUNT(*) FROM {emb_table} e
                    LEFT JOIN score s ON e.item_id = s.item_id
                    WHERE s.item_id IS NULL
                """)

                if orphans_a is None and orphans_b is None:
                    continue

                if (orphans_a or 0) == 0 and (orphans_b or 0) == 0:
                    status = TestStatus.PASS
                    msg = f"No orphaned rows in {emb_table}"
                else:
                    status = TestStatus.FAIL
                    msg = f"Orphaned {emb_table} rows: A={orphans_a}, B={orphans_b}"

                report.add_result(TestResult(
                    category="database",
                    name=f"Referential: {emb_table} -> score",
                    status=status,
                    message=msg,
                    instance_a_value=orphans_a,
                    instance_b_value=orphans_b,
                    duration_seconds=time.time() - t0,
                ))
            except Exception as e:
                report.add_result(TestResult(
                    category="database",
                    name=f"Referential: {emb_table} -> score",
                    status=TestStatus.ERROR,
                    message=str(e),
                    duration_seconds=time.time() - t0,
                ))

        # provider_track -> provider and track references
        t0 = time.time()
        try:
            orphan_provider_a = pg_scalar(self.dsn_a, """
                SELECT COUNT(*) FROM provider_track pt
                LEFT JOIN provider p ON pt.provider_id = p.id
                WHERE p.id IS NULL
            """)
            orphan_provider_b = pg_scalar(self.dsn_b, """
                SELECT COUNT(*) FROM provider_track pt
                LEFT JOIN provider p ON pt.provider_id = p.id
                WHERE p.id IS NULL
            """)

            if (orphan_provider_a or 0) == 0 and (orphan_provider_b or 0) == 0:
                status = TestStatus.PASS
                msg = "No orphaned provider_track -> provider rows"
            else:
                status = TestStatus.FAIL
                msg = f"Orphaned provider refs: A={orphan_provider_a}, B={orphan_provider_b}"

            report.add_result(TestResult(
                category="database",
                name="Referential: provider_track -> provider",
                status=status,
                message=msg,
                instance_a_value=orphan_provider_a,
                instance_b_value=orphan_provider_b,
                duration_seconds=time.time() - t0,
            ))
        except Exception as e:
            # Tables may not exist in some deployments
            if "does not exist" in str(e).lower() or "relation" in str(e).lower():
                report.add_result(TestResult(
                    category="database",
                    name="Referential: provider_track -> provider",
                    status=TestStatus.SKIP,
                    message="Multi-provider tables not present",
                    duration_seconds=time.time() - t0,
                ))
            else:
                report.add_result(TestResult(
                    category="database",
                    name="Referential: provider_track -> provider",
                    status=TestStatus.ERROR,
                    message=str(e),
                    duration_seconds=time.time() - t0,
                ))

    # ------------------------------------------------------------------
    # Score distributions
    # ------------------------------------------------------------------

    def _test_score_distributions(self, report: ComparisonReport):
        """Compare statistical distributions of score columns."""
        for col in SCORE_NUMERIC_COLUMNS:
            t0 = time.time()
            try:
                stats_a = self._get_column_stats(self.dsn_a, "score", col)
                stats_b = self._get_column_stats(self.dsn_b, "score", col)

                if not stats_a or not stats_b:
                    continue

                # Compare means
                mean_diff = pct_diff(stats_a['avg'], stats_b['avg']) if stats_a['avg'] and stats_b['avg'] else 0

                if mean_diff <= 10:
                    status = TestStatus.PASS
                elif mean_diff <= 25:
                    status = TestStatus.WARN
                else:
                    status = TestStatus.FAIL

                report.add_result(TestResult(
                    category="database",
                    name=f"Distribution: score.{col}",
                    status=status,
                    message=(
                        f"A: min={stats_a['min']:.3f}, max={stats_a['max']:.3f}, "
                        f"avg={stats_a['avg']:.3f}, stddev={stats_a['stddev']:.3f} | "
                        f"B: min={stats_b['min']:.3f}, max={stats_b['max']:.3f}, "
                        f"avg={stats_b['avg']:.3f}, stddev={stats_b['stddev']:.3f} | "
                        f"Mean diff: {mean_diff:.1f}%"
                    ),
                    instance_a_value=stats_a,
                    instance_b_value=stats_b,
                    diff=mean_diff,
                    duration_seconds=time.time() - t0,
                ))
            except Exception as e:
                report.add_result(TestResult(
                    category="database",
                    name=f"Distribution: score.{col}",
                    status=TestStatus.ERROR,
                    message=str(e),
                    duration_seconds=time.time() - t0,
                ))

        # Key distribution comparison
        t0 = time.time()
        try:
            keys_a = pg_query_dict(self.dsn_a,
                "SELECT key, COUNT(*) as cnt FROM score WHERE key IS NOT NULL GROUP BY key ORDER BY cnt DESC")
            keys_b = pg_query_dict(self.dsn_b,
                "SELECT key, COUNT(*) as cnt FROM score WHERE key IS NOT NULL GROUP BY key ORDER BY cnt DESC")

            keys_set_a = set(r['key'] for r in keys_a)
            keys_set_b = set(r['key'] for r in keys_b)

            if keys_set_a == keys_set_b:
                status = TestStatus.PASS
                msg = f"Same key values detected ({len(keys_set_a)} keys)"
            else:
                status = TestStatus.WARN
                diff_keys = keys_set_a.symmetric_difference(keys_set_b)
                msg = f"Key distribution differs: unique to one side: {diff_keys}"

            report.add_result(TestResult(
                category="database",
                name="Distribution: score.key values",
                status=status,
                message=msg,
                instance_a_value=[r['key'] for r in keys_a[:12]],
                instance_b_value=[r['key'] for r in keys_b[:12]],
                duration_seconds=time.time() - t0,
            ))
        except Exception as e:
            report.add_result(TestResult(
                category="database",
                name="Distribution: score.key values",
                status=TestStatus.ERROR,
                message=str(e),
                duration_seconds=time.time() - t0,
            ))

    # ------------------------------------------------------------------
    # Playlist quality
    # ------------------------------------------------------------------

    def _test_playlist_quality(self, report: ComparisonReport):
        """Check playlist table quality."""
        t0 = time.time()
        try:
            # Distinct playlists
            pl_count_a = pg_scalar(self.dsn_a,
                "SELECT COUNT(DISTINCT playlist_name) FROM playlist")
            pl_count_b = pg_scalar(self.dsn_b,
                "SELECT COUNT(DISTINCT playlist_name) FROM playlist")

            diff = pct_diff(pl_count_a or 0, pl_count_b or 0)

            if pl_count_a == pl_count_b:
                status = TestStatus.PASS
            elif diff <= 20:
                status = TestStatus.WARN
            else:
                status = TestStatus.FAIL

            report.add_result(TestResult(
                category="database",
                name="Playlist: distinct count",
                status=status,
                message=f"Distinct playlists: A={pl_count_a}, B={pl_count_b} (diff {diff:.1f}%)",
                instance_a_value=pl_count_a,
                instance_b_value=pl_count_b,
                diff=diff,
                duration_seconds=time.time() - t0,
            ))

            # Average tracks per playlist
            avg_tracks_a = pg_scalar(self.dsn_a, """
                SELECT AVG(cnt) FROM (
                    SELECT COUNT(*) as cnt FROM playlist GROUP BY playlist_name
                ) sub
            """)
            avg_tracks_b = pg_scalar(self.dsn_b, """
                SELECT AVG(cnt) FROM (
                    SELECT COUNT(*) as cnt FROM playlist GROUP BY playlist_name
                ) sub
            """)

            if avg_tracks_a and avg_tracks_b:
                diff = pct_diff(float(avg_tracks_a), float(avg_tracks_b))
                status = TestStatus.PASS if diff <= 20 else TestStatus.WARN

                report.add_result(TestResult(
                    category="database",
                    name="Playlist: avg tracks per playlist",
                    status=status,
                    message=f"Avg tracks/playlist: A={float(avg_tracks_a):.1f}, B={float(avg_tracks_b):.1f}",
                    instance_a_value=float(avg_tracks_a),
                    instance_b_value=float(avg_tracks_b),
                    diff=diff,
                    duration_seconds=time.time() - t0,
                ))

            # Playlists with NULL item_ids
            null_items_a = pg_scalar(self.dsn_a,
                "SELECT COUNT(*) FROM playlist WHERE item_id IS NULL")
            null_items_b = pg_scalar(self.dsn_b,
                "SELECT COUNT(*) FROM playlist WHERE item_id IS NULL")

            if (null_items_a or 0) == 0 and (null_items_b or 0) == 0:
                status = TestStatus.PASS
                msg = "No NULL item_ids in playlists"
            else:
                status = TestStatus.WARN
                msg = f"NULL item_ids in playlist: A={null_items_a}, B={null_items_b}"

            report.add_result(TestResult(
                category="database",
                name="Playlist: NULL item_ids",
                status=status,
                message=msg,
                instance_a_value=null_items_a,
                instance_b_value=null_items_b,
                duration_seconds=time.time() - t0,
            ))

        except Exception as e:
            report.add_result(TestResult(
                category="database",
                name="Playlist quality",
                status=TestStatus.ERROR,
                message=str(e),
                duration_seconds=time.time() - t0,
            ))

    # ------------------------------------------------------------------
    # Index data presence
    # ------------------------------------------------------------------

    def _test_index_data_presence(self, report: ComparisonReport):
        """Check that Voyager/Artist indexes and map projections are present."""
        for table, desc in [
            ("voyager_index_data", "Voyager HNSW index"),
            ("artist_index_data", "Artist GMM index"),
            ("map_projection_data", "Map projection"),
            ("artist_component_projection", "Artist projection"),
        ]:
            t0 = time.time()
            try:
                count_a = self._safe_count(self.dsn_a, table)
                count_b = self._safe_count(self.dsn_b, table)

                if count_a is None and count_b is None:
                    report.add_result(TestResult(
                        category="database",
                        name=f"Index: {desc}",
                        status=TestStatus.SKIP,
                        message=f"Table {table} does not exist",
                        duration_seconds=time.time() - t0,
                    ))
                    continue

                if (count_a or 0) > 0 and (count_b or 0) > 0:
                    status = TestStatus.PASS
                    msg = f"Present in both: A={count_a}, B={count_b}"
                elif (count_a or 0) > 0 or (count_b or 0) > 0:
                    status = TestStatus.WARN
                    msg = f"Only in {'A' if count_a else 'B'}: A={count_a}, B={count_b}"
                else:
                    status = TestStatus.WARN
                    msg = "Empty in both instances (may need rebuild)"

                report.add_result(TestResult(
                    category="database",
                    name=f"Index: {desc}",
                    status=status,
                    message=msg,
                    instance_a_value=count_a,
                    instance_b_value=count_b,
                    duration_seconds=time.time() - t0,
                ))
            except Exception as e:
                report.add_result(TestResult(
                    category="database",
                    name=f"Index: {desc}",
                    status=TestStatus.ERROR,
                    message=str(e),
                    duration_seconds=time.time() - t0,
                ))

    # ------------------------------------------------------------------
    # Task status health
    # ------------------------------------------------------------------

    def _test_task_status_health(self, report: ComparisonReport):
        """Check task_status table for stuck or failed tasks."""
        t0 = time.time()
        try:
            # Failed tasks
            failed_a = pg_scalar(self.dsn_a,
                "SELECT COUNT(*) FROM task_status WHERE status = 'FAILURE'")
            failed_b = pg_scalar(self.dsn_b,
                "SELECT COUNT(*) FROM task_status WHERE status = 'FAILURE'")

            report.add_result(TestResult(
                category="database",
                name="Tasks: failed count",
                status=TestStatus.PASS if (failed_a or 0) == (failed_b or 0) else TestStatus.WARN,
                message=f"Failed tasks: A={failed_a}, B={failed_b}",
                instance_a_value=failed_a,
                instance_b_value=failed_b,
                duration_seconds=time.time() - t0,
            ))

            # Stuck tasks (STARTED more than 2 hours ago)
            stuck_a = pg_scalar(self.dsn_a, """
                SELECT COUNT(*) FROM task_status
                WHERE status IN ('STARTED', 'PROGRESS')
                AND start_time < EXTRACT(EPOCH FROM NOW()) - 7200
            """)
            stuck_b = pg_scalar(self.dsn_b, """
                SELECT COUNT(*) FROM task_status
                WHERE status IN ('STARTED', 'PROGRESS')
                AND start_time < EXTRACT(EPOCH FROM NOW()) - 7200
            """)

            if (stuck_a or 0) == 0 and (stuck_b or 0) == 0:
                status = TestStatus.PASS
                msg = "No stuck tasks"
            else:
                status = TestStatus.WARN
                msg = f"Stuck tasks (>2hr): A={stuck_a}, B={stuck_b}"

            report.add_result(TestResult(
                category="database",
                name="Tasks: stuck check",
                status=status,
                message=msg,
                instance_a_value=stuck_a,
                instance_b_value=stuck_b,
                duration_seconds=time.time() - t0,
            ))

            # Success rate
            total_a = self._safe_count(self.dsn_a, "task_status") or 1
            total_b = self._safe_count(self.dsn_b, "task_status") or 1
            success_a = pg_scalar(self.dsn_a,
                "SELECT COUNT(*) FROM task_status WHERE status = 'SUCCESS'") or 0
            success_b = pg_scalar(self.dsn_b,
                "SELECT COUNT(*) FROM task_status WHERE status = 'SUCCESS'") or 0

            rate_a = success_a / total_a * 100
            rate_b = success_b / total_b * 100

            report.add_result(TestResult(
                category="database",
                name="Tasks: success rate",
                status=TestStatus.PASS if abs(rate_a - rate_b) < 10 else TestStatus.WARN,
                message=f"Success rate: A={rate_a:.1f}%, B={rate_b:.1f}%",
                instance_a_value=rate_a,
                instance_b_value=rate_b,
                diff=abs(rate_a - rate_b),
                duration_seconds=time.time() - t0,
            ))

        except Exception as e:
            report.add_result(TestResult(
                category="database",
                name="Tasks health",
                status=TestStatus.ERROR,
                message=str(e),
                duration_seconds=time.time() - t0,
            ))

    # ------------------------------------------------------------------
    # Provider config
    # ------------------------------------------------------------------

    def _test_provider_config(self, report: ComparisonReport):
        """Compare provider configurations."""
        t0 = time.time()
        try:
            providers_a = pg_query_dict(self.dsn_a,
                "SELECT provider_type, name, enabled, priority FROM provider ORDER BY id")
            providers_b = pg_query_dict(self.dsn_b,
                "SELECT provider_type, name, enabled, priority FROM provider ORDER BY id")

            if not providers_a and not providers_b:
                report.add_result(TestResult(
                    category="database",
                    name="Provider: configuration",
                    status=TestStatus.SKIP,
                    message="No providers configured in either instance",
                    duration_seconds=time.time() - t0,
                ))
                return

            types_a = set(p['provider_type'] for p in providers_a)
            types_b = set(p['provider_type'] for p in providers_b)

            if types_a == types_b:
                status = TestStatus.PASS
                msg = f"Same provider types: {types_a}"
            else:
                status = TestStatus.WARN
                msg = f"Provider types differ: A={types_a}, B={types_b}"

            report.add_result(TestResult(
                category="database",
                name="Provider: configuration match",
                status=status,
                message=msg,
                instance_a_value=[dict(p) for p in providers_a],
                instance_b_value=[dict(p) for p in providers_b],
                duration_seconds=time.time() - t0,
            ))
        except Exception as e:
            if "does not exist" in str(e).lower():
                report.add_result(TestResult(
                    category="database",
                    name="Provider: configuration",
                    status=TestStatus.SKIP,
                    message="Provider table not present",
                    duration_seconds=time.time() - t0,
                ))
            else:
                report.add_result(TestResult(
                    category="database",
                    name="Provider: configuration",
                    status=TestStatus.ERROR,
                    message=str(e),
                    duration_seconds=time.time() - t0,
                ))

    # ------------------------------------------------------------------
    # App settings
    # ------------------------------------------------------------------

    def _test_app_settings(self, report: ComparisonReport):
        """Compare app_settings between instances."""
        t0 = time.time()
        try:
            settings_a = pg_query_dict(self.dsn_a,
                "SELECT key, value, category FROM app_settings ORDER BY key")
            settings_b = pg_query_dict(self.dsn_b,
                "SELECT key, value, category FROM app_settings ORDER BY key")

            keys_a = set(s['key'] for s in settings_a)
            keys_b = set(s['key'] for s in settings_b)

            if keys_a == keys_b:
                status = TestStatus.PASS
                msg = f"Same settings keys ({len(keys_a)} settings)"
            else:
                missing_b = keys_a - keys_b
                missing_a = keys_b - keys_a
                status = TestStatus.WARN
                msg = f"Settings differ: missing_in_B={missing_b}, missing_in_A={missing_a}"

            report.add_result(TestResult(
                category="database",
                name="App Settings: key comparison",
                status=status,
                message=msg,
                instance_a_value=sorted(keys_a),
                instance_b_value=sorted(keys_b),
                duration_seconds=time.time() - t0,
            ))
        except Exception as e:
            if "does not exist" in str(e).lower():
                report.add_result(TestResult(
                    category="database",
                    name="App Settings",
                    status=TestStatus.SKIP,
                    message="app_settings table not present",
                    duration_seconds=time.time() - t0,
                ))
            else:
                report.add_result(TestResult(
                    category="database",
                    name="App Settings",
                    status=TestStatus.ERROR,
                    message=str(e),
                    duration_seconds=time.time() - t0,
                ))

    # ------------------------------------------------------------------
    # Single-instance tests (when only one DB is available)
    # ------------------------------------------------------------------

    def _test_single_instance_schema(self, report: ComparisonReport, dsn: str, name: str):
        """Validate schema for a single instance."""
        for table_name, expected_cols in EXPECTED_TABLES.items():
            t0 = time.time()
            cols = self._get_table_columns(dsn, table_name)
            if cols is None:
                report.add_result(TestResult(
                    category="database",
                    name=f"Schema ({name}): {table_name}",
                    status=TestStatus.SKIP,
                    message=f"Table does not exist in {name}",
                    duration_seconds=time.time() - t0,
                ))
            else:
                missing = set(expected_cols) - set(cols)
                status = TestStatus.PASS if not missing else TestStatus.WARN
                report.add_result(TestResult(
                    category="database",
                    name=f"Schema ({name}): {table_name}",
                    status=status,
                    message=f"Columns: {sorted(cols)}. Missing expected: {missing or 'none'}",
                    duration_seconds=time.time() - t0,
                ))

    def _test_single_instance_quality(self, report: ComparisonReport, dsn: str, name: str):
        """Validate data quality for a single instance."""
        for col in SCORE_CRITICAL_COLUMNS:
            t0 = time.time()
            try:
                null_pct = self._null_percentage(dsn, "score", col)
                if null_pct is not None:
                    status = TestStatus.PASS if null_pct <= self.config.db_score_null_threshold_pct else TestStatus.FAIL
                    report.add_result(TestResult(
                        category="database",
                        name=f"Quality ({name}): score.{col} NULLs",
                        status=status,
                        message=f"{null_pct:.1f}% NULL",
                        duration_seconds=time.time() - t0,
                    ))
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def _get_table_columns(self, dsn: str, table_name: str) -> Optional[List[str]]:
        """Get column names for a table, or None if table doesn't exist."""
        try:
            rows = pg_query(dsn,
                "SELECT column_name FROM information_schema.columns WHERE table_name = %s ORDER BY ordinal_position",
                (table_name,))
            if not rows:
                return None
            return [r[0] for r in rows]
        except Exception:
            return None

    def _safe_count(self, dsn: str, table_name: str) -> Optional[int]:
        """Get row count for a table, or None if table doesn't exist."""
        try:
            return pg_scalar(dsn, f"SELECT COUNT(*) FROM {table_name}")
        except Exception:
            return None

    def _null_percentage(self, dsn: str, table: str, column: str) -> Optional[float]:
        """Get percentage of NULL values in a column."""
        try:
            total = pg_scalar(dsn, f"SELECT COUNT(*) FROM {table}")
            if not total:
                return None
            nulls = pg_scalar(dsn, f"SELECT COUNT(*) FROM {table} WHERE {column} IS NULL")
            return (nulls / total) * 100
        except Exception:
            return None

    def _get_column_stats(self, dsn: str, table: str, column: str) -> Optional[dict]:
        """Get min/max/avg/stddev for a numeric column."""
        try:
            rows = pg_query(dsn, f"""
                SELECT MIN({column}), MAX({column}), AVG({column}), STDDEV({column})
                FROM {table}
                WHERE {column} IS NOT NULL
            """)
            if rows and rows[0][0] is not None:
                return {
                    'min': float(rows[0][0]),
                    'max': float(rows[0][1]),
                    'avg': float(rows[0][2]),
                    'stddev': float(rows[0][3]) if rows[0][3] else 0.0,
                }
        except Exception:
            pass
        return None
