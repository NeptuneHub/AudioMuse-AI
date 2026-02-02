#!/usr/bin/env python3
"""
Schema migration utility for AudioMuse-AI
Converts PostgreSQL schema to SQLite-compatible schema and vice versa.
"""

import os
import re
import sqlite3
import argparse
import logging
from typing import List, Tuple

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def postgres_to_sqlite_type(pg_type: str) -> str:
    """Convert PostgreSQL data type to SQLite equivalent."""
    type_map = {
        'SERIAL': 'INTEGER PRIMARY KEY AUTOINCREMENT',
        'BIGSERIAL': 'INTEGER PRIMARY KEY AUTOINCREMENT',
        'BYTEA': 'BLOB',
        'TEXT': 'TEXT',
        'VARCHAR': 'TEXT',
        'INTEGER': 'INTEGER',
        'BIGINT': 'INTEGER',
        'REAL': 'REAL',
        'DOUBLE PRECISION': 'REAL',
        'BOOLEAN': 'INTEGER',
        'TIMESTAMP': 'TEXT',
        'DATE': 'TEXT',
        'TIME': 'TEXT',
        'JSON': 'TEXT',
        'JSONB': 'TEXT',
    }
    
    pg_type_upper = pg_type.upper()
    
    # Handle types with parameters like VARCHAR(255)
    for pg, sqlite in type_map.items():
        if pg_type_upper.startswith(pg):
            return sqlite
    
    # Default to TEXT for unknown types
    logger.warning(f"Unknown PostgreSQL type '{pg_type}', using TEXT")
    return 'TEXT'


def convert_postgres_to_sqlite_schema(pg_schema: str) -> str:
    """
    Convert PostgreSQL CREATE TABLE statements to SQLite-compatible format.
    
    Args:
        pg_schema: PostgreSQL schema SQL
    
    Returns:
        SQLite-compatible schema SQL
    """
    # Remove PostgreSQL-specific syntax
    schema = pg_schema
    
    # Convert data types
    schema = re.sub(r'\bSERIAL\b', 'INTEGER', schema, flags=re.IGNORECASE)
    schema = re.sub(r'\bBIGSERIAL\b', 'INTEGER', schema, flags=re.IGNORECASE)
    schema = re.sub(r'\bBYTEA\b', 'BLOB', schema, flags=re.IGNORECASE)
    schema = re.sub(r'\bVARCHAR\s*\(\d+\)', 'TEXT', schema, flags=re.IGNORECASE)
    schema = re.sub(r'\bDOUBLE\s+PRECISION\b', 'REAL', schema, flags=re.IGNORECASE)
    schema = re.sub(r'\bTIMESTAMP\b', 'TEXT', schema, flags=re.IGNORECASE)
    schema = re.sub(r'\bBOOLEAN\b', 'INTEGER', schema, flags=re.IGNORECASE)
    schema = re.sub(r'\bJSONB?\b', 'TEXT', schema, flags=re.IGNORECASE)
    
    # Remove PostgreSQL-specific clauses
    schema = re.sub(r'\bDEFAULT\s+CURRENT_TIMESTAMP\b', "DEFAULT CURRENT_TIMESTAMP", schema, flags=re.IGNORECASE)
    schema = re.sub(r'\bDEFAULT\s+NOW\(\)', "DEFAULT CURRENT_TIMESTAMP", schema, flags=re.IGNORECASE)
    
    # Fix SERIAL PRIMARY KEY (should become INTEGER PRIMARY KEY AUTOINCREMENT)
    schema = re.sub(
        r'\b(\w+)\s+SERIAL\s+PRIMARY\s+KEY\b',
        r'\1 INTEGER PRIMARY KEY AUTOINCREMENT',
        schema,
        flags=re.IGNORECASE
    )
    
    return schema


def initialize_sqlite_schema(db_path: str, schema_sql: str = None):
    """
    Initialize SQLite database with schema.
    
    Args:
        db_path: Path to SQLite database file
        schema_sql: Optional custom schema SQL, otherwise uses default from init_db()
    """
    # Create directory if needed
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    # Connect to database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Enable foreign keys
    cursor.execute("PRAGMA foreign_keys = ON")
    cursor.execute("PRAGMA journal_mode = WAL")
    
    logger.info(f"Initializing SQLite database at {db_path}")
    
    if schema_sql:
        # Use provided schema
        converted_schema = convert_postgres_to_sqlite_schema(schema_sql)
        cursor.executescript(converted_schema)
    else:
        # Use default schema from app_helper.py init_db()
        _create_default_schema(cursor)
    
    conn.commit()
    conn.close()
    
    logger.info(f"Database initialized successfully")


def _create_default_schema(cursor):
    """Create default AudioMuse-AI schema for SQLite."""
    
    # Score table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS score (
            item_id TEXT PRIMARY KEY,
            title TEXT,
            author TEXT,
            album TEXT,
            tempo REAL,
            key TEXT,
            scale TEXT,
            mood_vector TEXT,
            energy REAL,
            other_features TEXT
        )
    """)
    
    # Playlist table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS playlist (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            playlist_name TEXT,
            item_id TEXT,
            title TEXT,
            author TEXT,
            UNIQUE (playlist_name, item_id)
        )
    """)
    
    # Task status table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS task_status (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id TEXT UNIQUE NOT NULL,
            parent_task_id TEXT,
            task_type TEXT NOT NULL,
            sub_type_identifier TEXT,
            status TEXT,
            progress INTEGER DEFAULT 0,
            details TEXT,
            timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
            start_time REAL,
            end_time REAL
        )
    """)
    
    # Embedding table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS embedding (
            item_id TEXT PRIMARY KEY,
            embedding BLOB,
            FOREIGN KEY (item_id) REFERENCES score (item_id) ON DELETE CASCADE
        )
    """)
    
    # CLAP embedding table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS clap_embedding (
            item_id TEXT PRIMARY KEY,
            embedding BLOB,
            FOREIGN KEY (item_id) REFERENCES score (item_id) ON DELETE CASCADE
        )
    """)
    
    # MuLan embedding table (conditional)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS mulan_embedding (
            item_id TEXT PRIMARY KEY,
            embedding BLOB,
            FOREIGN KEY (item_id) REFERENCES score (item_id) ON DELETE CASCADE
        )
    """)
    
    # Voyager index data table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS voyager_index_data (
            index_name TEXT PRIMARY KEY,
            index_data BLOB NOT NULL,
            id_map_json TEXT NOT NULL,
            embedding_dimension INTEGER NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Artist index data table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS artist_index_data (
            index_name TEXT PRIMARY KEY,
            index_data BLOB NOT NULL,
            artist_map_json TEXT NOT NULL,
            gmm_params_json TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Map projection data table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS map_projection_data (
            index_name TEXT PRIMARY KEY,
            projection_data BLOB NOT NULL,
            id_map_json TEXT NOT NULL,
            embedding_dimension INTEGER NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Artist component projection table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS artist_component_projection (
            index_name TEXT PRIMARY KEY,
            projection_data BLOB NOT NULL,
            artist_component_map_json TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Cron table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS cron (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            task_type TEXT NOT NULL,
            cron_expr TEXT NOT NULL,
            enabled INTEGER DEFAULT 0,
            last_run REAL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Artist mapping table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS artist_mapping (
            artist_name TEXT PRIMARY KEY,
            artist_id TEXT
        )
    """)
    
    # Text search queries table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS text_search_queries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query_text TEXT NOT NULL,
            score REAL NOT NULL,
            rank INTEGER NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(rank)
        )
    """)
    
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_text_search_queries_rank ON text_search_queries(rank)")
    
    logger.info("Created default schema tables")


def export_postgres_to_sqlite(pg_url: str, sqlite_path: str):
    """
    Export data from PostgreSQL to SQLite.
    
    Args:
        pg_url: PostgreSQL connection URL
        sqlite_path: Path to SQLite database file
    """
    try:
        import psycopg2
    except ImportError:
        logger.error("psycopg2 not installed. Install with: pip install psycopg2-binary")
        return
    
    logger.info(f"Exporting data from PostgreSQL to {sqlite_path}")
    
    # Connect to PostgreSQL
    pg_conn = psycopg2.connect(pg_url)
    pg_cursor = pg_conn.cursor()
    
    # Connect to SQLite
    sqlite_conn = sqlite3.connect(sqlite_path)
    sqlite_cursor = sqlite_conn.cursor()
    
    # Get list of tables
    pg_cursor.execute("""
        SELECT table_name FROM information_schema.tables
        WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
    """)
    tables = [row[0] for row in pg_cursor.fetchall()]
    
    logger.info(f"Found {len(tables)} tables to export")
    
    for table in tables:
        logger.info(f"Exporting table: {table}")
        
        # Get data from PostgreSQL
        pg_cursor.execute(f"SELECT * FROM {table}")
        rows = pg_cursor.fetchall()
        
        if not rows:
            logger.info(f"  No data in {table}")
            continue
        
        # Get column names
        columns = [desc[0] for desc in pg_cursor.description]
        placeholders = ','.join(['?'] * len(columns))
        
        # Insert into SQLite
        try:
            sqlite_cursor.executemany(
                f"INSERT OR REPLACE INTO {table} ({','.join(columns)}) VALUES ({placeholders})",
                rows
            )
            logger.info(f"  Inserted {len(rows)} rows into {table}")
        except sqlite3.Error as e:
            logger.error(f"  Error inserting into {table}: {e}")
    
    sqlite_conn.commit()
    pg_conn.close()
    sqlite_conn.close()
    
    logger.info("Export completed successfully")


def main():
    parser = argparse.ArgumentParser(description='AudioMuse-AI Schema Migration Utility')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Initialize SQLite
    init_parser = subparsers.add_parser('init-sqlite', help='Initialize SQLite database')
    init_parser.add_argument('db_path', help='Path to SQLite database file')
    init_parser.add_argument('--schema', help='Optional schema SQL file')
    
    # Export from PostgreSQL
    export_parser = subparsers.add_parser('export-pg', help='Export PostgreSQL to SQLite')
    export_parser.add_argument('pg_url', help='PostgreSQL connection URL')
    export_parser.add_argument('sqlite_path', help='Path to SQLite database file')
    
    # Convert schema
    convert_parser = subparsers.add_parser('convert-schema', help='Convert PostgreSQL schema to SQLite')
    convert_parser.add_argument('input_file', help='Input PostgreSQL schema file')
    convert_parser.add_argument('output_file', help='Output SQLite schema file')
    
    args = parser.parse_args()
    
    if args.command == 'init-sqlite':
        schema_sql = None
        if args.schema:
            with open(args.schema, 'r') as f:
                schema_sql = f.read()
        initialize_sqlite_schema(args.db_path, schema_sql)
    
    elif args.command == 'export-pg':
        export_postgres_to_sqlite(args.pg_url, args.sqlite_path)
    
    elif args.command == 'convert-schema':
        with open(args.input_file, 'r') as f:
            pg_schema = f.read()
        sqlite_schema = convert_postgres_to_sqlite_schema(pg_schema)
        with open(args.output_file, 'w') as f:
            f.write(sqlite_schema)
        logger.info(f"Converted schema written to {args.output_file}")
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
