"""
DuckDB wrapper providing psycopg2-compatible interface.
This allows swapping PostgreSQL with DuckDB with ZERO code changes.

DuckDB natively supports PostgreSQL syntax (ILIKE, REGEXP_REPLACE, SERIAL, etc.)

Key translations handled:
  - %s -> ? placeholder conversion with tuple expansion for IN clauses
  - = ANY(%s) -> IN (?, ?, ...) conversion
  - octet_length() -> octet_length() (DuckDB supports it natively)
  - psycopg2.Binary unwrapping
  - SERIAL -> INTEGER with sequence
  - Foreign key CASCADE/SET NULL removal
"""
import duckdb
import logging
import re
from typing import Any, Optional, Tuple, List
from psycopg2.extras import DictCursor

logger = logging.getLogger(__name__)


def _unwrap_binary(val):
    """Unwrap psycopg2.Binary objects to raw bytes for DuckDB."""
    # psycopg2.Binary returns an adapter object; extract raw bytes
    if hasattr(val, 'adapted'):
        return bytes(val.adapted)
    if hasattr(val, 'getquoted'):
        # It's a psycopg2 adapter - get the wrapped value
        if hasattr(val, 'adapted'):
            return bytes(val.adapted)
    return val


def _convert_query_and_params(query: str, params: Optional[tuple] = None):
    """
    Convert a psycopg2-style query with %s placeholders to DuckDB ? placeholders.

    Handles:
      1. Tuple expansion for IN %s: IN %s with tuple param -> IN (?, ?, ...)
      2. ANY(%s) conversion: = ANY(%s) with list/tuple param -> IN (?, ?, ...)
      3. Scalar %s -> ? with param unwrapping (psycopg2.Binary -> bytes)
      4. PostgreSQL function replacements (octet_length -> octet_length, etc.)
      5. SERIAL -> INTEGER with sequence
      6. Foreign key CASCADE removal

    Returns:
      (converted_query, converted_params_tuple)
    """
    if params is None:
        return _convert_sql_syntax(query), None

    # Step 1: Convert PostgreSQL-specific SQL syntax (non-param related)
    query = _convert_sql_syntax(query)

    # Step 2: Convert = ANY(%s) to IN %s BEFORE doing placeholder replacement.
    # This normalizes the query so all list-param patterns use IN %s.
    query = re.sub(r'=\s*ANY\s*\(\s*%s\s*\)', 'IN %s', query)

    # Step 3: Walk through the query, replacing %s with ? or (?, ?, ...)
    #   depending on whether the corresponding param is a tuple/list.
    new_query_parts = []
    new_params = []
    param_idx = 0
    i = 0
    qlen = len(query)

    while i < qlen:
        if i + 1 < qlen and query[i] == '%' and query[i + 1] == 's':
            if param_idx < len(params):
                param = params[param_idx]
                param = _unwrap_binary(param)

                if isinstance(param, (tuple, list)):
                    # Tuple/list param: expand for IN clause
                    if len(param) == 0:
                        # Empty tuple: use impossible condition
                        new_query_parts.append('(NULL)')
                    else:
                        unwrapped = [_unwrap_binary(p) for p in param]
                        placeholders = ', '.join(['?'] * len(unwrapped))
                        new_query_parts.append(f'({placeholders})')
                        new_params.extend(unwrapped)
                else:
                    new_query_parts.append('?')
                    new_params.append(param)
                param_idx += 1
            else:
                # More %s than params - shouldn't happen, but be safe
                new_query_parts.append('?')
            i += 2
        else:
            new_query_parts.append(query[i])
            i += 1

    return ''.join(new_query_parts), tuple(new_params)


def _convert_sql_syntax(query: str) -> str:
    """Convert PostgreSQL-specific SQL syntax to DuckDB equivalents."""
    # Convert SERIAL to INTEGER with default nextval
    query = re.sub(
        r'(\w+)\s+SERIAL\s+PRIMARY\s+KEY',
        r"\1 INTEGER PRIMARY KEY DEFAULT nextval('\1_seq')",
        query, flags=re.IGNORECASE
    )
    query = re.sub(
        r'(\w+)\s+SERIAL(?!\s+PRIMARY)',
        r"\1 INTEGER DEFAULT nextval('\1_seq')",
        query, flags=re.IGNORECASE
    )

    # Remove CASCADE/SET NULL/SET DEFAULT from foreign keys
    for clause in (
        ' ON DELETE CASCADE', ' ON UPDATE CASCADE',
        ' ON DELETE SET NULL', ' ON UPDATE SET NULL',
        ' ON DELETE SET DEFAULT', ' ON UPDATE SET DEFAULT',
    ):
        query = query.replace(clause, '')

    # Remove schema prefix "public." (DuckDB uses main schema by default)
    query = re.sub(r'\bpublic\.', '', query)

    # Convert CURRENT_TIMESTAMP to now() - DuckDB handles this better in value contexts
    query = re.sub(
        r'\bCURRENT_TIMESTAMP\b',
        'now()',
        query,
        flags=re.IGNORECASE
    )

    return query


def _maybe_create_sequences(cursor, query: str):
    """Create sequences for SERIAL columns if this is a CREATE TABLE with nextval."""
    if 'CREATE TABLE' in query.upper() and 'nextval' in query:
        seq_names = re.findall(r"nextval\('(\w+_seq)'\)", query)
        for seq_name in seq_names:
            try:
                cursor.execute(f"CREATE SEQUENCE IF NOT EXISTS {seq_name} START 1")
            except Exception:
                pass  # Sequence might already exist


class DuckDBDictCursor:
    """DuckDB cursor that returns dict-like rows (like psycopg2.extras.DictCursor)"""

    def __init__(self, conn):
        self._conn = conn
        self._cursor = conn.cursor()
        self.rowcount = -1
        self._description = None

    def execute(self, query: str, params: Optional[Tuple] = None):
        """Execute query - converts PostgreSQL syntax to DuckDB"""
        _maybe_create_sequences(self._cursor, query)
        query, params = _convert_query_and_params(query, params)
        if params:
            self._cursor.execute(query, params)
        else:
            self._cursor.execute(query)
        self._description = self._cursor.description
        self.rowcount = -1

    def fetchone(self):
        """Fetch one row as dict"""
        row = self._cursor.fetchone()
        if row and self._description:
            return dict(zip([desc[0] for desc in self._description], row))
        return row

    def fetchall(self):
        """Fetch all rows as list of dicts"""
        rows = self._cursor.fetchall()
        if rows and self._description:
            col_names = [desc[0] for desc in self._description]
            return [dict(zip(col_names, row)) for row in rows]
        return rows

    def fetchmany(self, size: int = 1):
        """Fetch many rows as list of dicts"""
        rows = self._cursor.fetchmany(size)
        if rows and self._description:
            col_names = [desc[0] for desc in self._description]
            return [dict(zip(col_names, row)) for row in rows]
        return rows

    def close(self):
        """Close cursor"""
        self._cursor.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class DuckDBCursor:
    """Standard DuckDB cursor (returns tuples like psycopg2 default cursor)"""

    def __init__(self, conn):
        self._conn = conn
        self._cursor = conn.cursor()
        self.rowcount = -1
        self._description = None

    def execute(self, query: str, params: Optional[Tuple] = None):
        """Execute query - converts PostgreSQL syntax to DuckDB"""
        _maybe_create_sequences(self._cursor, query)
        query, params = _convert_query_and_params(query, params)
        if params:
            self._cursor.execute(query, params)
        else:
            self._cursor.execute(query)
        self._description = self._cursor.description
        self.rowcount = -1

    def fetchone(self):
        return self._cursor.fetchone()

    def fetchall(self):
        return self._cursor.fetchall()

    def fetchmany(self, size: int = 1):
        return self._cursor.fetchmany(size)

    def close(self):
        self._cursor.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class DuckDBConnection:
    """DuckDB connection that behaves like psycopg2 connection"""

    def __init__(self, db_path: str):
        self._conn = duckdb.connect(db_path, read_only=False)
        logger.info(f"DuckDB connection established: {db_path}")
        
        # Track this connection for cleanup
        global _all_connections
        _all_connections.append(self)

    def cursor(self, cursor_factory=None):
        """Return appropriate cursor type"""
        if cursor_factory == DictCursor:
            return DuckDBDictCursor(self._conn)
        return DuckDBCursor(self._conn)

    def commit(self):
        """Commit transaction"""
        self._conn.commit()

    def rollback(self):
        """Rollback transaction"""
        self._conn.rollback()

    def close(self):
        """Close connection"""
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - commit or rollback, but DON'T close connection.

        In standalone mode, connections are cached in thread-local storage and reused.
        Closing them here would break subsequent queries in the same thread.
        """
        try:
            if exc_type:
                self.rollback()
            else:
                self.commit()
        except Exception as e:
            # Ignore transaction errors (e.g., no active transaction)
            logger.debug(f"Ignoring transaction error in __exit__: {e}")
        # DON'T close the connection - it's managed by app_helper's thread-local storage


def connect_duckdb(db_path: str) -> DuckDBConnection:
    """
    Create DuckDB connection with psycopg2-compatible interface.

    Usage:
        conn = connect_duckdb('/path/to/database.duckdb')
        cur = conn.cursor()
        cur.execute("SELECT * FROM table WHERE id = %s", (123,))
        rows = cur.fetchall()
    """
    return DuckDBConnection(db_path)


# Track all connections for cleanup
_all_connections = []

def close_all_connections():
    """Close all open DuckDB connections to release locks."""
    global _all_connections
    for conn in _all_connections:
        try:
            if hasattr(conn, '_conn') and conn._conn:
                conn._conn.close()
        except Exception:
            pass
    _all_connections.clear()

