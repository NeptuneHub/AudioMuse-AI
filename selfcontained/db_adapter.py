"""
Database adapter module for AudioMuse-AI standalone mode.
Provides a unified SQLAlchemy-based interface for both PostgreSQL and SQLite.
Now uses SQLAlchemy ORM/Core expressions instead of raw SQL patching.
"""

import os
import logging
from pathlib import Path
from sqlalchemy import create_engine, event, pool

logger = logging.getLogger(__name__)

# Global singleton instance
_db_adapter_instance = None


class SQLAlchemyAdapter:
    """
    SQLAlchemy-based database adapter supporting both PostgreSQL and SQLite.
    Returns SQLAlchemy engine for use with ORM and Core expressions.
    """
    
    def __init__(self, database_url: str, is_sqlite: bool = False):
        """
        Initialize SQLAlchemy engine with appropriate configuration.
        
        Args:
            database_url: SQLAlchemy connection URL
            is_sqlite: Whether this is a SQLite database
        """
        self.database_url = database_url
        self.is_sqlite = is_sqlite
        
        # Configure engine based on database type
        if is_sqlite:
            # SQLite specific configuration
            self.engine = create_engine(
                database_url,
                poolclass=pool.StaticPool,  # Single connection pool for SQLite
                connect_args={
                    'check_same_thread': False,
                    'timeout': 30.0
                },
                echo=False
            )
            
            # Enable SQLite features
            @event.listens_for(self.engine, "connect")
            def set_sqlite_pragma(dbapi_conn, connection_record):
                cursor = dbapi_conn.cursor()
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.close()
            
        else:
            # PostgreSQL specific configuration
            self.engine = create_engine(
                database_url,
                pool_size=10,
                max_overflow=20,
                pool_pre_ping=True,  # Verify connections before using
                pool_recycle=3600,   # Recycle connections after 1 hour
                connect_args={
                    'connect_timeout': 30,
                    'keepalives': 1,
                    'keepalives_idle': 600,
                    'keepalives_interval': 30,
                    'keepalives_count': 3,
                    'options': '-c statement_timeout=300000'
                },
                echo=False
            )
        
        logger.info(f"SQLAlchemy engine initialized for {'SQLite' if is_sqlite else 'PostgreSQL'}")
    
    def get_engine(self):
        """
        Get the SQLAlchemy engine for ORM/Core query execution.
        
        Returns:
            SQLAlchemy Engine instance
        """
        return self.engine


def get_db_adapter():
    """
    Factory function to get the appropriate SQLAlchemy database adapter based on deployment mode.
    Returns the SQLAlchemy engine directly.
    
    Returns:
        SQLAlchemy Engine instance
    """
    global _db_adapter_instance
    
    if _db_adapter_instance is None:
        from config import DEPLOYMENT_MODE, DATABASE_URL, SQLITE_DATABASE_PATH
        
        if DEPLOYMENT_MODE == 'standalone':
            # Build SQLite connection URL
            sqlite_url = f'sqlite:///{SQLITE_DATABASE_PATH}'
            logger.info(f"Using SQLAlchemy with SQLite: {SQLITE_DATABASE_PATH}")
            _db_adapter_instance = SQLAlchemyAdapter(sqlite_url, is_sqlite=True)
        else:
            logger.info("Using SQLAlchemy with PostgreSQL")
            _db_adapter_instance = SQLAlchemyAdapter(DATABASE_URL, is_sqlite=False)
    
    return _db_adapter_instance.get_engine()
