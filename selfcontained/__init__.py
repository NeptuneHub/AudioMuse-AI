# selfcontained/__init__.py
"""
Self-contained mode adapters for AudioMuse-AI.
Provides database and queue abstraction layers for standalone deployment.
"""

from .db_adapter import get_db_adapter
from .queue_adapter import get_queue_adapter

__all__ = ['get_db_adapter', 'get_queue_adapter']
