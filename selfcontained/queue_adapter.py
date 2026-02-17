# selfcontained/queue_adapter.py
"""
Queue adapter to switch between Redis Queue (RQ) and Huey based on deployment mode.
Uses Huey library which natively supports both Redis and in-memory backends.
Maintains RQ's API interface for compatibility with existing code.
"""

import os
import logging
import threading
from typing import Optional, Callable, Any
from datetime import datetime

logger = logging.getLogger(__name__)

# Thread-local storage for tracking the current job ID in standalone worker threads
_standalone_job_info = threading.local()


class StandaloneJobProxy:
    """Minimal RQ Job-like object for standalone mode worker threads.

    This proxy implements a small subset of the RQ Job interface used by
    tasks (e.g. `is_started`, `is_finished`, `is_stopped`, `meta`,
    `save_meta`, and `get_status`). The goal is to behave sensibly in
    standalone (Huey-backed) mode so task code that expects an RQ job
    object doesn't need special-casing.
    """

    def __init__(self, job_id):
        self.id = job_id
        self._meta = {}
        # Internal runtime flags (conservative defaults for a running job)
        self._stopped = False

    @property
    def meta(self):
        return self._meta

    def save_meta(self):
        # No-op for standalone, but present for compatibility with RQ Job
        return None

    # Compatibility helpers used throughout the codebase -----------------
    def get_status(self):
        """Return a simple status string similar to RQ Job's state.

        Defaults to 'started' for an in-progress standalone job. Other
        terminal checks should consult the database (task status) where
        appropriate.
        """
        return 'started'

    @property
    def is_stopped(self) -> bool:
        """Indicates an external stop request for the job.

        Standalone mode does not currently support direct in-process job
        cancellation via the proxy; return the internal flag (default
        False). Task code should also check DB task status where needed.
        """
        return bool(self._stopped)

    @property
    def is_finished(self) -> bool:
        """Standalone proxy never reports finished (worker is executing).

        Tasks should rely on DB child-task records for authoritative state.
        """
        return False

    @property
    def is_failed(self) -> bool:
        return False

    @property
    def is_canceled(self) -> bool:
        return False

    @property
    def is_started(self) -> bool:
        return True


def get_standalone_current_job():
    """Get the current job in standalone mode (equivalent to rq.get_current_job).
    Returns a StandaloneJobProxy with the correct job_id, or None if not in a worker thread."""
    job_id = getattr(_standalone_job_info, 'job_id', None)
    if job_id:
        return StandaloneJobProxy(job_id)
    return None

# Huey imports
from huey import RedisHuey, MemoryHuey, SqliteHuey
from huey.api import Task as HueyTask


class HueyJobWrapper:
    """
    Wrapper to make Huey Task compatible with RQ Job interface.
    Provides familiar RQ methods for existing code.
    """

    def __init__(self, huey_task: HueyTask, job_id: str = None):
        self.huey_task = huey_task
        # Use the provided job_id (from caller) if available, otherwise fall back to Huey's internal ID
        self.id = job_id if job_id else huey_task.id
    
    @property
    def result(self):
        """Get job result (blocks until complete)."""
        return self.huey_task()
    
    def get_status(self, refresh: bool = True) -> str:
        """
        Get job status in RQ format.
        
        Returns:
            Status string: 'queued', 'started', 'finished', 'failed'
        """
        # Huey doesn't have explicit status, we infer it
        if self.huey_task.is_revoked():
            return 'failed'
        
        # Check if result is ready without blocking
        try:
            result = self.huey_task(blocking=False, preserve=True)
            if result is not None:
                return 'finished'
        except:
            pass
        
        # If not finished and not revoked, it's either queued or started
        # We can't distinguish easily in Huey, default to 'queued'
        return 'queued'
    
    def is_finished(self) -> bool:
        """Check if job is finished."""
        try:
            result = self.huey_task(blocking=False, preserve=True)
            return result is not None
        except:
            return False
    
    def is_failed(self) -> bool:
        """Check if job failed."""
        return self.huey_task.is_revoked()
    
    def is_started(self) -> bool:
        """Check if job is started (hard to determine in Huey)."""
        return not self.is_finished() and not self.is_failed()
    
    def is_queued(self) -> bool:
        """Check if job is queued."""
        return not self.is_finished() and not self.is_failed()
    
    def refresh(self):
        """Refresh job data (no-op for Huey)."""
        pass
    
    def cancel(self):
        """Cancel the job."""
        self.huey_task.revoke()
    
    def delete(self):
        """Delete the job."""
        self.huey_task.revoke()
    
    def save(self):
        """Save job state (no-op for Huey)."""
        pass
    
    def save_meta(self):
        """Save job metadata (no-op for Huey)."""
        pass
    
    @property
    def meta(self):
        """Job metadata (Huey doesn't support this natively)."""
        return {}


class HueyQueueWrapper:
    """
    Wrapper to make Huey compatible with RQ Queue interface.
    """
    
    def __init__(self, huey_instance, name: str = 'default'):
        self.huey = huey_instance
        self.name = name
        self.connection = None  # For RQ compatibility
        self._task_cache = {}  # Cache decorated tasks to prevent re-registration
        self._job_id_map = {}  # Mapping: Huey internal task ID -> caller-provided job_id
    
    def enqueue(self, func: Callable, *args, **kwargs) -> HueyJobWrapper:
        """
        Enqueue a job (mimics RQ's enqueue).
        
        Args:
            func: Function to execute (can be string path like 'module.function' or callable)
            *args: Positional arguments for the function
            **kwargs: Keyword arguments (job options are extracted)
        
        Returns:
            HueyJobWrapper: Wrapped Huey task
        """
        # Extract RQ-specific kwargs that don't apply to Huey
        job_id = kwargs.pop('job_id', None)
        kwargs.pop('timeout', None)
        kwargs.pop('result_ttl', None)
        description = kwargs.pop('description', None)
        kwargs.pop('retry', None)
        kwargs.pop('job_timeout', None)
        
        # Extract actual function args
        func_args = kwargs.pop('args', args)
        func_kwargs = kwargs.pop('kwargs', {})
        
        # If func is a string path, import it
        func_path = None
        if isinstance(func, str):
            func_path = func
            from importlib import import_module
            module_path, func_name = func.rsplit('.', 1)
            module = import_module(module_path)
            func = getattr(module, func_name)
        
        # Use function path or name as cache key
        cache_key = func_path or f"{func.__module__}.{func.__name__}"
        
        # Check if we've already decorated this task
        if cache_key not in self._task_cache:
            logger.debug(f"Registering task: {cache_key}")
            task_decorator = self.huey.task()
            self._task_cache[cache_key] = task_decorator(func)
        
        decorated_func = self._task_cache[cache_key]
        
        # Calling the decorated function enqueues it and returns a Result object
        logger.info(f"Enqueueing task {description or cache_key} to queue '{self.name}' (job_id={job_id})")
        logger.debug(f"Calling decorated function with args={func_args}, kwargs={func_kwargs}")
        task_result = decorated_func(*func_args, **func_kwargs)

        # Store mapping from Huey's internal ID to the caller-provided job_id
        if job_id:
            self._job_id_map[task_result.id] = job_id
            logger.info(f"Task enqueued successfully: huey_id={task_result.id} -> job_id={job_id}")
        else:
            logger.info(f"Task enqueued successfully: {task_result.id}")

        return HueyJobWrapper(task_result, job_id=job_id)
    
    def enqueue_call(self, func: Callable, args: tuple = (), kwargs: dict = None,
                     timeout: int = None, result_ttl: int = None, description: str = None,
                     job_id: str = None) -> HueyJobWrapper:
        """Enqueue a job with explicit parameters (mimics RQ's enqueue_call)."""
        kwargs = kwargs or {}
        
        if hasattr(func, 'task_class'):
            task_result = func(*args, **kwargs)
        else:
            task = self.huey.task()(func)
            task_result = task(*args, **kwargs)
        
        return HueyJobWrapper(task_result)
    
    def count(self) -> int:
        """Get the number of jobs in the queue."""
        # Huey doesn't expose queue size easily
        return 0
    
    def is_empty(self) -> bool:
        """Check if queue is empty."""
        return True  # Approximate
    
    def empty(self):
        """Empty the queue."""
        pass  # Huey doesn't support this directly
    
    def shutdown(self):
        """Shutdown the queue."""
        pass  # Huey handles this internally



class QueueAdapter:
    """Base class for queue adapters."""
    
    def __init__(self):
        self.connection = None
        self.queue_high = None
        self.queue_default = None
    
    def get_connection(self):
        """Get queue connection."""
        raise NotImplementedError
    
    def get_queue(self, name: str):
        """Get a queue by name."""
        raise NotImplementedError
    
    def fetch_job(self, job_id: str):
        """Fetch a job by ID."""
        raise NotImplementedError


class RedisQueueAdapter(QueueAdapter):
    """Redis Queue (RQ) adapter - uses existing Redis implementation."""
    
    def __init__(self, redis_url: str):
        super().__init__()
        from redis import Redis
        from rq import Queue
        
        self.connection = Redis.from_url(
            redis_url,
            socket_connect_timeout=30,
            socket_timeout=60,
            socket_keepalive=True,
            health_check_interval=30,
            retry_on_timeout=True
        )
        
        self.queue_high = Queue('high', connection=self.connection, default_timeout=-1)
        self.queue_default = Queue('default', connection=self.connection, default_timeout=-1)
        
        logger.info("Using Redis Queue (RQ) adapter")
    
    def get_connection(self):
        """Get Redis connection."""
        return self.connection
    
    def get_queue(self, name: str):
        """Get a Redis queue by name."""
        if name == 'high':
            return self.queue_high
        elif name == 'default':
            return self.queue_default
        else:
            from rq import Queue
            return Queue(name, connection=self.connection, default_timeout=-1)
    
    def fetch_job(self, job_id: str):
        """Fetch a job by ID from Redis."""
        from rq.job import Job
        return Job.fetch(job_id, connection=self.connection)


class HueyQueueAdapter(QueueAdapter):
    """Huey queue adapter - uses TWO separate Huey instances for high/default queues."""
    
    def __init__(self, num_workers: int = None):
        super().__init__()
        
        self.num_workers = 2  # Total: 1 worker per queue
        self.consumer_started = False
        
        # Create TWO separate MemoryHuey instances - one per priority
        self.huey_high = MemoryHuey(
            name='audiomuse_high',
            immediate=False,
            results=True,
            store_none=False,
            utc=True
        )
        
        self.huey_default = MemoryHuey(
            name='audiomuse_default',
            immediate=False,
            results=True,
            store_none=False,
            utc=True
        )
        
        # Create a dummy connection object for RQ/Redis compatibility in standalone mode
        class DummyConnection:
            """Dummy Redis connection for standalone mode - no-ops for Redis operations."""
            def __init__(self):
                self.connection_pool = type('ConnectionPool', (), {'connection_kwargs': {}})()
            
            def ping(self):
                return True
            
            def publish(self, channel, message):
                """No-op publish - index updates not needed in standalone mode."""
                logger.debug(f"Dummy publish to {channel}: {message} (no-op in standalone)")
                return 0
            
            def set(self, key, value, nx=False, ex=None):
                """No-op distributed lock - always succeeds in standalone (single instance)."""
                logger.debug(f"Dummy set {key}={value} nx={nx} ex={ex} (no-op in standalone)")
                return True
            
            def eval(self, script, numkeys, *keys_and_args):
                """No-op eval - always succeeds in standalone."""
                logger.debug(f"Dummy eval script (no-op in standalone)")
                return 1
            
            def hgetall(self, key):
                """No-op hgetall - returns empty dict (jobs not stored in Redis in standalone)."""
                logger.debug(f"Dummy hgetall {key} (no-op in standalone, returning empty dict)")
                return {}
        
        self.connection = DummyConnection()
        
        # Create wrapped queues - EACH with its OWN Huey instance
        self.queue_high = HueyQueueWrapper(self.huey_high, 'high')
        self.queue_default = HueyQueueWrapper(self.huey_default, 'default')
        
        logger.info(f"Using 2 separate Huey queues: 'high' and 'default' (1 worker each)")
    
    def get_connection(self):
        """Get dummy connection."""
        return self.connection
    
    def get_queue(self, name: str):
        """Get a Huey queue by name."""
        if name == 'high':
            return self.queue_high
        elif name == 'default':
            return self.queue_default
        else:
            # Unknown queue - use default
            logger.warning(f"Unknown queue '{name}', using default")
            return self.queue_default
    
    def get_huey(self):
        """Get the Huey instance - returns high priority one (for compatibility)."""
        return self.huey_high
    
    def start_workers(self):
        """Start 2 workers: 1 dedicated to HIGH priority queue, 1 to DEFAULT queue."""
        if self.consumer_started:
            logger.warning("Workers already started")
            return
        
        import threading
        import sys
        import time
        
        # Worker function for a specific Huey instance
        def worker_loop(worker_id, huey_instance, queue_name):
            logger.info(f"Worker-{worker_id} started (queue: {queue_name})")
            print(f"[WORKER] Worker-{worker_id} started (queue: {queue_name})", file=sys.stderr, flush=True)
            iteration = 0
            # Get the queue wrapper for job_id mapping lookup
            queue_wrapper = self.queue_high if queue_name == 'high' else self.queue_default
            while True:
                try:
                    # Dequeue and execute tasks from THIS specific queue
                    task = huey_instance.dequeue()
                    if task is not None:
                        # Look up the caller-provided job_id from the mapping
                        huey_task_id = getattr(task, 'id', None)
                        mapped_job_id = queue_wrapper._job_id_map.pop(huey_task_id, None) if huey_task_id else None
                        effective_job_id = mapped_job_id or huey_task_id

                        # Set thread-local so the task can retrieve its own job_id
                        _standalone_job_info.job_id = effective_job_id

                        logger.info(f"Worker-{worker_id} [{queue_name}] executing: {task} (job_id={effective_job_id})")
                        print(f"[WORKER] Worker-{worker_id} [{queue_name}] executing: {task} (job_id={effective_job_id})", file=sys.stderr, flush=True)
                        try:
                            result = huey_instance.execute(task)
                        finally:
                            # Always clear thread-local after execution
                            _standalone_job_info.job_id = None
                        logger.info(f"Worker-{worker_id} [{queue_name}] completed: {task} (job_id={effective_job_id})")
                        print(f"[WORKER] Worker-{worker_id} [{queue_name}] completed: {task} (job_id={effective_job_id})", file=sys.stderr, flush=True)
                    else:
                        # No task available, sleep briefly
                        iteration += 1
                        if iteration % 100 == 0:
                            logger.debug(f"Worker-{worker_id} [{queue_name}] idle, checked {iteration} times")
                        time.sleep(0.1)
                except Exception as e:
                    _standalone_job_info.job_id = None  # Ensure cleanup on error
                    logger.error(f"Worker-{worker_id} [{queue_name}] error: {e}", exc_info=True)
                    time.sleep(1)
        
        # Start worker 0 for HIGH priority queue
        print(f"[QUEUE] Starting Worker-0 for HIGH priority queue...", file=sys.stderr, flush=True)
        thread_high = threading.Thread(target=worker_loop, args=(0, self.huey_high, 'high'), daemon=True, name='HueyWorker-High')
        thread_high.start()
        
        # Start worker 1 for DEFAULT queue
        print(f"[QUEUE] Starting Worker-1 for DEFAULT queue...", file=sys.stderr, flush=True)
        thread_default = threading.Thread(target=worker_loop, args=(1, self.huey_default, 'default'), daemon=True, name='HueyWorker-Default')
        thread_default.start()
        
        self.consumer_started = True
        logger.info(f"✓ Started 2 workers: Worker-0 (high), Worker-1 (default)")
        print(f"[QUEUE] ✓ Started 2 workers: Worker-0 (high), Worker-1 (default)", file=sys.stderr, flush=True)
    
    def fetch_job(self, job_id: str):
        """Fetch a job by ID from Huey."""
        # Huey doesn't support fetching jobs by ID easily
        # We create a wrapper that can check status
        class HueyJobById(HueyJobWrapper):
            def __init__(self, job_id):
                self.id = job_id
                self.huey_task = None  # We don't have the actual task reference
            
            def get_status(self, refresh=True):
                return 'queued'  # Can't determine without task reference
            
            def is_finished(self):
                return False
            
            def is_failed(self):
                return False
        
        return HueyJobById(job_id)


def get_queue_adapter() -> QueueAdapter:
    """
    Factory function to get the appropriate queue adapter based on deployment mode.
    SINGLETON: Returns the same instance every time to ensure shared queue state.
    
    Returns:
        QueueAdapter: Redis RQ or Huey adapter based on DEPLOYMENT_MODE
    """
    # Singleton pattern - store instance as function attribute
    if not hasattr(get_queue_adapter, '_instance'):
        from config import DEPLOYMENT_MODE, REDIS_URL, STANDALONE_WORKER_COUNT
        
        if DEPLOYMENT_MODE == 'standalone':
            logger.info("Using Huey (in-memory) Queue")
            get_queue_adapter._instance = HueyQueueAdapter(num_workers=STANDALONE_WORKER_COUNT)
        else:
            logger.info("Using Redis Queue (RQ)")
            get_queue_adapter._instance = RedisQueueAdapter(REDIS_URL)
    
    return get_queue_adapter._instance

