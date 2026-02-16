#!/usr/bin/env python3
"""
AudioMuse-AI Standalone Launcher
Starts the application in standalone mode with SQLite + Huey Queue
"""

import os
import sys
import logging
from pathlib import Path

# Add parent directory to Python path so we can import from selfcontained and app
sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

# ===================================================================
# CRITICAL: Set standalone mode BEFORE any other imports
# This ensures config.py picks up the right deployment mode
# ===================================================================
os.environ['DEPLOYMENT_MODE'] = 'standalone'

# Set up logging to file when running as packaged app
log_handlers = [logging.StreamHandler()]

# Custom handler that prepends new log entries to the top of the log file
class PrependFileHandler(logging.Handler):
    """Logging handler that writes newest log lines at the top of the file.

    Note: prepending requires rewriting the file on each emit. This is
    intentional per UX request (newest-first log file) but may be slower
    for very large files.
    """
    def __init__(self, filename, encoding='utf-8'):
        super().__init__()
        self.filename = str(filename)
        self.encoding = encoding
        # ensure file exists
        open(self.filename, 'a', encoding=self.encoding).close()

    def emit(self, record):
        try:
            msg = self.format(record) + '\n'
            # Write atomically: create temp file in same dir then replace
            import tempfile, os
            dir_name = os.path.dirname(self.filename) or '.'
            fd, tmp_path = tempfile.mkstemp(dir=dir_name)
            try:
                with os.fdopen(fd, 'w', encoding=self.encoding) as tmpf:
                    tmpf.write(msg)
                    # copy existing contents in chunks to avoid high memory use
                    if os.path.exists(self.filename):
                        with open(self.filename, 'r', encoding=self.encoding, errors='ignore') as oldf:
                            while True:
                                chunk = oldf.read(8192)
                                if not chunk:
                                    break
                                tmpf.write(chunk)
                # atomic replace
                os.replace(tmp_path, self.filename)
            finally:
                if os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except Exception:
                        pass
        except Exception:
            self.handleError(record)

if getattr(sys, 'frozen', False):
    # Running as PyInstaller bundle - also log to file (newest-first)
    data_dir = Path.home() / '.audiomuse'
    data_dir.mkdir(parents=True, exist_ok=True)
    log_file = data_dir / 'audiomuse.log'
    file_handler = PrependFileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    log_handlers.append(file_handler)
    print(f"AudioMuse-AI is starting... Logs: {log_file}")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=log_handlers
)
logger = logging.getLogger(__name__)

def setup_standalone_environment():
    """Set up environment for standalone mode."""
    
    # Deployment mode already set above (before imports)
    
    # Set up data directory
    data_dir = Path.home() / '.audiomuse'
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up model and cache directories
    model_dir = data_dir / 'model'
    cache_dir = data_dir / '.cache'
    model_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Set model paths in environment
    os.environ['MODEL_DIR'] = str(model_dir)
    os.environ['HF_HOME'] = str(cache_dir / 'huggingface')
    os.environ['TRANSFORMERS_CACHE'] = str(cache_dir / 'huggingface')

    # Ensure SQLite DB path is available early so modules that import `config`
    # (which reads environment at import-time) will pick up the correct value.
    if 'SQLITE_DATABASE_PATH' not in os.environ:
        db_path = data_dir / 'audiomuse.db'
        os.environ['SQLITE_DATABASE_PATH'] = str(db_path)
    else:
        db_path = Path(os.environ['SQLITE_DATABASE_PATH'])

    # Check system dependencies
    from selfcontained.model_downloader import check_system_dependencies
    check_system_dependencies()
    
    # Report preferred ONNX provider (helps verify macOS GPU selection at startup)
    try:
        from tasks.onnx_utils import get_preferred_onnx_provider_options
        prefs = get_preferred_onnx_provider_options()
        top = prefs[0][0] if prefs else 'CPUExecutionProvider'
        logger.info(f"Preferred ONNX provider at startup: {top} (providers: {[p for p,_ in prefs]})")
    except Exception:
        logger.debug("ONNX provider detection skipped (onnxruntime not installed)")

    # Check and download models if needed
    from selfcontained.model_downloader import check_and_download_models
    logger.info("Checking ML models...")
    if not check_and_download_models(model_dir, cache_dir, include_optional=False):
        logger.error("Failed to download required models. Please check your internet connection.")
        sys.exit(1)
    
    # Set worker count if not already set
    # Default to 2 workers: enough to handle multiple task types without overwhelming CLAP
    # CLAP semaphore (in queue_adapter.py) limits concurrent CLAP analyses to 1
    if 'STANDALONE_WORKER_COUNT' not in os.environ:
        os.environ['STANDALONE_WORKER_COUNT'] = '2'
    
    # Load media server configuration from config.ini
    from selfcontained.config_manager import apply_config_to_environment, is_configured
    config_loaded = apply_config_to_environment()
    if not config_loaded:
        logger.warning("╔══════════════════════════════════════════════════════════╗")
        logger.warning("║  ⚠️  Media Server Not Configured                         ║")
        logger.warning("╚══════════════════════════════════════════════════════════╝")
        logger.warning("  Please visit http://localhost:8000/setup to configure")
    
    # Initialize database if it doesn't exist
    if not db_path.exists():
        logger.info(f"Initializing new database at {db_path}")
        # DuckDB will create the database on first connection
        # The init_db() in app_helper.py will create all tables
    else:
        logger.info(f"Using existing database at {db_path}")
    
    logger.info(f"╔══════════════════════════════════════════════════════════╗")
    logger.info(f"║       AudioMuse-AI Standalone Mode                       ║")
    logger.info(f"╚══════════════════════════════════════════════════════════╝")
    logger.info(f"")
    logger.info(f"  Mode:     STANDALONE (DuckDB + Huey)")
    logger.info(f"  Database: {db_path}")
    logger.info(f"  Models:   {model_dir}")
    logger.info(f"  Workers:  {os.environ['STANDALONE_WORKER_COUNT']}")
    logger.info(f"")
    
    return db_path


def main():
    """Main entry point for standalone launcher."""
    
    try:
        # Setup environment
        db_path = setup_standalone_environment()
        
        # Import and run Flask app
        logger.info("Starting AudioMuse-AI server...")
        
        # Add current directory to Python path
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        
        # Import Flask app
        from app import app, close_db, initialize_app

        # Register cleanup
        app.teardown_appcontext(close_db)

        # Start Huey workers in standalone mode
        if os.environ.get('DEPLOYMENT_MODE') == 'standalone':
            from selfcontained.queue_adapter import get_queue_adapter
            adapter = get_queue_adapter()
            if hasattr(adapter, 'start_workers'):
                adapter.start_workers()
                logger.info("✓ Huey workers started")

        # Initialize all in-memory caches, indexes, and background threads
        # This loads Voyager index, artist index, map projections, CLAP/MuLan caches,
        # and starts the cron manager + index reload polling threads.
        logger.info("Initializing caches and indexes...")
        initialize_app()
        logger.info("✓ Initialization complete")

        # Get configuration
        host = os.environ.get('FLASK_HOST', '127.0.0.1')
        port = int(os.environ.get('FLASK_PORT', '8000'))
        debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'

        logger.info(f"Server starting on http://{host}:{port}")
        
        # Check if we should run with menu bar (macOS only)
        use_menubar = sys.platform == 'darwin' and getattr(sys, 'frozen', False)
        
        if use_menubar:
            logger.info("Starting with menu bar icon...")
            
            # Import menu bar app
            from selfcontained.menubar_app import run_menubar_app, RUMPS_AVAILABLE
            
            if RUMPS_AVAILABLE:
                # Start Flask in a separate thread
                def run_flask():
                    app.run(host=host, port=port, debug=False, threaded=True, use_reloader=False)
                
                import threading
                flask_thread = threading.Thread(target=run_flask, daemon=True)
                flask_thread.start()
                
                # Wait a moment for Flask to start
                import time
                time.sleep(2)
                
                # Run menu bar app (this blocks until quit)
                menubar_app = run_menubar_app(f'http://{host}:{port}')
                if menubar_app:
                    menubar_app.update_status('Running')
                    logger.info("✓ Menu bar icon ready")
                    logger.info("Press Ctrl+C to stop, or use menu bar → Quit")
                    menubar_app.run()
                else:
                    # Fallback if rumps not available
                    logger.info("Menu bar not available, running in background")
                    logger.info("Press Ctrl+C to stop")
                    flask_thread.join()
            else:
                logger.info("Menu bar library not available, running without icon")
                logger.info("Press Ctrl+C to stop")
                app.run(host=host, port=port, debug=debug, threaded=True)
        else:
            # Not macOS or not frozen - run normally
            logger.info("Press Ctrl+C to stop")
            app.run(host=host, port=port, debug=debug, threaded=True)
        
    except KeyboardInterrupt:
        logger.info("\nShutting down...")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error starting application: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
