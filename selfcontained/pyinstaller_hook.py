"""
Runtime hook for PyInstaller to set correct paths when frozen
"""
import os
import sys

# When frozen (running as .app), adjust paths
if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
    # Running in PyInstaller bundle
    bundle_dir = sys._MEIPASS
    
    # Set working directory to bundle directory for relative imports
    os.chdir(bundle_dir)
    
    # Ensure bundle directory is in Python path
    if bundle_dir not in sys.path:
        sys.path.insert(0, bundle_dir)
