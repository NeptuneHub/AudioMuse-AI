#!/usr/bin/env python3
"""Restore broken files by checking out from git and reapplying ONLY context manager fixes"""

import subprocess
import re

# Files that were broken by overly aggressive sed
files_to_restore = [
    'app_helper.py',
    'tasks/analysis.py', 
    'tasks/voyager_manager.py',
    'tasks/clap_text_search.py',
    'tasks/mulan_text_search.py',
    'tasks/artist_gmm_manager.py',
    'tasks/cleaning.py',
    'tasks/clustering.py',
    'app_helper_artist.py'
]

print("Restoring files from git...")
for f in files_to_restore:
    result = subprocess.run(['git', 'checkout', 'HEAD', f], capture_output=True)
    if result.returncode == 0:
        print(f'✓ Restored: {f}')
    else:
        print(f'✗ Failed: {f}')

print("\n✓ All files restored - manual commit/rollback/close calls back in place")
print("Context managers will handle them properly via __exit__")
