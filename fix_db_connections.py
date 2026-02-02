#!/usr/bin/env python3
"""
Fix all database connection patterns to use context managers.
Converts:
    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute(...)
        conn.commit()
    except:
        conn.rollback()
    finally:
        cur.close()

To:
    with get_db() as conn, conn.cursor() as cur:
        cur.execute(...)
        # Auto-commit/rollback
"""

import re
import os
from pathlib import Path

def fix_file(filepath):
    """Fix database connection patterns in a single file."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    original_content = content
    changes = []
    
    # Pattern 1: Remove standalone conn.commit() calls inside context managers
    # Look for "conn.commit()" that appears inside "with get_db() as conn"
    pattern1 = r'(\s+)conn\.commit\(\)\s*\n'
    matches = list(re.finditer(pattern1, content))
    for match in reversed(matches):
        # Check if this is inside a with statement by looking backwards
        before = content[:match.start()]
        if 'with get_db()' in before.split('\n')[-10:]:  # Check last 10 lines
            content = content[:match.start()] + content[match.end():]
            changes.append(f"Removed conn.commit() at position {match.start()}")
    
    # Pattern 2: Remove standalone conn.rollback() calls inside context managers  
    pattern2 = r'(\s+)conn\.rollback\(\)\s*\n'
    matches = list(re.finditer(pattern2, content))
    for match in reversed(matches):
        before = content[:match.start()]
        if 'with get_db()' in before.split('\n')[-10:]:
            content = content[:match.start()] + content[match.end():]
            changes.append(f"Removed conn.rollback() at position {match.start()}")
    
    # Pattern 3: Remove cur.close() calls inside context managers
    pattern3 = r'(\s+)cur\.close\(\)\s*\n'
    matches = list(re.finditer(pattern3, content))
    for match in reversed(matches):
        before = content[:match.start()]
        if 'with' in before.split('\n')[-10:] and 'cursor()' in before.split('\n')[-10:]:
            content = content[:match.start()] + content[match.end():]
            changes.append(f"Removed cur.close() at position {match.start()}")
    
    if content != original_content:
        with open(filepath, 'w') as f:
            f.write(content)
        return changes
    return None

def main():
    # Files to fix
    files_to_fix = [
        'app_helper.py',
        'app.py',
        'app_map.py',
        'app_helper_artist.py',
        'tasks/voyager_manager.py',
        'tasks/clap_text_search.py',
        'tasks/clustering_postprocessing.py',
        'tasks/mulan_text_search.py',
        'tasks/artist_gmm_manager.py',
    ]
    
    base_dir = Path('/Users/guidocolangiuli/Music/AudioMuse-AI')
    
    total_changes = 0
    for filepath in files_to_fix:
        full_path = base_dir / filepath
        if full_path.exists():
            changes = fix_file(full_path)
            if changes:
                print(f"\n✓ Fixed {filepath}:")
                for change in changes:
                    print(f"  - {change}")
                total_changes += len(changes)
            else:
                print(f"  {filepath}: No changes needed")
    
    print(f"\n{'='*60}")
    print(f"Total changes: {total_changes}")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
