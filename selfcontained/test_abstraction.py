#!/usr/bin/env python3
"""
Test script to verify both deployment modes work correctly.
Run this to ensure the abstraction layer is functioning properly.
"""

import os
import sys
import time
import tempfile
import shutil
from pathlib import Path

def test_imports():
    """Test that imports work in both modes."""
    print("\n" + "="*60)
    print("TEST 1: Import Test")
    print("="*60)
    
    try:
        # Test standalone imports
        print("\n[Standalone Mode] Testing imports...")
        os.environ['DEPLOYMENT_MODE'] = 'standalone'
        os.environ['SQLITE_DATABASE_PATH'] = ':memory:'
        
        from selfcontained.db_adapter import get_db_adapter, PostgreSQLAdapter, SQLiteAdapter
        from selfcontained.queue_adapter import get_queue_adapter, RedisQueueAdapter, InProcessQueueAdapter
        
        print("✓ Adapter imports successful")
        
        # Test conditional imports in app_helper
        import importlib
        if 'app_helper' in sys.modules:
            del sys.modules['app_helper']
        
        import app_helper
        print("✓ app_helper imports successful in standalone mode")
        
        # Test docker mode
        print("\n[Docker Mode] Testing imports...")
        os.environ['DEPLOYMENT_MODE'] = 'docker'
        
        # Reload app_helper
        if 'app_helper' in sys.modules:
            del sys.modules['app_helper']
        
        import app_helper
        print("✓ app_helper imports successful in docker mode")
        
        print("\n✅ All imports passed")
        return True
        
    except Exception as e:
        print(f"\n❌ Import test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_sqlite_adapter():
    """Test SQLite adapter functionality."""
    print("\n" + "="*60)
    print("TEST 2: SQLite Adapter")
    print("="*60)
    
    try:
        from selfcontained.db_adapter import SQLiteAdapter
        
        # Create temporary database
        temp_db = tempfile.mktemp(suffix='.db')
        print(f"\nUsing temporary database: {temp_db}")
        
        adapter = SQLiteAdapter(temp_db)
        conn = adapter.connect()
        
        # Test basic operations
        print("\n[CREATE TABLE] Creating test table...")
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE test (id INTEGER PRIMARY KEY, name TEXT)")
        conn.commit()
        print("✓ Table created")
        
        # Test insert
        print("\n[INSERT] Inserting test data...")
        cursor.execute("INSERT INTO test (name) VALUES (?)", ("Test Item",))
        conn.commit()
        print("✓ Data inserted")
        
        # Test select
        print("\n[SELECT] Querying data...")
        cursor.execute("SELECT * FROM test")
        rows = cursor.fetchall()
        print(f"✓ Query returned {len(rows)} rows: {rows}")
        
        # Test parameter conversion
        print("\n[PARAMETER CONVERSION] Testing %s -> ? conversion...")
        cursor.execute("SELECT * FROM test WHERE name = %s", ("Test Item",))
        rows = cursor.fetchall()
        print(f"✓ Parameter conversion works: {rows}")
        
        # Cleanup
        adapter.close()
        os.unlink(temp_db)
        
        print("\n✅ SQLite adapter passed all tests")
        return True
        
    except Exception as e:
        print(f"\n❌ SQLite adapter test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_inprocess_queue():
    """Test in-process queue functionality."""
    print("\n" + "="*60)
    print("TEST 3: In-Process Queue")
    print("="*60)
    
    try:
        from selfcontained.queue_adapter import InProcessQueue, JobStatus
        
        # Create queue
        print("\n[QUEUE CREATION] Creating queue with 2 workers...")
        queue = InProcessQueue('test_queue', num_workers=2)
        print(f"✓ Queue created with {queue.num_workers} workers")
        
        # Define test function
        def test_task(x, y):
            """Simple test task."""
            time.sleep(0.1)  # Simulate work
            return x + y
        
        # Enqueue job
        print("\n[ENQUEUE] Submitting test job...")
        job = queue.enqueue(test_task, 5, 3)
        print(f"✓ Job enqueued with ID: {job.id}")
        
        # Wait for completion
        print("\n[WAIT] Waiting for job to complete...")
        timeout = 5
        start_time = time.time()
        while not job.is_finished() and not job.is_failed():
            if time.time() - start_time > timeout:
                print(f"❌ Job timed out after {timeout} seconds")
                return False
            time.sleep(0.1)
        
        # Check result
        if job.is_finished():
            print(f"✓ Job completed successfully")
            print(f"  Result: {job.result}")
            if job.result == 8:
                print("✓ Result is correct (5 + 3 = 8)")
            else:
                print(f"❌ Result is incorrect: expected 8, got {job.result}")
                return False
        else:
            print(f"❌ Job failed: {job._exc_info}")
            return False
        
        # Test multiple jobs
        print("\n[MULTIPLE JOBS] Submitting 10 concurrent jobs...")
        jobs = []
        for i in range(10):
            job = queue.enqueue(test_task, i, i*2)
            jobs.append(job)
        
        print(f"✓ Enqueued {len(jobs)} jobs")
        
        # Wait for all to complete
        print("\n[WAIT ALL] Waiting for all jobs to complete...")
        timeout = 10
        start_time = time.time()
        while True:
            if time.time() - start_time > timeout:
                print(f"❌ Jobs timed out after {timeout} seconds")
                return False
            
            finished = sum(1 for j in jobs if j.is_finished() or j.is_failed())
            if finished == len(jobs):
                break
            time.sleep(0.1)
        
        # Check results
        successful = sum(1 for j in jobs if j.is_finished())
        failed = sum(1 for j in jobs if j.is_failed())
        print(f"✓ All jobs completed: {successful} successful, {failed} failed")
        
        if failed > 0:
            print(f"❌ Some jobs failed")
            return False
        
        # Cleanup
        queue.shutdown()
        print("\n✓ Queue shut down cleanly")
        
        print("\n✅ In-process queue passed all tests")
        return True
        
    except Exception as e:
        print(f"\n❌ In-process queue test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mode_switching():
    """Test switching between modes."""
    print("\n" + "="*60)
    print("TEST 4: Mode Switching")
    print("="*60)
    
    try:
        # Test standalone mode
        print("\n[STANDALONE] Setting up standalone mode...")
        os.environ['DEPLOYMENT_MODE'] = 'standalone'
        os.environ['SQLITE_DATABASE_PATH'] = ':memory:'
        
        # Reload config
        import config
        import importlib
        importlib.reload(config)
        
        if config.DEPLOYMENT_MODE == 'standalone':
            print("✓ Standalone mode configured")
        else:
            print(f"❌ Mode not set correctly: {config.DEPLOYMENT_MODE}")
            return False
        
        # Test docker mode
        print("\n[DOCKER] Setting up docker mode...")
        os.environ['DEPLOYMENT_MODE'] = 'docker'
        
        # Reload config
        importlib.reload(config)
        
        if config.DEPLOYMENT_MODE == 'docker':
            print("✓ Docker mode configured")
        else:
            print(f"❌ Mode not set correctly: {config.DEPLOYMENT_MODE}")
            return False
        
        print("\n✅ Mode switching test passed")
        return True
        
    except Exception as e:
        print(f"\n❌ Mode switching test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("AudioMuse-AI Dual-Mode Architecture Test Suite")
    print("="*60)
    
    # Store original working directory
    original_dir = os.getcwd()
    
    # Change to project root if needed
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    os.chdir(project_root)
    
    # Add project root to Python path
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    # Run tests
    results = {
        'imports': test_imports(),
        'sqlite_adapter': test_sqlite_adapter(),
        'inprocess_queue': test_inprocess_queue(),
        'mode_switching': test_mode_switching(),
    }
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    total = len(results)
    passed = sum(1 for r in results.values() if r)
    failed = total - passed
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20s}: {status}")
    
    print("\n" + "-"*60)
    print(f"Total: {total} tests, {passed} passed, {failed} failed")
    print("-"*60)
    
    # Restore original directory
    os.chdir(original_dir)
    
    if failed == 0:
        print("\n🎉 All tests passed! The abstraction layer is working correctly.")
        return 0
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please review the errors above.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
