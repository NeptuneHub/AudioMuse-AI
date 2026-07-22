import sqlite3
from uuid import UUID, uuid4
from typing import Optional, Tuple

# Mock helper function/setup for demonstration purposes, as the original context was missing.
def setup_database(conn: sqlite3.Connection):
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tracks (
            id TEXT PRIMARY KEY,
            name TEXT
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS orphans (
            source_track_id TEXT,
            orphan_id TEXT
        )
    """)
    conn.commit()

def identify_true_orphans(test_uuid: UUID, conn: sqlite3.Connection) -> Optional[list]:
    """
    Identifies true orphaned tracks by checking the 'orphans' table 
    for records linked to the test_uuid that do not exist in the main tracks table.
    
    NOTE: The fix applied here converts the UUID object (test_uuid) to a string 
    before passing it as a bound parameter to SQLite, resolving the ProgrammingError.
    """
    cursor = conn.cursor()
    
    # We assume test_uuid is the UUID that needs to be checked against the database.
    track_id_str = str(test_uuid) 

    query = """
        SELECT o.orphan_id FROM orphans o 
        WHERE o.source_track_id = ? AND NOT EXISTS (
            SELECT 1 FROM tracks t WHERE t.id = o.orphan_id
        );
    """
    try:
        # FIX: Convert UUID object to string when executing the query, 
        # as sqlite3 cannot bind 'UUID' type directly.
        cursor.execute(query, (track_id_str,))
        results = cursor.fetchall()
        return [row[0] for row in results]

    except sqlite3.Error as e:
        print(f"Database error during identify_true_orphans: {e}")
        return None


def main():
    # Use an in-memory database connection for demonstration
    conn = sqlite3.connect(':memory:')
    setup_database(conn)

    # 1. Setup test data (assuming UUIDs are stored as text/strings in the DB)
    test_uuid = uuid4()
    mock_orphan_id_1 = str(uuid4()) # This will be marked as an orphan
    mock_orphan_id_2 = str(uuid4()) # This exists and should not be reported

    # Insert test records into 'orphans' table
    cursor = conn.cursor()
    cursor.execute("INSERT INTO orphans (source_track_id, orphan_id) VALUES (?, ?)", 
                   (str(test_uuid), mock_orphan_id_1)) # Orphan record
    cursor.execute("INSERT INTO orphans (source_track_id, orphan_id) VALUES (?, ?)", 
                   (str(test_uuid), mock_orphan_id_2)) # Non-orphan record

    # Insert one track that should exist corresponding to a second test UUID
    existing_uuid = uuid4()
    cursor.execute("INSERT INTO tracks (id, name) VALUES (?, ?)", 
                   ((str(mock_orphan_id_2), "Existing Mock Track")))

    conn.commit()

    print("-" * 30)
    print(f"Attempting to identify true orphans for UUID: {test_uuid}")

    # Calling the function which contained the error
    orphans = identify_true_orphans(test_uuid, conn)

    if orphans is not None and orphans:
        print("\n[SUCCESS] Identified True Orphans (IDs):")
        for orphan in orphans:
            print(f"- {orphan}")
    else:
        print("\nNo true orphans found or an error occurred.")

    conn.close()


if __name__ == "__main__":
    main()