#!/usr/bin/env python3
"""
Script to clean up kernelbench_api.db by removing:
1. Records with status=pending
2. Records with incorrect results (correctness=False)
3. Records with speedup < 0.3
"""

import sqlite3
import re
import sys
import os

def parse_eval_result(eval_str: str) -> dict:
    """Parse eval_result string to extract correctness and speedup"""
    result = {
        "correctness": None,
        "speedup": None
    }
    
    if not eval_str:
        return result
    
    # Extract correctness
    correctness_match = re.search(r'correctness=(True|False)', eval_str)
    if correctness_match:
        result["correctness"] = correctness_match.group(1) == "True"
    
    # Extract speedup
    speedup_match = re.search(r'speedup=([\d.]+)', eval_str)
    if speedup_match:
        result["speedup"] = float(speedup_match.group(1))
    
    return result

def cleanup_database(db_path: str, dry_run: bool = False):
    """
    Clean up database by removing incorrect or low speedup records
    
    Args:
        db_path: Path to the SQLite database
        dry_run: If True, only show what would be deleted without actually deleting
    """
    if not os.path.exists(db_path):
        print(f"Error: Database file not found: {db_path}")
        sys.exit(1)
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Get all records
    cursor.execute("SELECT id, problem_name, eval_result, status FROM generation_requests")
    rows = cursor.fetchall()
    
    print(f"Total records in database: {len(rows)}")
    print("-" * 60)
    
    ids_to_delete = []
    
    for row in rows:
        row_id = row['id']
        problem_name = row['problem_name'] or 'N/A'
        eval_result = row['eval_result']
        status = row['status']
        
        parsed = parse_eval_result(eval_result)
        correctness = parsed['correctness']
        speedup = parsed['speedup']
        
        should_delete = False
        reason = []
        
        # Check if status is pending
        if status == 'pending':
            should_delete = True
            reason.append("status=pending")
        
        # Check if incorrect
        if correctness is False:
            should_delete = True
            reason.append("incorrect")
        
        # Check if speedup < 0.3
        if speedup is not None and speedup < 0.3:
            should_delete = True
            reason.append(f"speedup={speedup:.4f} < 0.3")
        
        if should_delete:
            ids_to_delete.append(row_id)
            print(f"[DELETE] ID: {row_id[:8]}... | Problem: {problem_name} | Reason: {', '.join(reason)}")
    
    print("-" * 60)
    print(f"Records to delete: {len(ids_to_delete)}")
    print(f"Records to keep: {len(rows) - len(ids_to_delete)}")
    
    if ids_to_delete:
        if dry_run:
            print("\n[DRY RUN] No records were actually deleted.")
        else:
            # Confirm deletion
            print("\nProceeding with deletion...")
            
            # Delete records
            for row_id in ids_to_delete:
                cursor.execute("DELETE FROM generation_requests WHERE id = ?", (row_id,))
            
            conn.commit()
            print(f"Successfully deleted {len(ids_to_delete)} records.")
    else:
        print("\nNo records match the deletion criteria.")
    
    conn.close()

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Clean up kernelbench_api.db')
    parser.add_argument('--db', default='kernelbench_api.db', 
                        help='Path to database file (default: kernelbench_api.db)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Only show what would be deleted without actually deleting')
    
    args = parser.parse_args()
    
    print(f"Database: {args.db}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'ACTUAL DELETE'}")
    print("=" * 60)
    
    cleanup_database(args.db, dry_run=args.dry_run)

if __name__ == '__main__':
    main()

