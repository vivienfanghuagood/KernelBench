import sqlite3
import json
import uuid
from datetime import datetime
from typing import Optional, Dict, Any
from enum import Enum
import threading

class GenerationStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class DatabaseManager:
    def __init__(self, db_path: str = "kernelbench_api.db"):
        self.db_path = db_path
        self._local = threading.local()
        self.init_database()
    
    def get_connection(self):
        """Get thread-local database connection"""
        if not hasattr(self._local, 'connection'):
            self._local.connection = sqlite3.connect(self.db_path, check_same_thread=False)
            self._local.connection.row_factory = sqlite3.Row
        return self._local.connection
    
    def init_database(self):
        """Initialize the database with required tables"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS generation_requests (
                id TEXT PRIMARY KEY,
                ref_arch_src TEXT NOT NULL,
                gpu_arch TEXT NOT NULL,
                backend TEXT NOT NULL,
                model_name TEXT NOT NULL,
                server_type TEXT NOT NULL,
                max_tokens INTEGER DEFAULT 4096,
                temperature REAL DEFAULT 0.0,
                status TEXT NOT NULL DEFAULT 'pending',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                started_at TIMESTAMP,
                completed_at TIMESTAMP,
                generated_kernel TEXT,
                error_message TEXT,
                eval_result TEXT
            )
        """)
        
        conn.commit()
    
    def create_generation_request(self, 
                                ref_arch_src: str,
                                gpu_arch: list,
                                backend: str,
                                model_name: str,
                                server_type: str,
                                max_tokens: int = 4096,
                                temperature: float = 0.0) -> str:
        """Create a new generation request and return the request ID"""
        request_id = str(uuid.uuid4())
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO generation_requests 
            (id, ref_arch_src, gpu_arch, backend, model_name, server_type, max_tokens, temperature, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (request_id, ref_arch_src, json.dumps(gpu_arch), backend, model_name, 
              server_type, max_tokens, temperature, GenerationStatus.PENDING.value))
        
        conn.commit()
        return request_id
    
    def get_request(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Get a generation request by ID"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM generation_requests WHERE id = ?
        """, (request_id,))
        
        row = cursor.fetchone()
        if row:
            result = dict(row)
            result['gpu_arch'] = json.loads(result['gpu_arch'])
            return result
        return None
    
    def update_request_status(self, request_id: str, status: GenerationStatus, **kwargs):
        """Update the status of a generation request"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        update_fields = ["status = ?"]
        params = [status.value]
        
        if status == GenerationStatus.PROCESSING:
            update_fields.append("started_at = CURRENT_TIMESTAMP")
        elif status in [GenerationStatus.COMPLETED, GenerationStatus.FAILED]:
            update_fields.append("completed_at = CURRENT_TIMESTAMP")
        
        for key, value in kwargs.items():
            if key in ['generated_kernel', 'error_message', 'eval_result']:
                update_fields.append(f"{key} = ?")
                params.append(value)
        
        params.append(request_id)
        
        cursor.execute(f"""
            UPDATE generation_requests 
            SET {', '.join(update_fields)}
            WHERE id = ?
        """, params)
        
        conn.commit()
    
    def get_all_requests(self, limit: int = 100) -> list:
        """Get all generation requests with pagination"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM generation_requests 
            ORDER BY created_at DESC 
            LIMIT ?
        """, (limit,))
        
        rows = cursor.fetchall()
        results = []
        for row in rows:
            result = dict(row)
            result['gpu_arch'] = json.loads(result['gpu_arch'])
            results.append(result)
        
        return results
    
    def close(self):
        """Close database connections"""
        if hasattr(self._local, 'connection'):
            self._local.connection.close()

# Global database instance
db = DatabaseManager()