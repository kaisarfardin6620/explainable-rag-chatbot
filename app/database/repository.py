import sqlite3
from typing import List, Dict, Any, Optional
from app.database.connection import (
    get_connection, 
    save_chunks, 
    get_chunk_texts
)

def get_or_create_document_id(filename: str) -> int:
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("SELECT id FROM documents WHERE filename = ?", (filename,))
        row = cursor.fetchone()
        
        if row:
            return row[0]
            
        cursor.execute("INSERT INTO documents (filename, status) VALUES (?, ?)", (filename, 'processed'))
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()

def get_all_documents() -> List[Dict[str, Any]]:
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT id, filename, status, upload_time FROM documents ORDER BY upload_time DESC")
        rows = cursor.fetchall()
        return [
            {
                "id": row[0],
                "filename": row[1],
                "status": row[2],
                "upload_time": str(row[3])
            }
            for row in rows
        ]
    finally:
        conn.close()

def save_chat_message(session_id: int, role: str, content: str):
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT OR IGNORE INTO chat_sessions (id) VALUES (?)", (session_id,))
        
        cursor.execute(
            "INSERT INTO chat_messages (session_id, role, content) VALUES (?, ?, ?)",
            (session_id, role, content)
        )
        conn.commit()
    finally:
        conn.close()

def get_session_history(session_id: int) -> List[Dict[str, Any]]:
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            SELECT role, content, timestamp 
            FROM chat_messages 
            WHERE session_id = ? 
            ORDER BY timestamp ASC
        """, (session_id,))
        
        rows = cursor.fetchall()
        return [
            {
                "role": row[0],
                "content": row[1],
                "timestamp": str(row[2])
            }
            for row in rows
        ]
    finally:
        conn.close()