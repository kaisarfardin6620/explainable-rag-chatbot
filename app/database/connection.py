import sqlite3
from pathlib import Path
from typing import List, Dict, Any
from app.core.config import settings

Path(settings.upload_folder).mkdir(parents=True, exist_ok=True)
DB_PATH = settings.sqlite_db_path

def get_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn

def init_db() -> None:
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL UNIQUE,
            upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            status TEXT DEFAULT 'processed'  -- processed, failed, etc.
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            role TEXT NOT NULL,  -- 'user' or 'assistant'
            content TEXT,
            citations TEXT,     -- JSON string of citations
            confidence REAL,
            explanation TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES chat_sessions (id) ON DELETE CASCADE
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chunk_texts (
            chunk_id TEXT PRIMARY KEY,           -- Matches Pinecone vector ID
            text TEXT NOT NULL,
            document_id INTEGER,
            page INTEGER DEFAULT 0,
            FOREIGN KEY (document_id) REFERENCES documents (id) ON DELETE CASCADE
        )
    """)

    conn.commit()
    conn.close()

def save_chunks(chunks: List[Dict[str, Any]], document_id: int) -> List[str]:
    if not chunks:
        return []

    conn = get_connection()
    cursor = conn.cursor()

    chunk_ids = []
    for chunk in chunks:
        chunk_id = str(chunk.get("chunk_id"))
        text = chunk["text"]
        page = chunk["metadata"].get("page", 0)

        cursor.execute("""
            INSERT OR REPLACE INTO chunk_texts (chunk_id, text, document_id, page)
            VALUES (?, ?, ?, ?)
        """, (chunk_id, text, document_id, page))

        chunk_ids.append(chunk_id)

    conn.commit()
    conn.close()
    return chunk_ids

def get_chunk_text(chunk_id: str) -> str | None:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT text FROM chunk_texts WHERE chunk_id = ?", (chunk_id,))
    row = cursor.fetchone()
    conn.close()
    return row[0] if row else None

def get_chunk_texts(chunk_ids: List[str]) -> Dict[str, str]:
    if not chunk_ids:
        return {}

    conn = get_connection()
    cursor = conn.cursor()
    placeholders = ','.join('?' for _ in chunk_ids)
    cursor.execute(f"SELECT chunk_id, text FROM chunk_texts WHERE chunk_id IN ({placeholders})", chunk_ids)
    results = {row[0]: row[1] for row in cursor.fetchall()}
    conn.close()
    return results

init_db()