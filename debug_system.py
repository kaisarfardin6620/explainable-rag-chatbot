import time
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.services.vector_store import index, pc, INDEX_NAME
from app.database.repository import get_all_documents
from app.database.connection import get_chunk_texts
from app.services.embedding_service import get_embeddings

print(f"\n🔍 DEBUGGING SYSTEM for Index: {INDEX_NAME}")

print("\n[1] Checking Local Database...")
try:
    docs = get_all_documents()
    print(f"   -> Found {len(docs)} documents in SQLite.")
    for d in docs:
        print(f"      - ID: {d['id']} | File: {d['filename']} | Status: {d['status']}")
except Exception as e:
    print(f"   ❌ SQLite Error: {e}")
    docs = []

print("\n[2] Checking Pinecone Stats...")
try:
    stats = index.describe_index_stats()
    count = stats['total_vector_count']
    print(f"   -> Total Vectors in Index: {count}")
    if count == 0:
        print("   ❌ CRITICAL: Pinecone is EMPTY! Upload failed to reach Pinecone.")
    else:
        print("   ✅ Pinecone has data.")
except Exception as e:
    print(f"   ❌ Pinecone Error: {e}")

if len(docs) > 0:
    print("\n[3] Testing Manual Search...")
    test_query = "DO-RAG"
    print(f"   -> Querying for: '{test_query}'")
    
    try:
        vec = get_embeddings([test_query])[0].tolist()
        
        results = index.query(
            vector=vec,
            top_k=3,
            include_metadata=True
        )
        
        print(f"   -> Found {len(results['matches'])} matches (Raw Pinecone):")
        for m in results['matches']:
            print(f"      - Score: {m['score']:.4f} | ID: {m['id']}")
            
            text_check = get_chunk_texts([m['id']])
            if not text_check:
                print(f"        ❌ SQLite MISSING TEXT for this ID (Ghost Data)")
            else:
                print(f"        ✅ Text Found in SQLite")
    except Exception as e:
        print(f"   ❌ Search Error: {e}")

print("\n--- End Debug ---")