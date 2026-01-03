# reset.py
import os
from app.services.vector_store import delete_all_vectors
from app.services.kg_builder import clear_kg

print("⚠️  STARTING SYSTEM RESET ⚠️")

print("1. Wiping Pinecone Vectors...")
try:
    delete_all_vectors()
    print("   ✅ Pinecone Cleared.")
except Exception as e:
    print(f"   ❌ Pinecone Error: {e}")

print("2. Wiping Neo4j Knowledge Graph...")
try:
    clear_kg()
    print("   ✅ Neo4j Cleared.")
except Exception as e:
    print(f"   ❌ Neo4j Error: {e}")

print("3. Deleting Local Database...")
if os.path.exists("data/chat_history.db"):
    os.remove("data/chat_history.db")
    print("   ✅ SQLite DB Deleted.")
else:
    print("   ℹ️ No SQLite DB found.")

print("\n🎉 SYSTEM IS CLEAN. PLEASE RESTART SERVER AND RE-UPLOAD FILES.")