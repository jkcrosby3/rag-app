"""Delete the vector database to start fresh."""
import os
from pathlib import Path

vector_db_file = Path("data/vector_db/faiss.index")

if vector_db_file.exists():
    os.remove(vector_db_file)
    print(f"✅ Deleted {vector_db_file}")
else:
    print(f"❌ File not found: {vector_db_file}")

print("\nNow run: python process_newspapers.py")
