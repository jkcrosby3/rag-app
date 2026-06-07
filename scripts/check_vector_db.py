"""Check if vector database has documents."""
import faiss
import os

vector_db_path = "data/vector_db/faiss.index"

if os.path.exists(vector_db_path):
    print(f"Vector database found at: {vector_db_path}")
    
    # Load the index
    index = faiss.read_index(vector_db_path)
    
    print(f"Number of vectors in database: {index.ntotal}")
    print(f"Vector dimension: {index.d}")
else:
    print(f"Vector database NOT found at: {vector_db_path}")
