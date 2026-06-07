"""Query vector database metadata for statistics and counts."""
import pickle
from pathlib import Path
from collections import defaultdict

def main():
    """Query metadata from vector database."""
    print("Loading vector database metadata...")
    lookup = pickle.load(open('data/vector_db/document_lookup.pkl', 'rb'))
    
    print(f"\nTotal documents in vector DB: {len(lookup)}")
    print("=" * 80)
    
    # Count by topic
    by_topic = defaultdict(lambda: {'chunks': 0, 'unique_files': set()})
    
    for doc in lookup.values():
        topic = doc.get('metadata', {}).get('topic', 'unknown')
        filename = doc.get('metadata', {}).get('file_name', '')
        
        by_topic[topic]['chunks'] += 1
        if filename:
            by_topic[topic]['unique_files'].add(filename)
    
    print("\nDocuments by topic:")
    for topic in sorted(by_topic.keys()):
        info = by_topic[topic]
        print(f"\n  {topic}:")
        print(f"    Total chunks: {info['chunks']}")
        print(f"    Unique files: {len(info['unique_files'])}")
        print(f"    Avg chunks/file: {info['chunks']/len(info['unique_files']):.1f}")
    
    # For pension files, show sample soldier names
    print("\n" + "=" * 80)
    print("PENSION FILES - Sample soldier names:")
    print("=" * 80)
    
    pension_files = sorted(by_topic['pension_files']['unique_files'])[:50]
    for i, filename in enumerate(pension_files, 1):
        # Extract record ID from filename
        record_id = filename.replace('.txt', '')
        print(f"{i:3d}. Record {record_id}")
    
    print(f"\n... and {len(by_topic['pension_files']['unique_files']) - 50} more pension files")
    
    print("\n" + "=" * 80)
    print(f"ANSWER: There are {len(by_topic['pension_files']['unique_files'])} unique pension files")
    print(f"        representing approximately {len(by_topic['pension_files']['unique_files'])} soldiers/officers")
    print("=" * 80)

if __name__ == '__main__':
    main()
