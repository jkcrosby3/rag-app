"""
Database statistics utility for the RAG system.

This module provides utilities to analyze and extract statistics from the vector database,
including document counts by type, unique source files, and collection breakdowns.
"""
import pickle
from pathlib import Path
from typing import Dict, Any, Set
from collections import defaultdict


def get_vector_db_statistics(vector_db_path: str = "data/vector_db") -> Dict[str, Any]:
    """
    Extract comprehensive statistics from the vector database.
    
    Args:
        vector_db_path: Path to the vector database directory
        
    Returns:
        Dictionary containing statistics including:
        - total_chunks: Total number of document chunks
        - total_unique_files: Total number of unique source files
        - by_type: Breakdown by document type (newspapers, pension_files, etc.)
        - by_topic: Breakdown by topic field
    """
    vector_db_path = Path(vector_db_path)
    lookup_path = vector_db_path / "document_lookup.pkl"
    
    if not lookup_path.exists():
        raise FileNotFoundError(f"Vector database not found at {lookup_path}")
    
    # Load document lookup
    with open(lookup_path, 'rb') as f:
        doc_lookup = pickle.load(f)
    
    # Initialize counters
    by_type = defaultdict(lambda: {
        'chunks': 0,
        'unique_files': set(),
        'file_paths': set()
    })
    
    by_topic = defaultdict(lambda: {
        'chunks': 0,
        'unique_files': set()
    })
    
    unique_files_global = set()
    
    # Process each document
    for doc_id, doc_data in doc_lookup.items():
        metadata = doc_data.get('metadata', {})
        
        # Get identifiers
        rel_path = metadata.get('relative_path', '').lower()
        file_name = metadata.get('file_name', '')
        file_path = metadata.get('file_path', '')
        topic = metadata.get('topic', 'unknown')
        
        # Determine document type from relative path
        doc_type = 'other'
        if 'newspaper' in rel_path:
            doc_type = 'newspapers'
        elif 'pension' in rel_path:
            doc_type = 'pension_files'
        elif 'collection' in rel_path or 'revolutionary_era' in rel_path:
            doc_type = 'collections'
        elif 'book' in rel_path or topic == 'books':
            doc_type = 'books'
        
        # Update type-based counters
        by_type[doc_type]['chunks'] += 1
        if file_name:
            by_type[doc_type]['unique_files'].add(file_name)
        if file_path:
            by_type[doc_type]['file_paths'].add(file_path)
            unique_files_global.add(file_path)
        
        # Update topic-based counters
        by_topic[topic]['chunks'] += 1
        if file_name:
            by_topic[topic]['unique_files'].add(file_name)
    
    # Convert sets to counts for JSON serialization
    by_type_serializable = {}
    for doc_type, data in by_type.items():
        by_type_serializable[doc_type] = {
            'chunks': data['chunks'],
            'unique_files': len(data['unique_files']),
            'avg_chunks_per_doc': data['chunks'] / len(data['unique_files']) if data['unique_files'] else 0
        }
    
    by_topic_serializable = {}
    for topic, data in by_topic.items():
        by_topic_serializable[topic] = {
            'chunks': data['chunks'],
            'unique_files': len(data['unique_files']),
            'avg_chunks_per_doc': data['chunks'] / len(data['unique_files']) if data['unique_files'] else 0
        }
    
    return {
        'total_chunks': len(doc_lookup),
        'total_unique_files': len(unique_files_global),
        'by_type': by_type_serializable,
        'by_topic': by_topic_serializable
    }


def format_statistics_summary(stats: Dict[str, Any]) -> str:
    """
    Format statistics into a human-readable summary.
    
    Args:
        stats: Statistics dictionary from get_vector_db_statistics()
        
    Returns:
        Formatted string summary
    """
    lines = []
    lines.append("="*60)
    lines.append("VECTOR DATABASE STATISTICS")
    lines.append("="*60)
    lines.append(f"Total Chunks:        {stats['total_chunks']:>6,}")
    lines.append(f"Total Source Files:  {stats['total_unique_files']:>6,}")
    lines.append("")
    
    lines.append("By Document Type:")
    lines.append("-" * 60)
    for doc_type, data in sorted(stats['by_type'].items()):
        lines.append(f"  {doc_type.title():20s} {data['unique_files']:>6,} files  {data['chunks']:>6,} chunks  ({data['avg_chunks_per_doc']:.1f} avg)")
    
    return "\n".join(lines)


if __name__ == "__main__":
    # Test the statistics generator
    try:
        stats = get_vector_db_statistics()
        print(format_statistics_summary(stats))
        
        # Print detailed breakdown
        print("\n" + "="*60)
        print("DETAILED BREAKDOWN")
        print("="*60)
        print("\nBy Document Type:")
        for doc_type, data in sorted(stats['by_type'].items()):
            print(f"\n{doc_type.upper()}:")
            print(f"  Files: {data['unique_files']:,}")
            print(f"  Chunks: {data['chunks']:,}")
            print(f"  Avg Chunks/File: {data['avg_chunks_per_doc']:.2f}")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
