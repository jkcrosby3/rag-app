"""
Storage Analysis for RAG Application

Analyzes disk usage and recommends what to keep vs delete.

Usage:
    python scripts/analyze_storage.py
"""

from pathlib import Path
from collections import defaultdict

def get_size(path):
    """Get total size of directory in bytes."""
    total = 0
    try:
        for f in Path(path).rglob('*'):
            if f.is_file():
                total += f.stat().st_size
    except:
        pass
    return total

def format_size(bytes):
    """Format bytes to human readable."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes < 1024:
            return f"{bytes:.1f} {unit}"
        bytes /= 1024
    return f"{bytes:.1f} TB"

def main():
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "data" / "documents" / "smithsonian"
    
    print(f"\n{'=' * 70}")
    print("💾 STORAGE ANALYSIS")
    print(f"{'=' * 70}\n")
    
    # Analyze each component
    components = {
        "Newspaper text files": data_dir / "newspapers" / "text_files",
        "Newspaper metadata": data_dir / "newspapers" / "text_files",
        "Pension text files": data_dir / "pension_files" / "text_files",
        "Pension metadata": data_dir / "pension_files" / "metadata",
    }
    
    total = 0
    details = {}
    
    for name, path in components.items():
        if path.exists():
            size = get_size(path)
            total += size
            details[name] = size
            
            # Count files
            if "metadata" in name.lower():
                files = list(path.glob("*_metadata.json"))
            elif "text" in name.lower():
                files = list(path.glob("*.txt"))
            else:
                files = list(path.rglob("*"))
            
            print(f"📁 {name}")
            print(f"   Size: {format_size(size)}")
            print(f"   Files: {len(files)}")
            print()
    
    # Check for improved OCR files
    improved_files = list((data_dir / "newspapers" / "text_files").glob("*_improved.txt"))
    if improved_files:
        improved_size = sum(f.stat().st_size for f in improved_files)
        print(f"📁 Improved OCR files (can delete after re-enrichment)")
        print(f"   Size: {format_size(improved_size)}")
        print(f"   Files: {len(improved_files)}")
        print()
    
    # Check for vector store
    vector_dir = base_dir / "vector_store"
    if vector_dir.exists():
        vector_size = get_size(vector_dir)
        print(f"📁 Vector Database (MUST KEEP)")
        print(f"   Size: {format_size(vector_size)}")
        print()
        total += vector_size
    
    print(f"{'=' * 70}")
    print(f"📊 TOTAL STORAGE: {format_size(total)}")
    print(f"{'=' * 70}\n")
    
    # Recommendations
    print("💡 RECOMMENDATIONS:\n")
    print("✅ MUST KEEP:")
    print("   - Vector database files (vector_store/)")
    print("   - Metadata JSON files (*_metadata.json)")
    print("   - Original text files (*.txt) - small and useful")
    print()
    print("❌ CAN DELETE AFTER VECTOR DB CREATED:")
    print("   - Improved OCR files (*_improved.txt)")
    print("   - Downloaded images (*.jp2, *.jpg)")
    print("   - Temporary/cache files")
    print()
    print("⚠️  SPACE SAVINGS:")
    if improved_files:
        print(f"   Delete improved OCR: ~{format_size(improved_size)} saved")
    print("   Delete images: Varies (if you downloaded any)")
    print()

if __name__ == "__main__":
    main()
