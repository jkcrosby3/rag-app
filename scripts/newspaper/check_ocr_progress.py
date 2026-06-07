"""
Check OCR Improvement Progress

Quick script to check how many newspapers have been processed.

Usage:
    python scripts/newspaper/check_ocr_progress.py
"""

from pathlib import Path
from datetime import datetime

def main():
    base_dir = Path(__file__).parent.parent.parent
    text_dir = base_dir / "data" / "documents" / "smithsonian" / "newspapers" / "text_files"
    
    # Count files
    total_metadata = len(list(text_dir.glob("*_metadata.json")))
    improved_files = list(text_dir.glob("*_improved.txt"))
    completed = len(improved_files)
    
    print(f"\n{'=' * 60}")
    print("📊 OCR IMPROVEMENT PROGRESS")
    print(f"{'=' * 60}\n")
    print(f"Total newspapers: {total_metadata}")
    print(f"Completed: {completed}/{total_metadata}")
    print(f"Remaining: {total_metadata - completed}")
    print(f"Progress: {completed/total_metadata*100:.1f}%")
    
    if completed < total_metadata:
        # Estimate time remaining (2 min per newspaper)
        remaining = total_metadata - completed
        est_minutes = remaining * 2
        print(f"\nEstimated time remaining: ~{est_minutes} minutes ({est_minutes/60:.1f} hours)")
    else:
        print("\n✅ All newspapers processed!")
        print("\nNext step: Run re-enrichment")
        print("  python scripts/newspaper/re_enrich_with_improved_ocr.py")
    
    # Show most recent files
    if improved_files:
        print(f"\n📄 Most recent 5 completed:")
        recent = sorted(improved_files, key=lambda x: x.stat().st_mtime, reverse=True)[:5]
        for f in recent:
            mtime = datetime.fromtimestamp(f.stat().st_mtime)
            print(f"  - {f.stem.replace('_improved', '')} ({mtime.strftime('%H:%M:%S')})")
    
    print()

if __name__ == "__main__":
    main()
