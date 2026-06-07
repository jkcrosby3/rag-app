#!/usr/bin/env python3
"""
RAG Application Data Cleanup Script

Purpose: Safely remove old documents, embeddings, and processing artifacts
         to prepare the system for new data.

Usage:
    python cleanup_data.py                    # Interactive mode with prompts
    python cleanup_data.py --confirm          # Skip confirmation prompts
    python cleanup_data.py --dry-run          # Preview what would be deleted
    python cleanup_data.py --keep-documents   # Keep source documents, delete embeddings
    python cleanup_data.py --vector-db-only   # Only delete vector database
    python cleanup_data.py --backup           # Create backup before deletion

Author: Generated for Smithsonian Hackathon
Date: May 7, 2026
"""

import os
import shutil
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import List, Tuple


class DataCleaner:
    """Handles cleanup of RAG application data directories."""
    
    def __init__(self, base_dir: str = None, dry_run: bool = False):
        """
        Initialize the data cleaner.
        
        Args:
            base_dir: Base directory of the RAG application (default: current directory)
            dry_run: If True, only show what would be deleted without deleting
        """
        self.base_dir = Path(base_dir) if base_dir else Path(__file__).parent
        self.data_dir = self.base_dir / "data"
        self.dry_run = dry_run
        self.deleted_items: List[str] = []
        self.deleted_size: int = 0
        
    def get_dir_size(self, path: Path) -> int:
        """Calculate total size of directory in bytes."""
        total = 0
        try:
            for entry in path.rglob('*'):
                if entry.is_file():
                    total += entry.stat().st_size
        except Exception as e:
            print(f"Warning: Could not calculate size for {path}: {e}")
        return total
    
    def format_size(self, size_bytes: int) -> str:
        """Format bytes as human-readable string."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.2f} TB"
    
    def clean_directory(self, dir_path: Path, description: str) -> bool:
        """
        Clean contents of a directory while preserving the directory itself.
        
        Args:
            dir_path: Path to directory to clean
            description: Human-readable description for logging
            
        Returns:
            True if successful, False otherwise
        """
        if not dir_path.exists():
            print(f"  ⚠️  {description}: Directory does not exist, skipping")
            return True
        
        try:
            size = self.get_dir_size(dir_path)
            item_count = sum(1 for _ in dir_path.rglob('*') if _.is_file())
            
            if item_count == 0:
                print(f"  ✓  {description}: Already empty")
                return True
            
            print(f"  📁 {description}: {item_count} items ({self.format_size(size)})")
            
            if self.dry_run:
                print(f"     [DRY RUN] Would delete contents of {dir_path}")
                self.deleted_items.append(f"{description} ({item_count} items)")
                self.deleted_size += size
                return True
            
            for item in dir_path.iterdir():
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)
            
            self.deleted_items.append(f"{description} ({item_count} items)")
            self.deleted_size += size
            print(f"  ✓  {description}: Cleaned successfully")
            return True
            
        except Exception as e:
            print(f"  ❌ {description}: Error - {e}")
            return False
    
    def clean_file(self, file_path: Path, description: str) -> bool:
        """
        Delete a specific file.
        
        Args:
            file_path: Path to file to delete
            description: Human-readable description for logging
            
        Returns:
            True if successful, False otherwise
        """
        if not file_path.exists():
            print(f"  ⚠️  {description}: File does not exist, skipping")
            return True
        
        try:
            size = file_path.stat().st_size
            print(f"  📄 {description}: {self.format_size(size)}")
            
            if self.dry_run:
                print(f"     [DRY RUN] Would delete {file_path}")
                self.deleted_items.append(description)
                self.deleted_size += size
                return True
            
            file_path.unlink()
            self.deleted_items.append(description)
            self.deleted_size += size
            print(f"  ✓  {description}: Deleted successfully")
            return True
            
        except Exception as e:
            print(f"  ❌ {description}: Error - {e}")
            return False
    
    def create_backup(self, backup_dir: Path = None) -> bool:
        """
        Create backup of data directory before cleanup.
        
        Args:
            backup_dir: Directory to store backup (default: data_backup_TIMESTAMP)
            
        Returns:
            True if successful, False otherwise
        """
        if not backup_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = self.base_dir / f"data_backup_{timestamp}"
        
        print(f"\n📦 Creating backup at: {backup_dir}")
        
        try:
            if self.dry_run:
                print(f"   [DRY RUN] Would create backup at {backup_dir}")
                return True
            
            shutil.copytree(self.data_dir, backup_dir)
            backup_size = self.get_dir_size(backup_dir)
            print(f"✓  Backup created successfully ({self.format_size(backup_size)})")
            return True
            
        except Exception as e:
            print(f"❌ Backup failed: {e}")
            return False
    
    def full_cleanup(self) -> bool:
        """Perform full cleanup of all data."""
        print("\n🧹 Starting FULL cleanup...")
        print("=" * 60)
        
        success = True
        success &= self.clean_directory(self.data_dir / "documents", "Source Documents")
        success &= self.clean_directory(self.data_dir / "vector_db", "Vector Database")
        success &= self.clean_directory(self.data_dir / "chunked", "Chunked Documents")
        success &= self.clean_directory(self.data_dir / "embedded", "Embedded Documents")
        success &= self.clean_directory(self.data_dir / "metadata", "Metadata")
        success &= self.clean_directory(self.data_dir / "cache", "Cache")
        success &= self.clean_file(self.data_dir / "document_registry.json", "Document Registry")
        
        return success
    
    def embeddings_only_cleanup(self) -> bool:
        """Clean embeddings and processing artifacts, keep source documents."""
        print("\n🧹 Starting EMBEDDINGS cleanup (keeping source documents)...")
        print("=" * 60)
        
        success = True
        success &= self.clean_directory(self.data_dir / "vector_db", "Vector Database")
        success &= self.clean_directory(self.data_dir / "chunked", "Chunked Documents")
        success &= self.clean_directory(self.data_dir / "embedded", "Embedded Documents")
        success &= self.clean_directory(self.data_dir / "metadata", "Metadata")
        success &= self.clean_directory(self.data_dir / "cache", "Cache")
        success &= self.clean_file(self.data_dir / "document_registry.json", "Document Registry")
        
        return success
    
    def vector_db_only_cleanup(self) -> bool:
        """Clean only the vector database."""
        print("\n🧹 Starting VECTOR DATABASE cleanup...")
        print("=" * 60)
        
        success = True
        success &= self.clean_directory(self.data_dir / "vector_db", "Vector Database")
        success &= self.clean_file(self.data_dir / "document_registry.json", "Document Registry")
        
        return success
    
    def artifacts_only_cleanup(self) -> bool:
        """Clean only processing artifacts (cache, metadata)."""
        print("\n🧹 Starting ARTIFACTS cleanup...")
        print("=" * 60)
        
        success = True
        success &= self.clean_directory(self.data_dir / "metadata", "Metadata")
        success &= self.clean_directory(self.data_dir / "cache", "Cache")
        
        return success
    
    def print_summary(self):
        """Print summary of cleanup operation."""
        print("\n" + "=" * 60)
        print("📊 CLEANUP SUMMARY")
        print("=" * 60)
        
        if self.dry_run:
            print("🔍 DRY RUN MODE - No files were actually deleted")
            print()
        
        if self.deleted_items:
            print(f"✓  Items cleaned: {len(self.deleted_items)}")
            print(f"✓  Space freed: {self.format_size(self.deleted_size)}")
            print("\nDeleted items:")
            for item in self.deleted_items:
                print(f"  • {item}")
        else:
            print("ℹ️  No items were deleted (directories may already be empty)")
        
        print("\n" + "=" * 60)


def confirm_action(message: str) -> bool:
    """Ask user for confirmation."""
    response = input(f"\n{message} (yes/no): ").strip().lower()
    return response in ['yes', 'y']


def main():
    parser = argparse.ArgumentParser(
        description="Clean RAG application data directories",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cleanup_data.py                    # Interactive mode
  python cleanup_data.py --confirm          # Skip confirmation
  python cleanup_data.py --dry-run          # Preview only
  python cleanup_data.py --keep-documents   # Keep documents, delete embeddings
  python cleanup_data.py --backup           # Create backup first
        """
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be deleted without actually deleting'
    )
    
    parser.add_argument(
        '--confirm',
        action='store_true',
        help='Skip confirmation prompts (use with caution)'
    )
    
    parser.add_argument(
        '--keep-documents',
        action='store_true',
        help='Keep source documents, only delete embeddings and artifacts'
    )
    
    parser.add_argument(
        '--vector-db-only',
        action='store_true',
        help='Only delete vector database'
    )
    
    parser.add_argument(
        '--artifacts-only',
        action='store_true',
        help='Only delete processing artifacts (cache, metadata)'
    )
    
    parser.add_argument(
        '--backup',
        action='store_true',
        help='Create backup before cleanup'
    )
    
    parser.add_argument(
        '--base-dir',
        type=str,
        help='Base directory of RAG application (default: current directory)'
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("🧹 RAG APPLICATION DATA CLEANUP")
    print("=" * 60)
    
    cleaner = DataCleaner(base_dir=args.base_dir, dry_run=args.dry_run)
    
    if not cleaner.data_dir.exists():
        print(f"\n❌ Error: Data directory not found at {cleaner.data_dir}")
        print("   Make sure you're running this script from the rag-app directory")
        return 1
    
    if args.dry_run:
        print("\n🔍 DRY RUN MODE - No files will be deleted")
    
    if args.backup and not args.dry_run:
        if not cleaner.create_backup():
            if not confirm_action("Backup failed. Continue anyway?"):
                print("\n❌ Cleanup cancelled")
                return 1
    
    if not args.confirm and not args.dry_run:
        print("\n⚠️  WARNING: This will permanently delete data!")
        print(f"   Data directory: {cleaner.data_dir}")
        
        if args.keep_documents:
            print("   Mode: Keep documents, delete embeddings")
        elif args.vector_db_only:
            print("   Mode: Delete vector database only")
        elif args.artifacts_only:
            print("   Mode: Delete artifacts only")
        else:
            print("   Mode: FULL cleanup (all data)")
        
        if not confirm_action("Are you sure you want to proceed?"):
            print("\n❌ Cleanup cancelled")
            return 0
    
    if args.vector_db_only:
        success = cleaner.vector_db_only_cleanup()
    elif args.artifacts_only:
        success = cleaner.artifacts_only_cleanup()
    elif args.keep_documents:
        success = cleaner.embeddings_only_cleanup()
    else:
        success = cleaner.full_cleanup()
    
    cleaner.print_summary()
    
    if not args.dry_run and success:
        print("\n✅ Cleanup completed successfully!")
        print("\nNext steps:")
        print("  1. Add your new documents to data/documents/")
        print("  2. Run: ./start_web_app.bat (Windows) or ./start_web_app.sh (Linux/Mac)")
        print("  3. The system will automatically rebuild the vector database")
    elif args.dry_run:
        print("\n✅ Dry run completed!")
        print("   Run without --dry-run to actually delete files")
    else:
        print("\n⚠️  Cleanup completed with some errors")
        print("   Check the output above for details")
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
