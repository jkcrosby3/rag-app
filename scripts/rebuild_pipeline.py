"""
Master pipeline script for RAG document processing.

This script orchestrates the full document processing pipeline:
1. Clean intermediate directories (optional)
2. Process documents (extract text and merge metadata)
3. Chunk documents
4. Generate embeddings
5. Build vector database

Usage:
    python scripts/rebuild_pipeline.py --clean          # Full clean rebuild
    python scripts/rebuild_pipeline.py                  # Incremental update
    python scripts/rebuild_pipeline.py --clean --skip-processing  # Rebuild from processed files
"""

import argparse
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PipelineError(Exception):
    """Custom exception for pipeline errors."""
    pass


class RAGPipeline:
    """Manages the full RAG document processing pipeline."""
    
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.data_dir = base_dir / "data"
        self.scripts_dir = base_dir / "scripts"
        
        # Pipeline directories
        self.dirs = {
            'documents': self.data_dir / "documents",
            'processed': self.data_dir / "processed",
            'chunked': self.data_dir / "chunked",
            'embedded': self.data_dir / "embedded",
            'vector_db': self.data_dir / "vector_db"
        }
        
        # Residual directories that can be cleaned up
        self.residual_dirs = [
            self.data_dir / "chunks",
            self.data_dir / "staging"
        ]
        
    def clean_intermediate_files(self, keep_documents: bool = True) -> None:
        """
        Delete all intermediate files to ensure a clean rebuild.
        
        Args:
            keep_documents: If True, preserves the documents directory
        """
        logger.info("=" * 70)
        logger.info("CLEANING INTERMEDIATE FILES")
        logger.info("=" * 70)
        
        dirs_to_clean = ['processed', 'chunked', 'embedded', 'vector_db']
        if not keep_documents:
            dirs_to_clean.append('documents')
        
        for dir_name in dirs_to_clean:
            dir_path = self.dirs[dir_name]
            if dir_path.exists():
                logger.info(f"Deleting: {dir_path}")
                try:
                    # On Windows, use rmtree with error handling
                    def handle_remove_readonly(func, path, exc):
                        """Error handler for Windows readonly files."""
                        import stat
                        if not os.access(path, os.W_OK):
                            os.chmod(path, stat.S_IWUSR)
                            func(path)
                        else:
                            raise
                    
                    shutil.rmtree(dir_path, onerror=handle_remove_readonly)
                    logger.info(f"  ✓ Deleted {dir_name}/")
                except Exception as e:
                    logger.error(f"  ✗ Failed to delete {dir_name}/: {e}")
                    logger.error(f"  Please close any open files and try again")
                    raise PipelineError(f"Could not delete {dir_name}/: {e}")
            else:
                logger.info(f"  - {dir_name}/ does not exist, skipping")
        
        # Clean up residual directories
        for residual_dir in self.residual_dirs:
            if residual_dir.exists():
                logger.info(f"Deleting residual directory: {residual_dir}")
                try:
                    def handle_remove_readonly(func, path, exc):
                        import stat
                        if not os.access(path, os.W_OK):
                            os.chmod(path, stat.S_IWUSR)
                            func(path)
                        else:
                            raise
                    
                    shutil.rmtree(residual_dir, onerror=handle_remove_readonly)
                    logger.info(f"  ✓ Deleted {residual_dir.name}/")
                except Exception as e:
                    logger.warning(f"  ! Could not delete {residual_dir.name}/: {e}")
        
        logger.info("Cleanup complete!\n")
    
    def run_script(self, script_name: str, description: str, env: Dict[str, str] = None) -> None:
        """
        Run a pipeline script and handle errors.
        
        Args:
            script_name: Name of the script to run
            description: Human-readable description of the step
            env: Optional environment variables to set
        """
        logger.info("=" * 70)
        logger.info(f"{description.upper()}")
        logger.info("=" * 70)
        
        script_path = self.scripts_dir / script_name
        if not script_path.exists():
            raise PipelineError(f"Script not found: {script_path}")
        
        # Prepare environment
        run_env = os.environ.copy()
        if env:
            run_env.update(env)
        
        # Run the script
        start_time = time.time()
        try:
            result = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=str(self.base_dir),
                env=run_env,
                capture_output=False,
                text=True,
                check=True
            )
            elapsed = time.time() - start_time
            logger.info(f"✓ {description} completed in {elapsed:.1f} seconds\n")
            
        except subprocess.CalledProcessError as e:
            raise PipelineError(f"{description} failed with exit code {e.returncode}")
    
    def get_statistics(self) -> Dict[str, int]:
        """Gather statistics about the pipeline state."""
        stats = {}
        
        for name, dir_path in self.dirs.items():
            if dir_path.exists():
                # Count files (excluding directories)
                files = [f for f in dir_path.rglob('*') if f.is_file()]
                stats[name] = len(files)
            else:
                stats[name] = 0
        
        return stats
    
    def print_statistics(self) -> None:
        """Print pipeline statistics."""
        stats = self.get_statistics()
        
        logger.info("=" * 70)
        logger.info("PIPELINE STATISTICS")
        logger.info("=" * 70)
        logger.info(f"Source documents:    {stats.get('documents', 0):>6} files")
        logger.info(f"Processed documents: {stats.get('processed', 0):>6} files")
        logger.info(f"Chunked documents:   {stats.get('chunked', 0):>6} files")
        logger.info(f"Embedded documents:  {stats.get('embedded', 0):>6} files")
        logger.info(f"Vector DB files:     {stats.get('vector_db', 0):>6} files")
        
        # Read vector DB metadata if it exists
        metadata_file = self.dirs['vector_db'] / "metadata.json"
        if metadata_file.exists():
            import json
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            logger.info(f"\nVector Database:")
            logger.info(f"  Documents indexed: {metadata.get('document_count', 0):>6}")
            logger.info(f"  Embedding dimension: {metadata.get('dimension', 0):>4}")
            logger.info(f"  Index type: {metadata.get('index_type', 'Unknown')}")
        
        logger.info("=" * 70)
    
    def run_full_pipeline(self, skip_processing: bool = False) -> None:
        """
        Run the complete pipeline.
        
        Args:
            skip_processing: If True, skip document processing step
        """
        overall_start = time.time()
        
        logger.info("\n" + "=" * 70)
        logger.info("RAG PIPELINE - FULL REBUILD")
        logger.info("=" * 70)
        logger.info(f"Base directory: {self.base_dir}")
        logger.info(f"Starting at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 70 + "\n")
        
        try:
            # Step 1: Process documents (if not skipped)
            if not skip_processing:
                self.run_script(
                    "process_documents.py",
                    "Step 1/4: Processing Documents"
                )
            else:
                logger.info("=" * 70)
                logger.info("STEP 1/4: PROCESSING DOCUMENTS - SKIPPED")
                logger.info("=" * 70 + "\n")
            
            # Step 2: Chunk documents
            self.run_script(
                "chunk_documents.py",
                "Step 2/4: Chunking Documents"
            )
            
            # Step 3: Generate embeddings (requires PyTorch env variable)
            self.run_script(
                "generate_embeddings.py",
                "Step 3/4: Generating Embeddings",
                env={'KMP_DUPLICATE_LIB_OK': 'TRUE'}
            )
            
            # Step 4: Build vector database
            self.run_script(
                "build_vector_db.py",
                "Step 4/4: Building Vector Database",
                env={'KMP_DUPLICATE_LIB_OK': 'TRUE'}
            )
            
            # Print final statistics
            self.print_statistics()
            
            overall_elapsed = time.time() - overall_start
            logger.info(f"\n✓ PIPELINE COMPLETED SUCCESSFULLY")
            logger.info(f"Total time: {overall_elapsed / 60:.1f} minutes")
            logger.info("=" * 70 + "\n")
            
        except PipelineError as e:
            logger.error(f"\n✗ PIPELINE FAILED: {e}")
            logger.error("=" * 70 + "\n")
            sys.exit(1)
        except Exception as e:
            logger.error(f"\n✗ UNEXPECTED ERROR: {e}")
            logger.error("=" * 70 + "\n")
            raise


def main():
    """Main entry point for the pipeline script."""
    parser = argparse.ArgumentParser(
        description="RAG Pipeline - Process documents and build vector database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full clean rebuild (recommended)
  python scripts/rebuild_pipeline.py --clean
  
  # Incremental update (processes new/changed documents only)
  python scripts/rebuild_pipeline.py
  
  # Rebuild from already processed files
  python scripts/rebuild_pipeline.py --clean --skip-processing
  
  # Clean only (useful for testing)
  python scripts/rebuild_pipeline.py --clean-only
        """
    )
    
    parser.add_argument(
        '--clean',
        action='store_true',
        help='Delete all intermediate files before starting (recommended for full rebuild)'
    )
    
    parser.add_argument(
        '--clean-only',
        action='store_true',
        help='Only clean intermediate files, do not run pipeline'
    )
    
    parser.add_argument(
        '--skip-processing',
        action='store_true',
        help='Skip document processing step (use existing processed files)'
    )
    
    parser.add_argument(
        '--base-dir',
        type=Path,
        default=Path.cwd(),
        help='Base directory of the RAG application (default: current directory)'
    )
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = RAGPipeline(args.base_dir)
    
    # Validate base directory
    if not (pipeline.scripts_dir / "process_documents.py").exists():
        logger.error(f"Error: Could not find scripts in {pipeline.scripts_dir}")
        logger.error("Please run this script from the RAG application root directory")
        sys.exit(1)
    
    # Clean intermediate files if requested
    if args.clean or args.clean_only:
        pipeline.clean_intermediate_files(keep_documents=True)
    
    # Run pipeline unless clean-only
    if not args.clean_only:
        pipeline.run_full_pipeline(skip_processing=args.skip_processing)
    else:
        logger.info("Clean-only mode: Pipeline execution skipped\n")


if __name__ == "__main__":
    main()
