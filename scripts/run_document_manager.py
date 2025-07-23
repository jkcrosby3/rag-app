"""
Run the document management UI for the RAG system.

This script launches a web interface for managing document ingestion,
listing, and processing from various sources.
"""
import os
import sys
import logging
import traceback
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

def ensure_directories_exist():
    """Ensure all required data directories exist."""
    data_dir = project_root / "data"
    directories = [
        data_dir,
        data_dir / "documents",
        data_dir / "processed",
        data_dir / "chunks",
        data_dir / "embedded",
        data_dir / "staging"
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"Ensured directory exists: {directory}")

def main():
    try:
        print("Starting Document Management UI...")
        
        # Ensure directories exist
        ensure_directories_exist()
        
        # Check for required packages
        try:
            import gradio
            import pandas
            from src.document_processing.chunker import Chunker
            from src.document_processing.batch_processor import BatchProcessor
            logger.info("All required packages are available")
        except ImportError as e:
            logger.error(f"Missing required package: {e}")
            print(f"Error: Missing required package: {e}")
            print("Please install required packages with: pip install PyMuPDF Office365-REST-Python-Client gradio pandas")
            return
        
        # Import UI module
        from src.web.document_ui import launch_ui
        
        print("Access the UI in your web browser")
        launch_ui(share=False)
        
    except Exception as e:
        logger.error(f"Error starting document management UI: {e}")
        logger.error(traceback.format_exc())
        print(f"Error: {e}")
        print("Check the logs for more details")

if __name__ == "__main__":
    main()
