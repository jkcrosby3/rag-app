#!/usr/bin/env python
"""
Prepare Files for Email Script

This script prepares the RAG application files for email by:
1. Creating a 'to-email' directory
2. Copying all relevant files to this directory
3. Renaming files to include their folder path in the filename
4. Adding .txt extension to all files to bypass email filters

=== HOW TO USE THIS SCRIPT ===

1. Navigate to the project directory in your terminal/command prompt
2. Run the script with one of these commands:

   # Use default directories (recommended):
   python tools/prepare_files_for_email.py
   
   # Specify source directory:
   python tools/prepare_files_for_email.py /path/to/rag-app
   
   # Specify both source and output directories:
   python tools/prepare_files_for_email.py /path/to/rag-app /path/to/output

3. Find your email-ready files in the 'to-email' directory
4. Attach these files to your email

The script will create filenames like 'src-embeddings-model.py.txt' that preserve
the directory structure in the filename itself.

Usage:
    python prepare_files_for_email.py [source_directory] [output_directory]

    - source_directory: Directory containing the RAG application (default: parent of script directory)
    - output_directory: Directory where renamed files will be placed (default: 'to-email' in source directory)
"""

import os
import sys
import shutil
import re
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Directories to exclude from processing
EXCLUDE_DIRS = [
    # Virtual environment directories - comment out if you want to include them
    'venv',  # Standard virtual environment
    # '.venv',  # Uncomment this line if you want to include .venv directory
    
    # Other directories to exclude
    '.git',
    '__pycache__',
    'to-email',
    '.pytest_cache',
    '.ipynb_checkpoints',
    'node_modules'
]
# to recreate the environment using dependency files:
# For Python: Use pip install -r requirements.txt
# For Node.js: Use npm install

# File extensions to process (empty list means process all files)
# Add specific extensions if you want to limit which files are processed
INCLUDE_EXTENSIONS = []

# Extensions that might cause issues with email systems
# These will be specifically monitored in the logs
POTENTIALLY_PROBLEMATIC_EXTENSIONS = [
    '.py',    # Python files
    '.json',  # JSON files
    '.yaml',  # YAML files
    '.toml',  # TOML files
    '.html',  # HTML files
    '.md',    # Markdown files
    '.png',   # Image files
    '.template'  # Template files
]

# Files to exclude
EXCLUDE_FILES = [
    '.DS_Store',
    'Thumbs.db'
]

def should_process_file(filepath):
    """
    Determine if a file should be processed based on its path and extension.
    
    Args:
        filepath (str): Path to the file
        
    Returns:
        bool: True if the file should be processed, False otherwise
    """
    # Get the filename and check if it's in the exclude list
    filename = os.path.basename(filepath)
    if filename in EXCLUDE_FILES:
        return False
        
    # If we have a specific list of extensions to include, check against it
    if INCLUDE_EXTENSIONS:
        ext = os.path.splitext(filename)[1].lower()
        if ext and ext[1:] not in INCLUDE_EXTENSIONS:  # Remove the dot from extension
            return False
            
    # Check if the file is in an excluded directory
    parts = Path(filepath).parts
    for excluded in EXCLUDE_DIRS:
        if excluded in parts:
            return False
            
    return True

def get_relative_path(filepath, base_dir):
    """
    Get the relative path of a file from the base directory.
    
    Args:
        filepath (str): Full path to the file
        base_dir (str): Base directory
        
    Returns:
        str: Relative path from base_dir to the file
    """
    return os.path.relpath(filepath, base_dir)

def create_email_filename(relative_path):
    """
    Create a filename suitable for email by replacing path separators with hyphens.
    
    Args:
        relative_path (str): Relative path of the file
        
    Returns:
        str: Email-friendly filename with .txt extension
    """
    # Replace directory separators with hyphens
    email_name = relative_path.replace(os.sep, '-')
    
    # Get the extension
    _, ext = os.path.splitext(email_name)
    
    # Check if this is a potentially problematic extension
    if ext.lower() in POTENTIALLY_PROBLEMATIC_EXTENSIONS:
        logger.info(f"Found potentially problematic extension: {ext} in {relative_path}")
    
    # Add .txt extension if the file doesn't already have it
    if not email_name.endswith('.txt'):
        email_name += '.txt'
        
    return email_name

def prepare_files(source_dir, output_dir):
    """
    Copy and rename files from source_dir to output_dir for email preparation.
    
    Args:
        source_dir (str): Source directory containing the RAG application
        output_dir (str): Output directory where renamed files will be placed
        
    Returns:
        tuple: (Number of files processed, Dictionary of extension counts)
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created output directory: {output_dir}")
    
    count = 0
    extension_counts = {}
    
    # Walk through the source directory
    for root, _, files in os.walk(source_dir):
        for filename in files:
            filepath = os.path.join(root, filename)
            
            # Skip files that shouldn't be processed
            if not should_process_file(filepath):
                continue
                
            # Get the relative path and create the email filename
            relative_path = get_relative_path(filepath, source_dir)
            email_filename = create_email_filename(relative_path)
            
            # Track extension counts
            _, ext = os.path.splitext(filename)
            ext = ext.lower() if ext else '(no extension)'
            extension_counts[ext] = extension_counts.get(ext, 0) + 1
            
            # Create the output path
            output_path = os.path.join(output_dir, email_filename)
            
            # Copy the file
            shutil.copy2(filepath, output_path)
            logger.info(f"Copied: {relative_path} → {email_filename}")
            count += 1
    
    return count, extension_counts

def main():
    """Main function to prepare files for email."""
    # Determine source directory (default: parent of script directory)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_source_dir = os.path.dirname(script_dir)
    
    # Parse command line arguments
    if len(sys.argv) > 2:
        source_dir = sys.argv[1]
        output_dir = sys.argv[2]
    elif len(sys.argv) > 1:
        source_dir = sys.argv[1]
        output_dir = os.path.join(source_dir, 'to-email')
    else:
        source_dir = default_source_dir
        output_dir = os.path.join(source_dir, 'to-email')
    
    # Create to-email directory if it doesn't exist
    if not os.path.exists('to-email'):
        try:
            os.makedirs('to-email')
            logger.info("Created to-email directory")
        except Exception as e:
            logger.error(f"Failed to create to-email directory: {str(e)}")
            return 1
    
    # Validate source directory
    if not os.path.isdir(source_dir):
        logger.error(f"Source directory does not exist: {source_dir}")
        return 1
    
    logger.info(f"Preparing files from {source_dir} for email")
    logger.info(f"Output directory: {output_dir}")
    
    # Process files
    count, extension_counts = prepare_files(source_dir, output_dir)
    
    # Log extension statistics
    logger.info(f"Extension statistics:")
    for ext, ext_count in sorted(extension_counts.items(), key=lambda x: (-x[1], x[0])):
        logger.info(f"  {ext}: {ext_count} files")
        
        # Highlight potentially problematic extensions
        if ext in POTENTIALLY_PROBLEMATIC_EXTENSIONS:
            logger.info(f"  ⚠️ Note: {ext} files may require special handling in some email systems")
    
    logger.info(f"Preparation complete! {count} files were processed.")
    logger.info(f"Files are ready for email in: {output_dir}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
