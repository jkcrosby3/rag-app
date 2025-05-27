"""
Archive.org book downloader for the RAG system.

This script downloads books from the Internet Archive and prepares them
for ingestion into the RAG system.
"""
import os
import sys
import json
import time
import logging
import argparse
import requests
from pathlib import Path
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constants
ARCHIVE_API_URL = "https://archive.org/advancedsearch.php"
ARCHIVE_DOWNLOAD_URL = "https://archive.org/download"
MAX_WORKERS = 4


def search_books(query: str, max_results: int = 10) -> List[Dict[str, Any]]:
    """
    Search for books on the Internet Archive.
    
    Args:
        query: Search query
        max_results: Maximum number of results to return
        
    Returns:
        List of book metadata
    """
    params = {
        "q": f"{query} AND mediatype:texts AND language:English",
        "fl[]": ["identifier", "title", "creator", "date", "description", "subject"],
        "rows": max_results,
        "page": 1,
        "output": "json"
    }
    
    try:
        response = requests.get(ARCHIVE_API_URL, params=params)
        response.raise_for_status()
        
        data = response.json()
        return data.get("response", {}).get("docs", [])
    except Exception as e:
        logger.error(f"Error searching for books: {str(e)}")
        return []


def get_book_files(identifier: str) -> List[Dict[str, Any]]:
    """
    Get list of files for a book.
    
    Args:
        identifier: Internet Archive identifier
        
    Returns:
        List of file metadata
    """
    metadata_url = f"{ARCHIVE_DOWNLOAD_URL}/{identifier}/{identifier}_files.json"
    
    try:
        response = requests.get(metadata_url)
        response.raise_for_status()
        
        data = response.json()
        return data.get("files", [])
    except Exception as e:
        logger.error(f"Error getting files for {identifier}: {str(e)}")
        return []


def download_book(book: Dict[str, Any], output_dir: str, file_format: str = "pdf") -> Optional[str]:
    """
    Download a book from the Internet Archive.
    
    Args:
        book: Book metadata
        output_dir: Directory to save the book
        file_format: File format to download (pdf, txt, epub)
        
    Returns:
        Path to the downloaded file or None if download failed
    """
    identifier = book.get("identifier")
    title = book.get("title", identifier)
    
    if not identifier:
        logger.error(f"Missing identifier for book: {title}")
        return None
    
    # Create sanitized filename
    safe_title = "".join(c if c.isalnum() or c in " ._-" else "_" for c in title)
    safe_title = safe_title[:100]  # Truncate long titles
    
    # Get list of files
    files = get_book_files(identifier)
    
    # Find the requested format
    target_file = None
    for file_data in files:
        file_name = file_data.get("name", "")
        if file_name.endswith(f".{file_format}"):
            target_file = file_name
            break
    
    # If format not found, try to find a text file
    if not target_file and file_format != "txt":
        for file_data in files:
            file_name = file_data.get("name", "")
            if file_name.endswith(".txt"):
                target_file = file_name
                file_format = "txt"
                break
    
    if not target_file:
        logger.warning(f"No {file_format} file found for {title} ({identifier})")
        return None
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Download file
    download_url = f"{ARCHIVE_DOWNLOAD_URL}/{identifier}/{target_file}"
    output_path = os.path.join(output_dir, f"{safe_title}.{file_format}")
    
    try:
        logger.info(f"Downloading {title} ({identifier}) to {output_path}")
        response = requests.get(download_url, stream=True)
        response.raise_for_status()
        
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        # Save metadata
        metadata = {
            "identifier": identifier,
            "title": title,
            "creator": book.get("creator", "Unknown"),
            "date": book.get("date", "Unknown"),
            "description": book.get("description", ""),
            "subjects": book.get("subject", []),
            "source": f"https://archive.org/details/{identifier}",
            "download_url": download_url
        }
        
        metadata_path = os.path.join(output_dir, f"{safe_title}.metadata.json")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Successfully downloaded {title}")
        return output_path
    except Exception as e:
        logger.error(f"Error downloading {title} ({identifier}): {str(e)}")
        return None


def download_books(query: str, output_dir: str, max_books: int = 10, file_format: str = "pdf") -> List[str]:
    """
    Search and download books from the Internet Archive.
    
    Args:
        query: Search query
        output_dir: Directory to save the books
        max_books: Maximum number of books to download
        file_format: File format to download (pdf, txt, epub)
        
    Returns:
        List of paths to downloaded files
    """
    # Search for books
    logger.info(f"Searching for books matching query: {query}")
    books = search_books(query, max_results=max_books * 2)  # Get more results in case some downloads fail
    
    if not books:
        logger.error(f"No books found for query: {query}")
        return []
    
    logger.info(f"Found {len(books)} books matching query")
    
    # Download books in parallel
    downloaded_paths = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for book in books[:max_books * 2]:
            future = executor.submit(download_book, book, output_dir, file_format)
            futures.append(future)
            
            # Throttle requests to avoid rate limiting
            time.sleep(1)
        
        # Collect results
        for future in futures:
            path = future.result()
            if path:
                downloaded_paths.append(path)
                
                # Stop if we've reached the desired number of books
                if len(downloaded_paths) >= max_books:
                    break
    
    return downloaded_paths[:max_books]


def main():
    parser = argparse.ArgumentParser(description="Download books from the Internet Archive")
    parser.add_argument("query", type=str, help="Search query")
    parser.add_argument("--output-dir", type=str, default="data/documents", help="Directory to save the books")
    parser.add_argument("--max-books", type=int, default=10, help="Maximum number of books to download")
    parser.add_argument("--format", type=str, default="pdf", choices=["pdf", "txt", "epub"], help="File format to download")
    
    args = parser.parse_args()
    
    # Download books
    downloaded_paths = download_books(
        query=args.query,
        output_dir=args.output_dir,
        max_books=args.max_books,
        file_format=args.format
    )
    
    # Print summary
    if downloaded_paths:
        logger.info(f"Successfully downloaded {len(downloaded_paths)} books to {args.output_dir}")
        for path in downloaded_paths:
            logger.info(f"  - {path}")
    else:
        logger.error("Failed to download any books")


if __name__ == "__main__":
    main()
