"""
Hugging Face Hub compatibility layer.

This module provides compatibility functions for different versions of huggingface_hub,
allowing the code to work with both older and newer versions of the library.
"""
import os
import logging
from typing import Optional, Union

logger = logging.getLogger(__name__)

def cached_download(
    url_or_filename: str,
    cache_dir: Optional[Union[str, os.PathLike]] = None,
    force_download: bool = False,
    resume_download: bool = False,
    proxies: Optional[dict] = None,
    etag_timeout: Optional[float] = None,
    **kwargs
) -> str:
    """
    Compatibility function that mimics the old cached_download function
    but uses the newer huggingface_hub APIs.
    
    Args:
        url_or_filename: The URL or filename to download
        cache_dir: The directory where to save the downloaded file
        force_download: Whether to force the download even if the file exists
        resume_download: Whether to resume the download if possible
        proxies: Dictionary mapping protocol to the URL of the proxy
        etag_timeout: Timeout in seconds for the HTTP request to get the ETag
        **kwargs: Additional arguments to pass to the underlying download function
        
    Returns:
        Path to the downloaded file
    """
    try:
        # Try to use the newer hf_hub_download function
        from huggingface_hub import hf_hub_download
        
        # Extract repo_id and filename from URL if it's a huggingface.co URL
        if "huggingface.co" in url_or_filename:
            # Example URL: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/config.json
            parts = url_or_filename.split("/")
            if len(parts) >= 5 and parts[2] == "huggingface.co":
                repo_id = f"{parts[3]}/{parts[4]}"
                filename = "/".join(parts[7:])
                
                logger.info(f"Using hf_hub_download for {repo_id}/{filename}")
                return hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    **kwargs
                )
        
        # Fall back to using the requests library for direct downloads
        import requests
        from pathlib import Path
        import hashlib
        
        # Create cache directory if it doesn't exist
        if cache_dir is None:
            cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
        os.makedirs(cache_dir, exist_ok=True)
        
        # Create a filename based on the URL
        url_hash = hashlib.sha256(url_or_filename.encode("utf-8")).hexdigest()
        filename = os.path.join(cache_dir, url_hash)
        
        # Download the file if it doesn't exist or force_download is True
        if not os.path.exists(filename) or force_download:
            logger.info(f"Downloading {url_or_filename} to {filename}")
            response = requests.get(url_or_filename, proxies=proxies, stream=True)
            response.raise_for_status()
            
            with open(filename, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        
        return filename
    
    except ImportError:
        # If huggingface_hub is not installed or is an old version, try to use the old function
        try:
            from huggingface_hub import cached_download as original_cached_download
            return original_cached_download(
                url_or_filename=url_or_filename,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                etag_timeout=etag_timeout,
                **kwargs
            )
        except ImportError:
            logger.error("Could not import cached_download from huggingface_hub")
            raise ImportError(
                "Could not find cached_download in huggingface_hub. "
                "Please install a compatible version with: "
                "pip install huggingface_hub==0.9.1"
            )
