"""
SharePoint connector for the RAG system.

This module provides functionality to connect to SharePoint,
list available documents, and download them for processing.
"""
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import json
import datetime

# Import Office 365 libraries
try:
    from office365.runtime.auth.authentication_context import AuthenticationContext
    from office365.sharepoint.client_context import ClientContext
    from office365.sharepoint.files.file import File
except ImportError:
    logging.getLogger(__name__).warning(
        "Office365-REST-Python-Client not installed. "
        "Install with: pip install Office365-REST-Python-Client"
    )

logger = logging.getLogger(__name__)


class SharePointConnector:
    """Connects to SharePoint and manages document retrieval."""
    
    def __init__(
        self,
        site_url: str,
        username: Optional[str] = None,
        password: Optional[str] = None,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        auth_type: str = "credentials",
        cache_dir: Optional[Union[str, Path]] = None
    ):
        """Initialize the SharePoint connector.
        
        Args:
            site_url: URL of the SharePoint site
            username: SharePoint username (for credential auth)
            password: SharePoint password (for credential auth)
            client_id: Client ID (for app authentication)
            client_secret: Client secret (for app authentication)
            auth_type: Authentication type ('credentials' or 'app')
            cache_dir: Directory to cache document metadata
        """
        self.site_url = site_url
        self.username = username
        self.password = password
        self.client_id = client_id
        self.client_secret = client_secret
        self.auth_type = auth_type
        
        # Set up cache directory
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = Path("data") / "sharepoint_cache"
            
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize client context
        self.ctx = None
        
    def connect(self) -> bool:
        """Establish connection to SharePoint.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            if self.auth_type == "credentials":
                if not self.username or not self.password:
                    logger.error("Username and password required for credential authentication")
                    return False
                    
                auth_context = AuthenticationContext(self.site_url)
                auth_context.acquire_token_for_user(self.username, self.password)
                self.ctx = ClientContext(self.site_url, auth_context)
                
            elif self.auth_type == "app":
                if not self.client_id or not self.client_secret:
                    logger.error("Client ID and secret required for app authentication")
                    return False
                    
                auth_context = AuthenticationContext(self.site_url)
                auth_context.acquire_token_for_app(self.client_id, self.client_secret)
                self.ctx = ClientContext(self.site_url, auth_context)
                
            else:
                logger.error(f"Unsupported authentication type: {self.auth_type}")
                return False
                
            # Test connection
            web = self.ctx.web
            self.ctx.load(web)
            self.ctx.execute_query()
            logger.info(f"Connected to SharePoint site: {web.properties['Title']}")
            return True
            
        except Exception as e:
            logger.error(f"Error connecting to SharePoint: {str(e)}")
            return False
            
    def list_documents(
        self, 
        library_name: str, 
        folder_path: str = "", 
        file_extensions: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """List documents in a SharePoint document library.
        
        Args:
            library_name: Name of the document library
            folder_path: Path to folder within the library
            file_extensions: List of file extensions to filter by
                
        Returns:
            List of document metadata dictionaries
        """
        if not self.ctx:
            if not self.connect():
                return []
                
        try:
            # Build the relative URL
            if folder_path:
                relative_url = f"{library_name}/{folder_path}"
            else:
                relative_url = library_name
                
            # Get folder
            folder = self.ctx.web.get_folder_by_server_relative_url(relative_url)
            self.ctx.load(folder)
            self.ctx.execute_query()
            
            # Get files
            files = folder.files
            self.ctx.load(files)
            self.ctx.execute_query()
            
            # Process files
            documents = []
            
            for file in files:
                # Check file extension if filter provided
                if file_extensions:
                    _, ext = os.path.splitext(file.properties["Name"])
                    if ext.lower() not in [ext.lower() if ext.startswith('.') else f'.{ext.lower()}' 
                                          for ext in file_extensions]:
                        continue
                        
                # Extract metadata
                doc_info = {
                    "name": file.properties["Name"],
                    "url": file.properties["ServerRelativeUrl"],
                    "size": file.properties["Length"],
                    "modified": file.properties["TimeLastModified"],
                    "created": file.properties["TimeCreated"],
                    "library": library_name,
                    "folder_path": folder_path
                }
                
                documents.append(doc_info)
                
            # Update cache
            self._update_cache(library_name, folder_path, documents)
            
            return documents
            
        except Exception as e:
            logger.error(f"Error listing documents: {str(e)}")
            return []
            
    def download_document(
        self, 
        document_url: str, 
        target_dir: Union[str, Path],
        preserve_path: bool = False
    ) -> Optional[Path]:
        """Download a document from SharePoint.
        
        Args:
            document_url: Server-relative URL of the document
            target_dir: Directory to save the document to
            preserve_path: Whether to preserve the folder structure
                
        Returns:
            Path to the downloaded file, or None if download failed
        """
        if not self.ctx:
            if not self.connect():
                return None
                
        try:
            # Ensure target directory exists
            target_dir = Path(target_dir)
            target_dir.mkdir(parents=True, exist_ok=True)
            
            # Get file
            file = self.ctx.web.get_file_by_server_relative_url(document_url)
            self.ctx.load(file)
            self.ctx.execute_query()
            
            # Determine target path
            file_name = os.path.basename(document_url)
            
            if preserve_path:
                # Extract relative path from document URL
                rel_path = os.path.dirname(document_url)
                # Remove library name from path
                parts = rel_path.split('/', 1)
                if len(parts) > 1:
                    rel_path = parts[1]
                else:
                    rel_path = ""
                    
                # Create subdirectories
                if rel_path:
                    sub_dir = target_dir / rel_path
                    sub_dir.mkdir(parents=True, exist_ok=True)
                    target_path = sub_dir / file_name
                else:
                    target_path = target_dir / file_name
            else:
                target_path = target_dir / file_name
                
            # Download file
            with open(target_path, "wb") as f:
                file_content = file.read()
                f.write(file_content)
                
            logger.info(f"Downloaded {document_url} to {target_path}")
            return target_path
            
        except Exception as e:
            logger.error(f"Error downloading document {document_url}: {str(e)}")
            return None
            
    def download_documents(
        self,
        documents: List[Dict[str, Any]],
        target_dir: Union[str, Path],
        preserve_path: bool = False
    ) -> List[Path]:
        """Download multiple documents from SharePoint.
        
        Args:
            documents: List of document metadata dictionaries
            target_dir: Directory to save the documents to
            preserve_path: Whether to preserve the folder structure
                
        Returns:
            List of paths to the downloaded files
        """
        downloaded_files = []
        
        for doc in documents:
            path = self.download_document(
                doc["url"],
                target_dir,
                preserve_path
            )
            
            if path:
                downloaded_files.append(path)
                
        return downloaded_files
        
    def _update_cache(
        self,
        library_name: str,
        folder_path: str,
        documents: List[Dict[str, Any]]
    ):
        """Update the local cache of document metadata.
        
        Args:
            library_name: Name of the document library
            folder_path: Path to folder within the library
            documents: List of document metadata dictionaries
        """
        # Create cache key
        cache_key = f"{library_name}_{folder_path.replace('/', '_')}"
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        # Update cache
        cache_data = {
            "library": library_name,
            "folder_path": folder_path,
            "last_updated": datetime.datetime.now().isoformat(),
            "documents": documents
        }
        
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, indent=2, default=str)
            
    def get_cached_documents(
        self,
        library_name: str,
        folder_path: str = "",
        max_age_minutes: int = 60
    ) -> Optional[List[Dict[str, Any]]]:
        """Get documents from cache if available and not expired.
        
        Args:
            library_name: Name of the document library
            folder_path: Path to folder within the library
            max_age_minutes: Maximum age of cache in minutes
                
        Returns:
            List of document metadata dictionaries or None if cache invalid
        """
        # Create cache key
        cache_key = f"{library_name}_{folder_path.replace('/', '_')}"
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        if not cache_file.exists():
            return None
            
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                cache_data = json.load(f)
                
            # Check cache age
            last_updated = datetime.datetime.fromisoformat(cache_data["last_updated"])
            age = datetime.datetime.now() - last_updated
            
            if age.total_seconds() > (max_age_minutes * 60):
                return None
                
            return cache_data["documents"]
            
        except Exception as e:
            logger.error(f"Error reading cache: {str(e)}")
            return None


def get_sharepoint_connector(config_file: Optional[Union[str, Path]] = None) -> SharePointConnector:
    """Get a SharePoint connector instance from configuration.
    
    Args:
        config_file: Path to configuration file
            
    Returns:
        Configured SharePointConnector instance
    """
    # Default config path
    if not config_file:
        config_file = Path("config") / "sharepoint.json"
    else:
        config_file = Path(config_file)
        
    # Check if config exists
    if not config_file.exists():
        logger.warning(f"SharePoint config file not found: {config_file}")
        logger.warning("Using environment variables for configuration")
        
        # Try environment variables
        site_url = os.environ.get("SHAREPOINT_SITE_URL")
        username = os.environ.get("SHAREPOINT_USERNAME")
        password = os.environ.get("SHAREPOINT_PASSWORD")
        client_id = os.environ.get("SHAREPOINT_CLIENT_ID")
        client_secret = os.environ.get("SHAREPOINT_CLIENT_SECRET")
        auth_type = os.environ.get("SHAREPOINT_AUTH_TYPE", "credentials")
        
        if not site_url:
            logger.error("SharePoint site URL not configured")
            raise ValueError("SharePoint site URL not configured")
            
    else:
        # Load config from file
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                config = json.load(f)
                
            site_url = config.get("site_url")
            username = config.get("username")
            password = config.get("password")
            client_id = config.get("client_id")
            client_secret = config.get("client_secret")
            auth_type = config.get("auth_type", "credentials")
            
            if not site_url:
                logger.error("SharePoint site URL not configured")
                raise ValueError("SharePoint site URL not configured")
                
        except Exception as e:
            logger.error(f"Error loading SharePoint config: {str(e)}")
            raise
            
    # Create connector
    return SharePointConnector(
        site_url=site_url,
        username=username,
        password=password,
        client_id=client_id,
        client_secret=client_secret,
        auth_type=auth_type
    )
