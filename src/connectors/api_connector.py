"""
Generic API connector for the RAG system.

This module provides functionality to connect to various APIs
and retrieve data for processing.
"""
import logging
import os
import time
import json
import requests
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Callable
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class APIConnector:
    """Connects to external APIs and retrieves data."""
    
    def __init__(
        self,
        base_url: str,
        auth_type: str = "none",
        api_key: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        token: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        cache_ttl_minutes: int = 60
    ):
        """Initialize the API connector.
        
        Args:
            base_url: Base URL of the API
            auth_type: Authentication type ('none', 'api_key', 'basic', 'bearer')
            api_key: API key for 'api_key' authentication
            username: Username for 'basic' authentication
            password: Password for 'basic' authentication
            token: Token for 'bearer' authentication
            headers: Additional headers to include in requests
            cache_dir: Directory to cache API responses
            cache_ttl_minutes: Time-to-live for cached responses in minutes
        """
        self.base_url = base_url.rstrip('/')
        self.auth_type = auth_type
        self.api_key = api_key
        self.username = username
        self.password = password
        self.token = token
        self.headers = headers or {}
        self.cache_ttl_minutes = cache_ttl_minutes
        
        # Set up cache directory
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = Path("data") / "api_cache"
            
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up session
        self.session = requests.Session()
        self._configure_auth()
        
    def _configure_auth(self):
        """Configure authentication for the API."""
        if self.auth_type == "api_key":
            if not self.api_key:
                logger.error("API key required for api_key authentication")
                return
                
            # Default to Authorization header, but this can be overridden in headers
            if "Authorization" not in self.headers:
                self.headers["Authorization"] = f"ApiKey {self.api_key}"
                
        elif self.auth_type == "basic":
            if not self.username or not self.password:
                logger.error("Username and password required for basic authentication")
                return
                
            self.session.auth = (self.username, self.password)
            
        elif self.auth_type == "bearer":
            if not self.token:
                logger.error("Token required for bearer authentication")
                return
                
            self.headers["Authorization"] = f"Bearer {self.token}"
            
        # Apply headers to session
        self.session.headers.update(self.headers)
        
    def _get_cache_key(self, endpoint: str, params: Dict[str, Any]) -> str:
        """Generate a cache key for the request.
        
        Args:
            endpoint: API endpoint
            params: Query parameters
            
        Returns:
            Cache key string
        """
        # Create a string representation of the parameters
        params_str = json.dumps(params, sort_keys=True)
        
        # Create a hash of the endpoint and parameters
        import hashlib
        hash_obj = hashlib.md5(f"{endpoint}:{params_str}".encode())
        return hash_obj.hexdigest()
        
    def _get_cached_response(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached response if available and not expired.
        
        Args:
            cache_key: Cache key
            
        Returns:
            Cached response or None if not available
        """
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        if not cache_file.exists():
            return None
            
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                cache_data = json.load(f)
                
            # Check cache age
            cached_time = datetime.fromisoformat(cache_data["_cached_at"])
            age = datetime.now() - cached_time
            
            if age > timedelta(minutes=self.cache_ttl_minutes):
                return None
                
            return cache_data["data"]
            
        except Exception as e:
            logger.error(f"Error reading cache: {str(e)}")
            return None
            
    def _cache_response(self, cache_key: str, response_data: Any):
        """Cache API response.
        
        Args:
            cache_key: Cache key
            response_data: Response data to cache
        """
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        try:
            cache_data = {
                "_cached_at": datetime.now().isoformat(),
                "data": response_data
            }
            
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(cache_data, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Error writing cache: {str(e)}")
            
    def get(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        use_cache: bool = True,
        transform: Optional[Callable[[Dict[str, Any]], Any]] = None
    ) -> Any:
        """Make a GET request to the API.
        
        Args:
            endpoint: API endpoint (relative to base_url)
            params: Query parameters
            use_cache: Whether to use cached responses
            transform: Optional function to transform the response
            
        Returns:
            API response data
        """
        params = params or {}
        endpoint = endpoint.lstrip('/')
        url = f"{self.base_url}/{endpoint}"
        
        # Check cache first if enabled
        if use_cache:
            cache_key = self._get_cache_key(endpoint, params)
            cached_response = self._get_cached_response(cache_key)
            
            if cached_response:
                logger.debug(f"Using cached response for {endpoint}")
                
                if transform:
                    return transform(cached_response)
                return cached_response
        
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            # Cache response if enabled
            if use_cache:
                cache_key = self._get_cache_key(endpoint, params)
                self._cache_response(cache_key, data)
                
            # Transform response if needed
            if transform:
                return transform(data)
                
            return data
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API request error: {str(e)}")
            return None
            
    def post(
        self,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        files: Optional[Dict[str, Any]] = None,
        transform: Optional[Callable[[Dict[str, Any]], Any]] = None
    ) -> Any:
        """Make a POST request to the API.
        
        Args:
            endpoint: API endpoint (relative to base_url)
            data: Form data
            json_data: JSON data
            files: Files to upload
            transform: Optional function to transform the response
            
        Returns:
            API response data
        """
        endpoint = endpoint.lstrip('/')
        url = f"{self.base_url}/{endpoint}"
        
        try:
            response = self.session.post(url, data=data, json=json_data, files=files)
            response.raise_for_status()
            
            # Some APIs might not return JSON
            try:
                data = response.json()
            except ValueError:
                data = {"text": response.text}
                
            # Transform response if needed
            if transform:
                return transform(data)
                
            return data
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API request error: {str(e)}")
            return None


def get_api_connector(api_name: str, config_file: Optional[Union[str, Path]] = None) -> Optional[APIConnector]:
    """Get an API connector instance from configuration.
    
    Args:
        api_name: Name of the API configuration to use
        config_file: Path to configuration file
            
    Returns:
        Configured APIConnector instance or None if not found
    """
    # Default config path
    if not config_file:
        config_file = Path("config") / "apis.json"
    else:
        config_file = Path(config_file)
        
    # Check if config exists
    if not config_file.exists():
        logger.warning(f"API config file not found: {config_file}")
        logger.warning(f"Looking for environment variables for {api_name}")
        
        # Try environment variables
        env_prefix = f"{api_name.upper()}_"
        base_url = os.environ.get(f"{env_prefix}BASE_URL")
        
        if not base_url:
            logger.error(f"API base URL not configured for {api_name}")
            return None
            
        auth_type = os.environ.get(f"{env_prefix}AUTH_TYPE", "none")
        api_key = os.environ.get(f"{env_prefix}API_KEY")
        username = os.environ.get(f"{env_prefix}USERNAME")
        password = os.environ.get(f"{env_prefix}PASSWORD")
        token = os.environ.get(f"{env_prefix}TOKEN")
        
        # Parse headers if available
        headers = {}
        headers_str = os.environ.get(f"{env_prefix}HEADERS")
        if headers_str:
            try:
                headers = json.loads(headers_str)
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON in {env_prefix}HEADERS")
                
    else:
        # Load config from file
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                config = json.load(f)
                
            if api_name not in config:
                logger.error(f"API configuration not found for {api_name}")
                return None
                
            api_config = config[api_name]
            base_url = api_config.get("base_url")
            
            if not base_url:
                logger.error(f"API base URL not configured for {api_name}")
                return None
                
            auth_type = api_config.get("auth_type", "none")
            api_key = api_config.get("api_key")
            username = api_config.get("username")
            password = api_config.get("password")
            token = api_config.get("token")
            headers = api_config.get("headers", {})
            
        except Exception as e:
            logger.error(f"Error loading API config: {str(e)}")
            return None
            
    # Create connector
    return APIConnector(
        base_url=base_url,
        auth_type=auth_type,
        api_key=api_key,
        username=username,
        password=password,
        token=token,
        headers=headers
    )
