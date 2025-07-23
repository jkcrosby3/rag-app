"""
Document tracking system that maintains a record of document updates and changes.
"""
import json
import os
from pathlib import Path
from datetime import datetime
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class DocumentTracker:
    """Tracks document updates and maintains a history of changes with version control."""
    
    def __init__(self, tracking_file: str = "document_tracking.json"):
        """
        Initialize the document tracker with version control and notifications.
        
        Args:
            tracking_file: Path to the tracking file
        """
        self.tracking_file = Path(tracking_file)
        self.tracking_data = self._load_tracking_data()
        self._setup_notifications()
    
    def _load_tracking_data(self) -> Dict[str, Any]:
        """Load existing tracking data or create a new file if it doesn't exist."""
        if self.tracking_file.exists():
            try:
                with open(self.tracking_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error loading tracking file: {str(e)}")
        return {}
    
    def _setup_notifications(self) -> None:
        """Setup notification system."""
        self.notifications = []
        self.notification_handlers = []
        
    def add_notification_handler(self, handler):
        """
        Add a notification handler function.
        
        Args:
            handler: Function that takes (event_type, message) as arguments
        """
        self.notification_handlers.append(handler)
    
    def _notify(self, event_type: str, message: str) -> None:
        """Send notifications to all registered handlers."""
        for handler in self.notification_handlers:
            try:
                handler(event_type, message)
            except Exception as e:
                logger.error(f"Error in notification handler: {str(e)}")
    
    def _save_tracking_data(self) -> None:
        """Save tracking data to file."""
        try:
            with open(self.tracking_file, 'w', encoding='utf-8') as f:
                json.dump(self.tracking_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving tracking file: {str(e)}")
    
    def _get_file_hash(self, file_path: Path) -> str:
        """Generate a hash of the file contents."""
        import hashlib
        try:
            hash_sha256 = hashlib.sha256()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception as e:
            logger.error(f"Error generating file hash: {str(e)}")
            return ""
    
    def update_document_tracking(self, file_path: Path, metadata: Dict[str, Any]) -> None:
        """
        Update tracking information for a document with version control.
        
        Args:
            file_path: Path to the document file
            metadata: Document metadata
        """
        now = datetime.now()
        
        # Get file hash
        file_hash = self._get_file_hash(file_path)
        
        # Get relative path
        relative_path = str(file_path.relative_to(Path.cwd()))
        
        # Get current tracking entry
        tracking_entry = self.tracking_data.get(relative_path, {
            "last_modified": now.isoformat(),
            "file_size": os.path.getsize(file_path),
            "file_hash": file_hash,
            "metadata": metadata,
            "history": []
        })
        
        # Create version entry
        version_entry = {
            "version": len(tracking_entry["history"]) + 1,
            "timestamp": now.isoformat(),
            "file_size": os.path.getsize(file_path),
            "file_hash": file_hash,
            "metadata": metadata.copy(),
            "changes": self._detect_metadata_changes(tracking_entry["metadata"], metadata)
        }
        
        # Update history
        tracking_entry["history"].append(version_entry)
        
        # Update current tracking
        tracking_entry.update({
            "last_modified": now.isoformat(),
            "file_size": os.path.getsize(file_path),
            "file_hash": file_hash,
            "metadata": metadata,
            "version": version_entry["version"]
        })
        
        # Save to tracking data
        self.tracking_data[relative_path] = tracking_entry
        
        # Save changes
        self._save_tracking_data()
        
        # Notify about changes
        self._notify("document_update", f"Document {relative_path} updated to version {version_entry['version']}")
        logger.info(f"Updated tracking for {relative_path} to version {version_entry['version']}")
    
    def get_document_tracking(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """
        Get tracking information for a document.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Dictionary containing tracking information or None if not found
        """
        relative_path = str(file_path.relative_to(Path.cwd()))
        return self.tracking_data.get(relative_path)
    
    def get_all_tracking(self) -> Dict[str, Any]:
        """
        Get tracking information for all documents.
        
        Returns:
            Dictionary containing tracking information for all documents
        """
        return self.tracking_data
    
    def check_for_changes(self, file_path: Path) -> bool:
        """
        Check if a document has been modified since last tracking.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            True if the document has changed, False otherwise
        """
        relative_path = str(file_path.relative_to(Path.cwd()))
        if relative_path not in self.tracking_data:
            return True
            
        current_hash = self._get_file_hash(file_path)
        last_hash = self.tracking_data[relative_path].get("file_hash", "")
        
        return current_hash != last_hash
    
    def generate_tracking_report(self, output_file: Path = None, include_versions: bool = False) -> str:
        """
        Generate a report of all tracked documents.
        
        Args:
            output_file: Optional file to save the report
            include_versions: Whether to include version history
            
        Returns:
            String containing the tracking report
        """
        report = "Document Tracking Report\n"
        report += "=====================\n\n"
        
        for path, info in sorted(self.tracking_data.items()):
            report += f"Document: {path}\n"
            report += f"Last Modified: {info['last_modified']}\n"
            report += f"File Size: {info['file_size'] / 1024:.2f} KB\n"
            report += f"Classification: {info['metadata'].get('classification', 'public')}\n"
            report += f"Current Version: {info['version']}\n"
            
            if include_versions:
                report += "\nVersion History:\n"
                for version in sorted(info["history"], key=lambda x: x["version"]):
                    report += f"  - Version {version['version']} ({version['timestamp']})\n"
                    report += f"    File Size: {version['file_size'] / 1024:.2f} KB\n"
                    if version.get("changes"):
                        report += "    Changes:\n"
                        for change, value in version["changes"].items():
                            action = "Added" if change.startswith("+") else "Removed"
                            field = change[1:] if change.startswith("+") else change[1:]
                            report += f"      - {action} {field}: {value}\n"
                report += "-" * 80 + "\n\n"
            else:
                report += "-" * 80 + "\n\n"
        
        if output_file:
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(report)
                logger.info(f"Saved tracking report to {output_file}")
            except Exception as e:
                logger.error(f"Error saving tracking report: {str(e)}")
        
        return report
