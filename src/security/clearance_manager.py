"""
Clearance management system for the RAG application.

This module handles user clearance verification, access control,
and audit logging for classified content.
"""
import os
import sys
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
import hashlib
from src.tools.metadata_validator import ClassificationHierarchy

logger = logging.getLogger(__name__)

class ClearanceManager:
    """Manages user clearances and access control."""
    
    def __init__(self, clearance_db_path: str = "data/clearances.json"):
        """Initialize the clearance manager.
        
        Args:
            clearance_db_path: Path to the clearance database file
        """
        self.clearance_db_path = Path(clearance_db_path)
        self.clearances = self._load_clearances()
        
    def _load_clearances(self) -> Dict[str, Any]:
        """Load clearances from the database."""
        if not self.clearance_db_path.exists():
            return {}
        
        with open(self.clearance_db_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _save_clearances(self):
        """Save clearances to the database."""
        with open(self.clearance_db_path, 'w', encoding='utf-8') as f:
            json.dump(self.clearances, f, indent=2)
    
    def verify_clearance(
        self,
        user_id: str,
        clearance_level: Dict[str, Any],
        document_id: Optional[str] = None
    ) -> bool:
        """Verify if a user has the required clearance.
        
        Args:
            user_id: User identifier
            clearance_level: Required clearance dictionary
            document_id: Optional document ID for need-to-know verification
            
        Returns:
            True if user has the required clearance, False otherwise
        """
        user_clearance = self.clearances.get(user_id)
        if not user_clearance:
            logger.warning(f"User {user_id} not found in clearance database")
            return False
            
        # Get user's clearance level dictionary
        user_clearance_level = user_clearance.get('level', {
            "classification": "U",
            "components": []
        })
        
        # Verify clearance level
        if not ClassificationHierarchy.is_accessible_by(
            user_clearance_level,
            clearance_level
        ):
            logger.warning(f"User {user_id} does not have required clearance level")
            return False
            
        # Verify expiration
        if user_clearance.get('expires'):
            expires = datetime.fromisoformat(user_clearance['expires'])
            if expires < datetime.now():
                logger.warning(f"User {user_id} clearance has expired")
                return False
                
        # Verify need-to-know if document_id is provided
        if document_id and 'need_to_know' in user_clearance:
            if document_id not in user_clearance['need_to_know']:
                logger.warning(f"User {user_id} does not have need-to-know for document {document_id}")
                return False
                
        return True
    
    def add_clearance(
        self,
        user_id: str,
        clearance_level: Dict[str, Any],
        expires: Optional[datetime] = None,
        need_to_know: Optional[List[str]] = None
    ) -> bool:
        """Add or update a user's clearance.
        
        Args:
            user_id: User identifier
            clearance_level: Clearance level (e.g., TS//SI)
            expires: Optional expiration date
            need_to_know: Optional list of document IDs for need-to-know access
            
        Returns:
            True if clearance was added/updated successfully
        """
        # Validate clearance level
        if not ClassificationHierarchy.is_valid_classification(clearance_level):
            logger.error(f"Invalid clearance level: {clearance_level}")
            return False
            
        # Convert string clearance to dictionary if needed
        if isinstance(clearance_level, str):
            clearance_level = ClassificationHierarchy.parse_classification(clearance_level)
            
        self.clearances[user_id] = {
            'level': clearance_level,
            'expires': expires.isoformat() if expires else None,
            'need_to_know': need_to_know or [],
            'last_updated': datetime.now().isoformat()
        }
        self._save_clearances()
        logger.info(f"Added clearance for user {user_id}: {clearance_level}")
        return True
    
    def remove_clearance(self, user_id: str) -> bool:
        """Remove a user's clearance.
        
        Args:
            user_id: User identifier
            
        Returns:
            True if clearance was removed successfully
        """
        if user_id in self.clearances:
            del self.clearances[user_id]
            self._save_clearances()
            logger.info(f"Removed clearance for user {user_id}")
            return True
        return False
    
    def get_user_clearance(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get a user's clearance information.
        
        Args:
            user_id: User identifier
            
        Returns:
            User's clearance information or None if not found
        """
        return self.clearances.get(user_id)
    
    def log_access_attempt(
        self,
        user_id: str,
        document_id: str,
        success: bool,
        reason: Optional[str] = None
    ):
        """Log an access attempt.
        
        Args:
            user_id: User identifier
            document_id: Document identifier
            success: Whether access was granted
            reason: Optional reason for denial
        """
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'user_id': user_id,
            'document_id': document_id,
            'success': success,
            'reason': reason
        }
        
        # Create audit log directory if it doesn't exist
        audit_dir = Path("data/audit")
        audit_dir.mkdir(parents=True, exist_ok=True)
        
        # Create daily log file
        log_date = datetime.now().strftime("%Y%m%d")
        log_path = audit_dir / f"access_{log_date}.jsonl"
        
        # Append log entry
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def redact_content(
        self,
        content: List[Dict[str, Any]],
        user_id: str
    ) -> List[Dict[str, Any]]:
        """Redact content based on user's clearance.
        
        Args:
            content: List of content chunks with classification metadata
            user_id: User identifier
            
        Returns:
            List of content chunks with redacted content as needed
        """
        user_clearance = self.get_user_clearance(user_id)
        if not user_clearance:
            return [{
                'text': '[ACCESS DENIED - USER NOT AUTHORIZED]',
                'metadata': {'classification': 'U', 'redacted': True}
            }]
            
        redacted_content = []
        for item in content:
            # Get classification from either metadata or top level
            classification = item.get("classification", {
                "classification": "U",
                "components": []
            })
            
            # Check if document is redacted
            is_redacted = item.get("redacted", False)
            if not is_redacted and not self.verify_clearance(
                user_id,
                classification,
                document_id=item.get('document_id')
            ):
                # Redact content above user's clearance
                redacted_content.append({
                    'text': '[REDACTED - ACCESS DENIED]',
                    'classification': classification,
                    'metadata': {
                        'redacted': True
                    }
                })
                self.log_access_attempt(
                    user_id,
                    item.get('document_id', 'unknown'),
                    success=False,
                    reason=f"Insufficient clearance for classification {item_classification['classification']}"
                )
            else:
                redacted_content.append(item)
                self.log_access_attempt(
                    user_id,
                    item['metadata'].get('document_id', 'unknown'),
                    success=True
                )
        
        return redacted_content
