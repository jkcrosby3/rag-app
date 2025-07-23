"""
Security configuration and validation for document metadata.

This module implements a multi-layered security model for document classification:

1. Classification Levels (Overall Document Classification):
   - public: General information accessible to everyone
   - confidential: Internal information requiring basic access control
   - secret: Sensitive information requiring strict access control
   - top_secret: Most sensitive information with very limited access
   - restricted: Specialized information with custom access rules

2. Security Levels (Technical Protection Requirements):
   - low: Basic protection measures
     * Standard encryption
     * Regular backups
     * Basic access controls
   - medium: Enhanced protection
     * Stronger encryption
     * Regular security audits
     * More restrictive access controls
     * Multi-factor authentication
   - high: High protection
     * Advanced encryption
     * Continuous monitoring
     * Strict access controls
     * Regular security assessments
     * Data masking/anonymization
   - critical: Maximum protection
     * Military-grade encryption
     * Air-gapped storage
     * Biometric authentication
     * Real-time monitoring
     * Data tokenization

3. Sensitivity Levels (Business Impact):
   - low: Minimal business impact if exposed
     * Publicly available information
     * Non-sensitive business data
   - moderate: Moderate business impact if exposed
     * Internal business processes
     * Customer contact information
   - high: Significant business impact if exposed
     * Financial data
     * Personal health information
     * Strategic plans
   - extreme: Catastrophic business impact if exposed
     * Trade secrets
     * Source code
     * Critical infrastructure information

These levels work together to provide a comprehensive security framework:
- Classification determines who can access the information
- Security level determines how the information is protected
- Sensitivity determines the potential impact of exposure

The combination of these three factors helps determine:
1. Access control requirements
2. Encryption requirements
3. Storage requirements
4. Audit and monitoring requirements
5. Backup and recovery requirements
"""
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class SecurityConfig:
    """Configuration for document classification levels."""
    
    # Standard classification levels (lowest to highest)
    CLASSIFICATION_LEVELS = {
        "public": 0,        # Publicly accessible information
        "internal": 1,     # Internal business information
        "confidential": 2, # Sensitive business information
        "restricted": 3    # Restricted access information
    }
    
    # Default access groups
    DEFAULT_ACCESS_GROUPS = [
        "public",
        "internal",
        "restricted",
        "admin"
    ]
    
    @staticmethod
    def validate_classification(classification: str) -> bool:
        """Validate if classification is valid."""
        return classification.lower() in SecurityConfig.CLASSIFICATION_LEVELS
    
    @staticmethod
    def validate_access_groups(groups: List[str]) -> bool:
        """Validate access groups."""
        if not isinstance(groups, list):
            return False
        
        invalid_groups = [g for g in groups if not isinstance(g, str)]
        if invalid_groups:
            logger.error(f"Invalid access groups: {invalid_groups}")
            return False
            
        return True
    
    @staticmethod
    def get_classification_hierarchy() -> Dict[str, int]:
        """Get classification hierarchy."""
        return SecurityConfig.CLASSIFICATION_LEVELS
    
    @staticmethod
    def get_default_access_groups() -> List[str]:
        """Get default access groups."""
        return SecurityConfig.DEFAULT_ACCESS_GROUPS
