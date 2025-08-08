"""
Simple chunker that handles classification metadata.
"""
import logging
from typing import List, Dict, Optional, Any

class ClassificationChunker:
    """Chunker that handles classification metadata."""
    
    def __init__(self):
        """Initialize the chunker."""
        self.logger = logging.getLogger(__name__)
    
    def chunk_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Split text into chunks with classification metadata.
        
        Args:
            text: Text to chunk
            metadata: Base metadata to apply to all chunks
            
        Returns:
            List of chunks with text and metadata
        """
        if not text:
            return []
            
        # Split text into paragraphs
        paragraphs = text.split('\n\n')
        
        # Initialize chunks
        chunks = []
        current_chunk = ""
        current_metadata = metadata or {}
        
        # Process each paragraph
        for paragraph in paragraphs:
            # Skip empty paragraphs
            if not paragraph.strip():
                continue

            # Extract classification if present
            classification = None
            # Look for classification markers in parentheses (case-insensitive)
            if "(U)" in paragraph or "(u)" in paragraph:
                classification = {
                    "classification": "U",
                    "components": []
                }
            elif "(C)" in paragraph or "(c)" in paragraph:
                classification = {
                    "classification": "C",
                    "components": []
                }
            elif "(S)" in paragraph or "(s)" in paragraph:
                classification = {
                    "classification": "S",
                    "components": []
                }
            elif "(TS)" in paragraph or "(ts)" in paragraph:
                classification = {
                    "classification": "TS",
                    "components": []
                }
            
            # If no parentheses marker found, check for regular text
            if not classification:
                if "unclassified" in paragraph.lower():
                    classification = {
                        "classification": "U",
                        "components": []
                    }
                elif "confidential" in paragraph.lower():
                    classification = {
                        "classification": "C",
                        "components": []
                    }
                elif "secret" in paragraph.lower():
                    classification = {
                        "classification": "S",
                        "components": []
                    }
                elif "top secret" in paragraph.lower():
                    classification = {
                        "classification": "TS",
                        "components": []
                    }

            # Create chunk with text and metadata
            chunk = {
                "text": paragraph,
                "classification": classification or {
                    "classification": "U",
                    "components": []
                },
                "metadata": {
                    **current_metadata  # Copy base metadata
                }
            }
            chunks.append(chunk)

            # Add paragraph to current chunk
            if current_chunk:
                current_chunk += '\n\n' + paragraph
            else:
                current_chunk = paragraph

            # Add chunk immediately if it contains classification information
            if 'classification' in current_metadata:
                chunks.append({
                    'text': current_chunk,
                    'metadata': current_metadata.copy()  # Copy metadata to avoid overwriting
                })
                current_chunk = ""
                current_metadata = metadata or {}  # Reset metadata

        # Add final chunk if not empty
        if current_chunk.strip():
            chunks.append({
                'text': current_chunk,
                'metadata': current_metadata.copy()  # Copy metadata to avoid overwriting
            })

        # Ensure all chunks have a classification
        for chunk in chunks:
            if "classification" not in chunk["metadata"]:
                chunk["metadata"]["classification"] = "U"  # Default to Unclassified if not specified
        
        return chunks
