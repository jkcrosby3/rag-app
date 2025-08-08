from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class AbstractDocumentProcessor(ABC):
    """
    Base class for all document processors that can handle multiple file types.
    
    This interface defines the standard methods that any document processor must implement.
    It allows us to process various file types (PDF, DOCX, XLSX, etc.) while maintaining
    consistent functionality.
    
    The processor will automatically choose the best processing method based on:
    1. File type
    2. Document complexity
    3. Available system resources
    4. Required features (e.g., table extraction, image processing)
    
    Available processors:
    1. PyMuPDF - Best for general PDF processing, especially when you need:
       - Fast text extraction
       - Complex PDF manipulation
       - Image extraction
       - Table extraction
       - System-level dependencies are allowed
       
    2. PyPDF2 - Good backup option when PyMuPDF isn't available, especially useful when:
       - You need a pure Python solution
       - Basic text extraction is sufficient
       - No system dependencies are allowed
       - Simple PDF processing is needed
       
    3. pdfplumber - Excellent for documents with tables and structured data, especially:
       - Financial documents
       - Government forms
       - Documents with complex tables
       - When precise text positioning matters
       
    4. pdfminer.six - Ideal for restricted environments where:
       - Pure Python solution is required
       - No external dependencies can be installed
       - Basic text extraction is sufficient
       - Lightweight solution is needed
       
    5. pdfbox-python - Best for when you need to:
       - Create or modify PDFs
       - Handle complex PDF features
       - Work with PDF forms
       - Need Apache PDFBox capabilities
    """
    
    @abstractmethod
    def can_process(self, file_path: str) -> bool:
        """
        Check if this processor can handle the given file.
        
        Args:
            file_path: Path to the file to check
            
        Returns:
            True if this processor can handle the file, False otherwise
        """
        pass

    @abstractmethod
    def extract_text(self, file_path: str) -> str:
        """
        Extract text from any supported document type.
        
        The processor will automatically choose the best method based on:
        - Document type
        - Document complexity
        - Required features
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Extracted text as a string
        """
        pass

    @abstractmethod
    def extract_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        Extract metadata from any supported document type.
        
        The processor will automatically choose the best method based on:
        - Document type
        - Available metadata fields
        - Required features
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Dictionary containing document metadata
        """
        pass

    @abstractmethod
    def extract_tables(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Extract tables from any supported document type.
        
        The processor will automatically choose the best method based on:
        - Table complexity
        - Required data accuracy
        - Document type
        
        Args:
            file_path: Path to the document file
            
        Returns:
            List of tables as dictionaries
        """
        pass

    @abstractmethod
    def extract_images(self, file_path: str) -> List[str]:
        """
        Extract images from any supported document type.
        
        The processor will automatically choose the best method based on:
        - Image quality requirements
        - Document type
        - Required image formats
        
        Args:
            file_path: Path to the document file
            
        Returns:
            List of image file paths or image data
        """
        pass
