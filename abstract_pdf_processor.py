from abc import ABC, abstractmethod
from typing import List, Dict, Any

# This is an abstract base class that defines the interface for all PDF processors
class AbstractPDFProcessor(ABC):
    """
    Base class for all PDF processors. This defines the standard methods that any PDF
    processor must implement. This allows us to easily switch between different PDF
    processing libraries while maintaining consistent functionality.
    
    Think of this as a blueprint that all concrete PDF processors must follow.
    """
    
    @abstractmethod
    def extract_text(self, file_path: str) -> str:
        """
        Extracts readable text from a PDF file.
        
        Args:
            file_path: The path to the PDF file to process
            
        Returns:
            A string containing all the text from the PDF
        """
        pass

    @abstractmethod
    def extract_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        Extracts metadata (like author, creation date, etc.) from a PDF file.
        
        Args:
            file_path: The path to the PDF file to process
            
        Returns:
            A dictionary containing various metadata fields about the PDF
        """
        pass

    @abstractmethod
    def extract_images(self, file_path: str) -> List[str]:
        """
        Extracts images from a PDF file.
        
        Args:
            file_path: The path to the PDF file to process
            
        Returns:
            A list of image file paths or image data extracted from the PDF
        """
        pass
