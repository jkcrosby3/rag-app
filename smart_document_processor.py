from typing import List, Dict, Any, Optional, Type
from pathlib import Path
from .abstract_document_processor import AbstractDocumentProcessor
from .pdf_processor import PyMuPDFProcessor
from .pdf_processor_alt import PyPDF2Processor
from .pdf_processor_example import PDFPlumberProcessor
import logging

logger = logging.getLogger(__name__)

class SmartDocumentProcessor:
    """
    Smart document processor that automatically selects the best processing method.
    
    This processor evaluates multiple factors to choose the optimal processing method:
    1. File type and format
    2. Document complexity
    3. Required features (tables, images, metadata)
    4. Available system resources
    5. Processing requirements
    
    Available processors and their strengths:
    - PyMuPDF: Best for complex PDFs with images and tables
    - PyPDF2: Good for basic PDF processing in restricted environments
    - pdfplumber: Excellent for documents with tables and structured data
    - pdfminer.six: Ideal for pure Python solutions
    - pdfbox-python: Best for PDF creation and modification
    """
    
    def __init__(self):
        """
        Initialize the smart processor with available processing methods.
        
        The processor will automatically detect which libraries are available and
        create appropriate processors for each.
        """
        self.processors = []
        self._initialize_processors()
    
    def _initialize_processors(self):
        """Initialize available processors based on installed libraries."""
        try:
            self.processors.append(PyMuPDFProcessor())
        except ImportError:
            logger.warning("PyMuPDF not available")
        
        try:
            self.processors.append(PyPDF2Processor())
        except ImportError:
            logger.warning("PyPDF2 not available")
        
        try:
            self.processors.append(PDFPlumberProcessor())
        except ImportError:
            logger.warning("pdfplumber not available")
    
    def _get_best_processor(self, file_path: str, requirements: Dict[str, Any]) -> Optional[AbstractDocumentProcessor]:
        """
        Select the best processor for the given file and requirements.
        
        Args:
            file_path: Path to the document file
            requirements: Dictionary of processing requirements, e.g.:
                {
                    'tables': True/False,      # Need table extraction
                    'images': True/False,      # Need image extraction
                    'metadata': True/False,    # Need metadata extraction
                    'complexity': 'low|medium|high'  # Document complexity
                }
            
        Returns:
            The most suitable processor for the task, or None if no suitable processor is found
        """
        file_extension = Path(file_path).suffix.lower()
        best_processor = None
        best_score = -1
        
        for processor in self.processors:
            # Check if processor can handle the file type
            if not processor.can_process(file_path):
                continue
            
            # Calculate a score based on processor capabilities and requirements
            score = 0
            
            # Add points for matching requirements
            if requirements.get('tables', False) and hasattr(processor, 'extract_tables'):
                score += 2
            
            if requirements.get('images', False) and hasattr(processor, 'extract_images'):
                score += 1
            
            if requirements.get('metadata', False) and hasattr(processor, 'extract_metadata'):
                score += 1
            
            # Adjust score based on document complexity
            if requirements.get('complexity', 'low') == 'high':
                if isinstance(processor, PyMuPDFProcessor):
                    score += 2
                elif isinstance(processor, PDFPlumberProcessor):
                    score += 1
            
            # Update best processor if this one is better
            if score > best_score:
                best_score = score
                best_processor = processor
                
        return best_processor
    
    def process_document(self, file_path: str, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a document using the best available processor.
        
        Args:
            file_path: Path to the document to process
            requirements: Processing requirements (see _get_best_processor)
            
        Returns:
            Dictionary containing processed document data:
            {
                'text': extracted text,
                'metadata': document metadata,
                'tables': extracted tables,
                'images': extracted images,
                'processor': name of the processor used
            }
        """
        processor = self._get_best_processor(file_path, requirements)
        if not processor:
            raise ValueError(f"No suitable processor found for file: {file_path}")
            
        result = {}
        try:
            # Extract required components based on requirements
            if requirements.get('text', True):
                result['text'] = processor.extract_text(file_path)
            
            if requirements.get('metadata', False):
                result['metadata'] = processor.extract_metadata(file_path)
            
            if requirements.get('tables', False):
                result['tables'] = processor.extract_tables(file_path)
            
            if requirements.get('images', False):
                result['images'] = processor.extract_images(file_path)
            
            result['processor'] = processor.__class__.__name__
            return result
            
        except Exception as e:
            logger.error(f"Error processing document: {e}")
            raise
    
    def get_available_processors(self) -> List[str]:
        """Get a list of available processor names."""
        return [p.__class__.__name__ for p in self.processors]
