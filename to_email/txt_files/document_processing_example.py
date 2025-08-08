from tools.smart_document_processor import SmartDocumentProcessor
import logging

logging.basicConfig(level=logging.INFO)

# Create a smart processor instance
processor = SmartDocumentProcessor()

# Example 1: Basic text extraction from a PDF
def process_basic_pdf(pdf_path: str):
    """
    Process a PDF file with basic requirements.
    
    This will use the most appropriate processor based on:
    - File type (PDF)
    - Basic text extraction
    - Available processors
    """
    result = processor.process_document(
        pdf_path,
        requirements={
            'text': True,
            'tables': False,
            'images': False,
            'metadata': False,
            'complexity': 'low'
        }
    )
    print(f"\nProcessed using: {result['processor']}")
    print(f"Text length: {len(result['text'])}")

# Example 2: Complex document processing with tables and images
def process_complex_document(doc_path: str):
    """
    Process a document requiring advanced features.
    
    This will use the most capable processor that can handle:
    - Table extraction
    - Image extraction
    - Metadata extraction
    - Complex document structure
    """
    result = processor.process_document(
        doc_path,
        requirements={
            'text': True,
            'tables': True,
            'images': True,
            'metadata': True,
            'complexity': 'high'
        }
    )
    print(f"\nProcessed using: {result['processor']}")
    print(f"Text length: {len(result['text'])}")
    print(f"Tables found: {len(result.get('tables', []))}")
    print(f"Images found: {len(result.get('images', []))}")

# Example 3: Processing a government document with tables
def process_government_form(form_path: str):
    """
    Process a government form requiring precise table extraction.
    
    This will prioritize processors that excel at table extraction:
    - pdfplumber is preferred for its table handling capabilities
    - Falls back to other processors if pdfplumber isn't available
    """
    result = processor.process_document(
        form_path,
        requirements={
            'text': True,
            'tables': True,
            'images': False,
            'metadata': False,
            'complexity': 'medium'
        }
    )
    print(f"\nProcessed using: {result['processor']}")
    print(f"Text length: {len(result['text'])}")
    print(f"Tables found: {len(result.get('tables', []))}")

# Example usage
if __name__ == "__main__":
    # Print available processors
    print(f"Available processors: {processor.get_available_processors()}")
    
    # Process different types of documents
    process_basic_pdf("example.pdf")
    process_complex_document("complex_document.pdf")
    process_government_form("government_form.pdf")
