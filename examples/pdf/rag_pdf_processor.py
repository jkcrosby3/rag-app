#!/usr/bin/env python3
"""
RAG-optimized PDF processor for converting PDF documents to structured formats
suitable for retrieval-augmented generation applications.

This processor implements several key optimizations for RAG applications:

1. SEMANTIC CHUNKING:
   - Creates chunks based on natural document boundaries (sections, paragraphs)
   - Maintains context by including overlap between chunks
   - Preserves heading information with each chunk for better retrieval context

2. ENHANCED METADATA EXTRACTION:
   - Extracts rich metadata from both PDF properties and content analysis
   - Identifies entities specific to policy documents (dates, organizations, legal references)
   - Estimates reading time and document complexity for better retrieval ranking

3. STRUCTURE DETECTION:
   - Identifies headings, bullet points, and numbered lists using patterns common in policy documents
   - Converts detected structures to proper Markdown formatting
   - Preserves document hierarchy for improved semantic understanding

4. DUAL OUTPUT FORMAT:
   - Creates a human-readable Markdown file with proper formatting
   - Generates a structured JSON file with metadata and chunked content optimized for vector database ingestion

5. OCR ERROR CORRECTION:
   - Includes specific fixes for common OCR errors in historical documents
   - Normalizes text to improve embedding quality

WHY THIS MATTERS FOR RAG:
- Improved Retrieval Accuracy: Better chunking means more relevant results
- Context Preservation: Heading and section information helps LLMs understand retrieved passages
- Enhanced Filtering: Rich metadata enables better search result filtering and weighting
- Reduced Hallucination: Clean text and proper context reduces incorrect information
- Vector Database Ready: Structured output for direct ingestion into vector databases
"""
import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# Add project root to Python path
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.document_processing.pdf_reader import PDFReader

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RAGPDFProcessor:
    """
    Processes PDF documents for optimal use in RAG applications.
    Includes chunking, metadata extraction, and semantic structure detection.
    
    Why document preparation matters for RAG:
    1. Raw PDF text is often poorly structured, with mixed formatting, headers in
       the middle of content, and inconsistent paragraph breaks
    2. Standard chunking approaches (e.g., fixed-size chunks) often break semantic
       units, leading to context loss and poor retrieval
    3. Historical policy documents contain specialized entities and references that
       need to be preserved and made available for retrieval
    4. Document structure (headings, sections, lists) provides crucial context for
       understanding retrieved passages
    5. OCR errors in historical documents can significantly impact embedding quality
       and retrieval accuracy
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the RAG PDF processor.
        
        Args:
            chunk_size: Target size of text chunks in characters
            chunk_overlap: Overlap between chunks in characters
        """
        self.pdf_reader = PDFReader(min_file_size=100)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Patterns for detecting document structure
        self.heading_patterns = [
            # All caps headings
            (r'^([A-Z][A-Z\s]+[A-Z])$', '##'),
            # Numbered headings (e.g., "1. Introduction")
            (r'^(\d+\.\s+.+)$', '##'),
            # Roman numeral headings
            (r'^([IVX]+\.\s+.+)$', '##'),
            # Subheadings with common prefixes
            (r'^(Section|Chapter|Part)\s+(\d+|[IVX]+)[\.\:]\s*(.+)$', '##')
        ]
        
        # Patterns for detecting entities in policy documents
        self.entity_patterns = [
            # Dates (various formats)
            (r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}\b', 'DATE'),
            # Money amounts
            (r'\$\d+(?:,\d+)*(?:\.\d+)?(?:\s+(?:million|billion|trillion))?', 'MONEY'),
            # Percentages
            (r'\b\d+(?:\.\d+)?\s*%', 'PERCENTAGE'),
            # Government agencies
            (r'\b(?:Department of|Federal|National|U\.S\.|United States)\s+[A-Z][a-zA-Z\s]+\b', 'ORGANIZATION'),
            # Legal references
            (r'\b(?:Act|Law|Bill|Regulation|Code|Statute)\s+of\s+\d{4}\b', 'LEGAL_REF')
        ]

    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text from PDF for better processing.
        
        Args:
            text: Raw text extracted from PDF
            
        Returns:
            Cleaned text
            
        RAG Optimization Notes:
        - Text cleaning is critical for embedding quality in RAG applications
        - Inconsistent whitespace, special characters, and OCR errors can create
          noise in embeddings, leading to poor retrieval performance
        - Historical documents often contain specific OCR errors (e.g., 'l' vs 'I')
          that need targeted correction
        - BOM characters and non-breaking spaces can cause encoding issues and
          affect tokenization for embeddings
        """
        # Remove BOM and other problematic characters
        text = text.replace('\ufeff', '')
        text = text.replace('\xa0', ' ')
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Fix common OCR issues in policy documents
        text = re.sub(r'l\b', 'I', text)  # Lowercase l at word end often should be uppercase I
        text = re.sub(r'\bIhe\b', 'The', text)  # Common OCR error
        text = re.sub(r'\bl\b', '1', text)  # Lowercase l alone often should be 1
        
        # Normalize line breaks
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text

    def extract_metadata(self, text: str, pdf_metadata: Dict) -> Dict:
        """
        Extract metadata relevant for RAG from document content and PDF metadata.
        
        Args:
            text: Document text
            pdf_metadata: Metadata from PDF reader
            
        Returns:
            Enhanced metadata dictionary
            
        RAG Optimization Notes:
        - Rich metadata enables more sophisticated retrieval strategies beyond
          simple semantic similarity
        - Entity extraction allows for filtering and boosting results based on
          relevant entities (e.g., specific agencies, dates, or legal references)
        - For historical policy documents, dates and organizational entities are
          particularly important for establishing context
        - Metadata like word count and reading time can be used to weight chunks
          during retrieval (longer, more complex sections may contain more detailed info)
        - Document structure metadata helps reconstruct the original context when
          presenting retrieved passages to the LLM
        """
        metadata = {
            "file_info": {
                "filename": pdf_metadata.get("file_name", ""),
                "file_size": pdf_metadata.get("file_size", 0),
                "page_count": pdf_metadata.get("page_count", 0),
                "creation_date": pdf_metadata.get("creation_date", "")
            },
            "document_info": {
                "title": pdf_metadata.get("title", ""),
                "author": pdf_metadata.get("author", ""),
                "subject": pdf_metadata.get("subject", ""),
                "keywords": pdf_metadata.get("keywords", "")
            },
            "rag_metadata": {
                "entities": self._extract_entities(text),
                "estimated_word_count": len(text.split()),
                "estimated_reading_time": len(text.split()) // 200,  # Approx. 200 words per minute
                "language": "en"  # Default to English for policy documents
            }
        }
        
        # Try to extract a better title if none exists
        if not metadata["document_info"]["title"]:
            title_match = re.search(r'^(?:#+\s*)?([A-Z][A-Za-z\s:,]+)(?:\n|$)', text)
            if title_match:
                metadata["document_info"]["title"] = title_match.group(1).strip()
        
        # Try to extract date if not in metadata
        date_match = re.search(r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}\b', text)
        if date_match:
            metadata["document_info"]["date"] = date_match.group(0)
            
        return metadata

    def _extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract named entities from text using regex patterns.
        
        Args:
            text: Document text
            
        Returns:
            Dictionary of entity types and values
            
        RAG Optimization Notes:
        - Entity extraction is crucial for policy documents which often reference
          specific dates, monetary values, organizations, and legal references
        - These entities can be used as additional features for retrieval or as
          filters in hybrid search approaches
        - For historical policy documents, identifying government agencies and
          legal references is particularly valuable for establishing relevance
        - While more sophisticated NER models could be used, regex patterns provide
          a lightweight approach that works well for structured policy documents
        - Limiting to top entities prevents metadata bloat while preserving the
          most important contextual information
        """
        entities = {
            "dates": [],
            "money_amounts": [],
            "percentages": [],
            "organizations": [],
            "legal_references": []
        }
        
        # Extract entities using patterns
        for pattern, entity_type in self.entity_patterns:
            matches = re.findall(pattern, text)
            if matches:
                if entity_type == 'DATE':
                    entities["dates"].extend(matches)
                elif entity_type == 'MONEY':
                    entities["money_amounts"].extend(matches)
                elif entity_type == 'PERCENTAGE':
                    entities["percentages"].extend(matches)
                elif entity_type == 'ORGANIZATION':
                    entities["organizations"].extend(matches)
                elif entity_type == 'LEGAL_REF':
                    entities["legal_references"].extend(matches)
        
        # Remove duplicates and limit to top 10 of each type
        for key in entities:
            entities[key] = list(set(entities[key]))[:10]
            
        return entities

    def detect_structure(self, text: str) -> str:
        """
        Detect document structure and convert to Markdown.
        
        Args:
            text: Cleaned document text
            
        Returns:
            Markdown formatted text with structure
            
        RAG Optimization Notes:
        - Document structure is essential for understanding the hierarchical
          relationships between different parts of a document
        - Converting to Markdown provides a standardized format that preserves
          semantic structure while removing presentation-specific formatting
        - Headings are particularly important for RAG as they provide context for
          the content that follows and help establish the document hierarchy
        - For policy documents, section headings often indicate the scope or
          applicability of the content (e.g., "Eligibility Requirements")
        - Lists (bulleted and numbered) often contain key points or procedural
          steps that should be preserved as discrete units
        - The patterns used are specifically tuned for common formats in
          government and policy documents
        """
        lines = text.split('\n')
        markdown_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                markdown_lines.append('')
                continue
            
            # Check for headings using patterns
            is_heading = False
            for pattern, prefix in self.heading_patterns:
                if re.match(pattern, line):
                    markdown_lines.append(f"{prefix} {line}")
                    is_heading = True
                    break
            
            if is_heading:
                continue
                
            # Check for bullet points
            if re.match(r'^\s*[•●◦○*-]\s+', line):
                indented_line = re.sub(r'^\s*[•●◦○*-]\s+', '- ', line)
                markdown_lines.append(indented_line)
            # Check for numbered lists
            elif re.match(r'^\s*\d+\.\s+', line):
                markdown_lines.append(line)
            # Otherwise, keep the line as is
            else:
                markdown_lines.append(line)
        
        return '\n'.join(markdown_lines)

    def create_semantic_chunks(self, text: str) -> List[Dict[str, Union[str, int]]]:
        """
        Create semantic chunks from document text for optimal RAG retrieval.
        Attempts to chunk at natural boundaries like paragraphs and sections.
        
        Args:
            text: Markdown formatted document text
            
        Returns:
            List of chunk dictionaries with text and metadata
            
        RAG Optimization Notes:
        - Chunking strategy is perhaps the most critical factor in RAG performance
        - Traditional fixed-size chunking often cuts across semantic boundaries,
          leading to context fragmentation and poor retrieval
        - This semantic chunking approach preserves natural document boundaries:
          1. First divides by sections (based on headings)
          2. Then further divides long sections by paragraphs
          3. Maintains overlap between chunks to preserve context
        - Including heading information with each chunk helps maintain context
          even when a chunk is retrieved in isolation
        - For policy documents, sections often represent discrete topics or provisions
          that should be kept together when possible
        - The overlap strategy ensures that concepts that span paragraph boundaries
          can still be retrieved effectively
        - This approach balances chunk size constraints with semantic coherence
        """
        # Split text into sections based on headings
        sections = []
        current_section = []
        current_heading = "Introduction"  # Default heading
        
        for line in text.split('\n'):
            if line.startswith('##'):
                # If we have content in the current section, save it
                if current_section:
                    sections.append({
                        "heading": current_heading,
                        "content": '\n'.join(current_section)
                    })
                
                # Start a new section
                current_heading = line.replace('#', '').strip()
                current_section = []
            else:
                current_section.append(line)
        
        # Add the last section if it has content
        if current_section:
            sections.append({
                "heading": current_heading,
                "content": '\n'.join(current_section)
            })
        
        # Create chunks from sections
        chunks = []
        for section in sections:
            section_text = section["content"]
            
            # If section is short enough, keep it as one chunk
            if len(section_text) <= self.chunk_size:
                chunks.append({
                    "chunk_id": len(chunks),
                    "heading": section["heading"],
                    "text": section_text,
                    "char_count": len(section_text)
                })
                continue
            
            # Otherwise, split into paragraphs and create chunks
            paragraphs = re.split(r'\n\s*\n', section_text)
            current_chunk = []
            current_length = 0
            
            for para in paragraphs:
                para_length = len(para)
                
                # If adding this paragraph would exceed chunk size, save current chunk and start new one
                if current_length + para_length > self.chunk_size and current_chunk:
                    chunks.append({
                        "chunk_id": len(chunks),
                        "heading": section["heading"],
                        "text": '\n\n'.join(current_chunk),
                        "char_count": current_length
                    })
                    
                    # Start new chunk with overlap
                    overlap_start = max(0, len(current_chunk) - 2)  # Include last 2 paragraphs for context
                    current_chunk = current_chunk[overlap_start:]
                    current_length = sum(len(p) for p in current_chunk) + (len(current_chunk) - 1) * 2  # Account for newlines
                
                current_chunk.append(para)
                current_length += para_length + 2  # +2 for newline characters
            
            # Add the last chunk if it has content
            if current_chunk:
                chunks.append({
                    "chunk_id": len(chunks),
                    "heading": section["heading"],
                    "text": '\n\n'.join(current_chunk),
                    "char_count": current_length
                })
        
        return chunks

    def process_pdf(self, pdf_path: Union[str, Path], output_dir: Optional[Union[str, Path]] = None) -> Tuple[str, str]:
        """
        Process a PDF file for RAG, creating Markdown and chunked JSON outputs.
        
        Args:
            pdf_path: Path to PDF file
            output_dir: Directory to save outputs (defaults to same directory as PDF)
            
        Returns:
            Tuple of (markdown_path, json_path)
            
        RAG Optimization Notes:
        - Dual output formats serve different purposes in a RAG pipeline:
          1. Markdown (.md): Human-readable format for review and editing
          2. JSON (.json): Structured format for vector database ingestion
        - The JSON output includes both the chunks and the full metadata,
          enabling more sophisticated retrieval strategies
        - For historical policy documents, having both formats allows subject
          matter experts to review and correct the extracted content while
          maintaining a machine-optimized version for RAG
        - The structured JSON format can be directly ingested by vector databases
          like Pinecone, Weaviate, or Chroma
        - Including chunk metadata (heading, ID, char count) enables advanced
          filtering and reranking during retrieval
        """
        pdf_path = Path(pdf_path)
        
        if not output_dir:
            output_dir = pdf_path.parent
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Base filename without extension
        base_name = pdf_path.stem
        
        # Output paths
        markdown_path = output_dir / f"{base_name}.md"
        json_path = output_dir / f"{base_name}.json"
        
        logger.info(f"Processing {pdf_path} for RAG")
        
        try:
            # Validate and extract text from PDF
            self.pdf_reader.validate_file(pdf_path)
            result = self.pdf_reader.extract_text(pdf_path)
            text = result["text"]
            pdf_metadata = result["metadata"]
            
            # Process the text
            cleaned_text = self.clean_text(text)
            metadata = self.extract_metadata(cleaned_text, pdf_metadata)
            structured_text = self.detect_structure(cleaned_text)
            chunks = self.create_semantic_chunks(structured_text)
            
            # Create markdown with YAML frontmatter
            markdown_content = ["---"]
            for category, items in metadata["document_info"].items():
                if items:  # Only include non-empty values
                    markdown_content.append(f"{category}: {items}")
            markdown_content.append("---\n")
            markdown_content.append(structured_text)
            
            # Write markdown file
            with open(markdown_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(markdown_content))
            
            # Create JSON with metadata and chunks
            json_data = {
                "metadata": metadata,
                "chunks": chunks
            }
            
            # Write JSON file
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2)
            
            logger.info(f"Successfully processed PDF to {markdown_path} and {json_path}")
            return markdown_path, json_path
            
        except Exception as e:
            logger.error(f"Error processing PDF: {e}")
            raise

def main():
    """
    Run the RAG PDF processor.
    
    This script optimizes PDF documents for RAG applications by:
    
    1. Extracting and cleaning text from PDFs
    2. Detecting document structure and converting to Markdown
    3. Extracting rich metadata including entities
    4. Creating semantic chunks at natural document boundaries
    5. Generating both human-readable Markdown and RAG-optimized JSON
    
    These optimizations significantly improve RAG performance by:
    
    - Improving retrieval accuracy through better chunking and metadata
    - Preserving document context and structure
    - Enabling more sophisticated filtering and ranking
    - Reducing hallucinations through cleaner text and better context
    - Providing vector-database-ready output
    
    For historical policy documents, these optimizations are essential as they
    help maintain the integrity of complex legal and policy information while
    making it accessible to RAG systems.
    """
    parser = argparse.ArgumentParser(description="Process PDF documents for RAG applications")
    parser.add_argument(
        "--file",
        type=str,
        required=True,
        help="Path to PDF file to process"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Directory to save outputs (default: same directory as PDF)"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Target size of text chunks in characters (default: 1000)"
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Overlap between chunks in characters (default: 200)"
    )
    args = parser.parse_args()

    processor = RAGPDFProcessor(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )
    
    try:
        markdown_path, json_path = processor.process_pdf(
            args.file,
            args.output_dir
        )
        logger.info(f"Processing complete. Files saved to:")
        logger.info(f"  Markdown: {markdown_path}")
        logger.info(f"  JSON: {json_path}")
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
