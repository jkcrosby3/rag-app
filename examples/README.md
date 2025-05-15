# RAG App Examples

This directory contains example scripts demonstrating how to use various components of the RAG application.

## Document Processing Examples

### PDF Processing
- `pdf/read_pdf.py`: Demonstrates basic PDF reading capabilities including:
  - Validating PDF files
  - Extracting text content
  - Reading metadata
  - Error handling

### Text Processing (Coming Soon)
- Text file reading and processing
- Document chunking
- Text normalization

## Running Examples

Each example can be run from the project root directory. For instance:

```bash
# Run PDF reader example
python examples/pdf/read_pdf.py

# Run with different PDF file
python examples/pdf/read_pdf.py --file path/to/your.pdf
```

## Directory Structure

```
examples/
├── pdf/              # PDF processing examples
│   └── read_pdf.py   # Basic PDF reading demo
└── text/             # Text processing examples (coming soon)
```