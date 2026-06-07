# Document Processing Examples

This notebook demonstrates the document processing capabilities of the RAG system.

## Setup

First, let's import our components:

```python
from pathlib import Path
from src.document_processing import DocumentReader, TextChunker
```

## 1. Document Reading
### 1.1 Basic File Reading

First, let's create some test files:

```python
# Create test files
def create_test_files():
    test_dir = Path("test_files")
    test_dir.mkdir(exist_ok=True)

    # Create PDF
    from reportlab.pdfgen import canvas
    test_pdf = test_dir / "test.pdf"
    c = canvas.Canvas(str(test_pdf))
    c.drawString(100, 750, "This is a test PDF document.")
    c.save()

    # Create TXT
    test_txt = test_dir / "test.txt"
    test_txt.write_text("This is a test text document.\nIt has multiple lines.")

    return {
        "pdf": test_pdf,
        "txt": test_txt
    }

# Create files and initialize reader
test_files = create_test_files()
reader = DocumentReader()
```

Now let's read different file types:

```python
# Read different file types
pdf_doc = reader.read_file(test_files['pdf'])
txt_doc = reader.read_file(test_files['txt'])

# Examine the results
print("PDF Document:")
print(f"Text: {pdf_doc['text'][:100]}...")
print(f"Metadata: {pdf_doc['metadata']}")

print("\nText Document:")
print(f"Text: {txt_doc['text'][:100]}...")
print(f"Metadata: {txt_doc['metadata']}")
```

### 1.2 Error Handling

Let's see how the reader handles errors:

```python
# Try reading non-existent file
try:
    reader.read_file('nonexistent.pdf')
except FileNotFoundError as e:
    print(f"Expected error: {e}")

# Try reading unsupported file type
try:
    reader.read_file('test.unsupported')
except ValueError as e:
    print(f"Expected error: {e}")
```

## 2. Text Chunking
### 2.1 Basic Chunking

Let's try basic chunking with default settings:

```python
# Initialize chunker
chunker = TextChunker(chunk_size=1000, chunk_overlap=200)

# Create chunks from PDF document
chunks = chunker.create_chunks(pdf_doc['text'], pdf_doc['metadata'])

print(f"Created {len(chunks)} chunks")
print("\nFirst chunk:")
print(f"Text: {chunks[0]['text'][:100]}...")
print(f"Metadata: {chunks[0]['metadata']}")
```

### 2.2 Chunking Strategies

Now let's explore different chunking strategies:

```python
# Try different chunking approaches
def show_chunks(chunks):
    print(f"Total chunks: {len(chunks)}")
    for i, chunk in enumerate(chunks[:2]):
        print(f"\nChunk {i+1}:")
        print(f"Text: {chunk['text'][:100]}...")
        print(f"Metadata: {chunk['metadata']}")

# With page boundaries
page_chunks = chunker._chunk_by_pages(
    pdf_doc['text'],
    pdf_doc['metadata']['page_boundaries']
)
print("Page-based chunks:")
show_chunks(page_chunks)

# With natural boundaries
natural_chunks = []
for chunk in chunks:
    end = chunker._find_natural_boundary(
        chunk['text'],
        chunk['start'],
        pdf_doc['metadata']
    )
    chunk['end'] = end
    natural_chunks.append(chunk)

print("\nNatural boundary chunks:")
show_chunks(natural_chunks)
```

## 3. Metadata Usage
### 3.1 Document Structure

Let's examine the document structure metadata:

```python
# Extract document hierarchy
hierarchy = reader._extract_document_hierarchy(pdf_doc['text'])
print("Document hierarchy:")
print(hierarchy)

# Extract text types
text_types = reader._extract_text_types(pdf_doc['text'])
print("\nText types:")
print(text_types)
```

### 3.2 Chunk Context

See how metadata improves chunk context:

```python
# Get chunk with its context
def show_chunk_context(chunk):
    print(f"Chunk text: {chunk['text'][:100]}...")
    print("\nContext:")
    meta = chunk['metadata']
    if 'hierarchy' in meta:
        print(f"Section: {meta['hierarchy'].get('section')}")
    if 'text_types' in meta:
        print(f"Text type: {meta['text_types']}")
    if 'local_context' in meta:
        print(f"Previous: {meta['local_context'].get('previous', '')[:50]}...")
        print(f"Next: {meta['local_context'].get('next', '')[:50]}...")

# Show context for first chunk
print("First chunk context:")
show_chunk_context(chunks[0])
```
