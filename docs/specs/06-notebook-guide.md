# Notebook Guide

This guide explains how to use and maintain the Jupyter notebooks in this project.

## Overview

The notebooks serve as interactive examples and documentation for the RAG system:
1. `01_document_processing.ipynb`: Document handling and chunking
2. `02_vector_store.ipynb`: Vector storage and retrieval
3. `03_rag_chain.ipynb`: RAG implementation
4. `04_semantic_search.ipynb`: Search functionality

## Notebook Structure

Each notebook follows this structure:
1. Setup and imports
2. Example data creation
3. Feature demonstrations
4. Error handling examples

## Maintaining Notebooks

To keep notebooks in sync with production code:

1. **Test Data**
   - Use `create_test_files()` from `04-code-examples.md`
   - Keep test data minimal and focused

2. **Imports**
   - Import from production code (`src/`)
   - Don't implement logic in notebooks

3. **Examples**
   - Demonstrate one feature per section
   - Include both success and error cases
   - Show metadata usage

4. **Documentation**
   - Reference relevant production code
   - Explain key concepts
   - Link to technical design docs

## Running Notebooks

```bash
# Create a new environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install jupyter

# Start Jupyter
jupyter notebook
```

## Example Updates

When updating production code:
1. Update corresponding notebook sections
2. Run all cells to verify
3. Clear all outputs before committing
4. Update implementation guide if needed
