# Implementation Steps for Business Plan Document Organization

This document provides a step-by-step guide for implementing the business plan document organization structure in your RAG application.

## Prerequisites

- Existing RAG application with document processing capabilities
- Python environment with necessary dependencies

## Step 1: Create Directory Structure

Create the following directory structure in your data directory:

```
data/
  documents/
    plans/                      # Main directory for business plans
    templates/                  # Templates for different document types
    semantic_categories/        # Alternative organization by topic
    question_mappings/          # Maps questions to documents containing answers
```

## Step 2: Implement Document Naming Convention

Follow this naming convention for all business plan documents:

```
plan_<NUMBER>-<TYPE>-<STATUS>.pdf

Examples:
plan_234456-main-DRAFT.pdf
plan_234456-annexT-APPROVED.pdf
plan_234456-annexT-appendixG-EXECUTED.pdf
```

For supporting documents:
```
plan_<NUMBER>-supporting-<SUPPORTING_TYPE>.pdf

Example:
plan_234456-supporting-market_analysis.pdf
```

## Step 3: Create Metadata Files

For each document, create a corresponding metadata JSON file with the same name but with `.metadata.json` extension:

```json
{
  "plan_number": "234456",
  "document_type": "main",
  "hierarchy_level": 1,
  "date": "2025-06-13",
  "status": "DRAFT",
  "keywords": ["strategic plan", "objectives", "timeline"],
  "author": "CEO Office",
  "department": "Executive",
  "revision": "1.0"
}
```

## Step 4: Create Relationship Mapping Files

For each plan, create a relationship mapping file that explicitly maps the connections between documents:

```json
{
  "plan_id": "234456",
  "main_document": "plan_234456-main-DRAFT.pdf",
  "status": "DRAFT",
  "annexes": [
    {
      "id": "T",
      "file": "plan_234456-annexT-DRAFT.pdf",
      "department": "IT",
      "appendices": [
        {
          "id": "G",
          "file": "plan_234456-annexT-appendixG-DRAFT.pdf",
          "answers_questions": ["Q1", "Q3"]
        }
      ]
    }
  ]
}
```

## Step 5: Create Question Mapping Files

Create question mapping files that track which documents across different plans answer the same questions:

```json
{
  "question_id": "Q1",
  "question_text": "How will we implement requirement AT1?",
  "answered_in": [
    {"plan": "234456", "document": "plan_234456-annexT-appendixG-DRAFT.pdf"},
    {"plan": "567890", "document": "plan_567890-annexT-appendixG-DRAFT.pdf"}
  ]
}
```

## Step 6: Update Document Processing Pipeline

Extend your existing document processing pipeline to:

1. Extract metadata from filenames and paths
2. Load relationship data
3. Extract document relationships
4. Process hierarchical documents

Use the `BusinessPlanProcessor` class in `scripts/update_document_processor.py` as a reference.

## Step 7: Update Chunking Process

Ensure your chunking process preserves the hierarchical relationships between documents by:

1. Including relationship metadata in each chunk
2. Preserving document hierarchy level information
3. Maintaining references to parent and child documents

## Step 8: Update Vector Database Schema

Update your vector database schema to include the enhanced metadata fields:

- plan_number
- document_type
- hierarchy_level
- status
- relationships

## Step 9: Enhance Retrieval System

Update your retrieval system to leverage the enhanced organization and metadata:

1. Filter by plan number, document type, or status
2. Retrieve related documents based on relationships
3. Answer questions using the question mapping

## Step 10: Test the System

Test the system with sample documents to ensure:

1. Documents are properly processed and chunked
2. Metadata is correctly extracted
3. Relationships are preserved
4. The RAG system can retrieve relevant documents based on the enhanced organization

## Additional Resources

- `docs/DOCUMENT_ORGANIZATION.md` - Detailed document organization plan
- `scripts/create_sample_plan_structure.py` - Script to create sample documents
- `scripts/update_document_processor.py` - Script to update document processing
