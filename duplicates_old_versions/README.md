# Duplicates and Old Versions Archive

This directory contains files that were moved during the reorganization to industry best practices folder structure on **May 8, 2026**.

## Purpose

These files are **NOT deleted** yet. They are archived here so we can:
1. Compare them to the versions in `src/` to understand differences
2. Verify nothing important is lost
3. Delete them safely after verification

## Directory Structure

### `root/` - Duplicate Core Modules
Files that duplicate functionality already in `src/`:
- `rag_system.py` - Duplicate of `src/rag_system.py`
- `chunker.py` - Duplicate of `src/document_processing/chunker.py`
- `faiss_db.py` - Duplicate of `src/vector_db/faiss_db.py`
- `abstract_document_processor.py` - Duplicate of `src/tools/abstract_document_processor.py`
- `abstract_pdf_processor.py` - Duplicate of `src/tools/abstract_pdf_processor.py`
- `document_processor.py` - Duplicate of `src/tools/document_processor.py`
- `metadata_validator.py` - Duplicate of `src/tools/metadata_validator.py`
- `pdf_processor.py` - Duplicate of `src/tools/pdf_processor.py`
- `smart_document_processor.py` - Duplicate of `src/tools/smart_document_processor.py`

### `old_apps/` - Obsolete Web Applications
Multiple versions of the web app that were created during development:
- `app.py` - Early version
- `web_app.py` - Large monolithic version (70KB)
- `simple_app.py` - Simplified version
- `direct_app.py` - Direct query version
- `clean_rag_app.py` - Cleaned version
- `basic_side_by_side.py` - Side-by-side comparison UI
- `simple_side_by_side.py` - Simple comparison UI
- `final_side_by_side.py` - Final comparison UI

**Current working app:** `unified_app.py` (will be moved to `src/web/app.py`)

### `old_processors/` - Old Processing Scripts
Test and example scripts that are no longer needed:
- `document_processing_example.py` - Example code
- `test_classification.py` - Classification testing
- `test_clearance.py` - Clearance testing
- `test_docs.py` - Document testing
- `test_claude_models.py` - Model testing (may still be useful)

## Next Steps

1. **Compare files** - Check if root duplicates differ from src/ versions
2. **Verify functionality** - Ensure `unified_app.py` has all features from old apps
3. **Test** - Run the reorganized app to confirm everything works
4. **Delete** - Once verified, delete this entire `duplicates_old_versions/` directory

## Verification Checklist

- [ ] Compare `root/rag_system.py` vs `src/rag_system.py`
- [ ] Compare `root/chunker.py` vs `src/document_processing/chunker.py`
- [ ] Compare `root/faiss_db.py` vs `src/vector_db/faiss_db.py`
- [ ] Verify `unified_app.py` has all needed features
- [ ] Test RAG system works after reorganization
- [ ] Delete this directory

## Date Archived
May 8, 2026

## Reorganization Goal
Move to industry best practices structure:
```
rag-app/
├── src/              ← All source code
├── scripts/          ← Utility scripts
├── data/             ← Data files
├── tests/            ← Unit tests
└── docs/             ← Documentation
```
