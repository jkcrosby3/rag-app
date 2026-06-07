# RAG Application - Data Cleanup Guide

**Purpose:** Instructions for removing old documents and embeddings to prepare for new data

**Date:** May 7, 2026

---

## Overview

When you need to replace the documents in your RAG system with new data, you must properly clean out:
1. Source documents
2. Vector embeddings (FAISS database)
3. Processing artifacts (chunks, metadata, cache)
4. Document registry

This ensures the system starts fresh and doesn't mix old and new data.

---

## What Gets Cleaned

### Core Data Directories

| Directory | Contents | Purpose |
| --------- | -------- | ------- |
| `data/documents/` | Source PDFs, text files, etc. | Your original documents |
| `data/vector_db/` | FAISS index, lookups, metadata | Vector embeddings database |
| `data/chunked/` | Split document chunks | Intermediate processing |
| `data/embedded/` | Chunks with embeddings | Intermediate processing |
| `data/metadata/` | Document metadata | Processing metadata |
| `data/cache/` | Cached results | Performance optimization |

### Key Files

- **`data/document_registry.json`** - Tracks which documents have been processed
- **`data/vector_db/faiss.index`** - The actual vector database
- **`data/vector_db/document_lookup.pkl`** - Maps vectors to source documents
- **`data/vector_db/metadata.json`** - Database metadata

---

## Cleanup Methods

### Method 1: Automated Python Script (Recommended)

Use the provided cleanup script:

```bash
# Windows
python cleanup_data.py

# With confirmation prompt
python cleanup_data.py --confirm

# Dry run (see what would be deleted without deleting)
python cleanup_data.py --dry-run
```

**Features:**
- Safe deletion with confirmation
- Dry-run mode to preview changes
- Preserves directory structure
- Detailed logging of what was deleted
- Backup option before deletion

---

### Method 2: Manual PowerShell Commands

```powershell
# Navigate to rag-app directory
cd "C:\Users\639250\OneDrive - BOOZ ALLEN HAMILTON\Documents\repos\smithsonian\rag-app"

# Delete source documents
Remove-Item -Recurse -Force "data\documents\*" -ErrorAction SilentlyContinue

# Delete vector database (embeddings)
Remove-Item -Recurse -Force "data\vector_db\*" -ErrorAction SilentlyContinue

# Delete document registry
Remove-Item -Force "data\document_registry.json" -ErrorAction SilentlyContinue

# Delete processing artifacts
Remove-Item -Recurse -Force "data\chunked\*" -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force "data\embedded\*" -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force "data\metadata\*" -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force "data\cache\*" -ErrorAction SilentlyContinue

# Verify cleanup
Get-ChildItem -Path "data" -Recurse | Measure-Object
```

---

### Method 3: Manual Bash Commands (Linux/Mac)

```bash
# Navigate to rag-app directory
cd ~/path/to/rag-app

# Delete all data (preserves directory structure)
rm -rf data/documents/*
rm -rf data/vector_db/*
rm -f data/document_registry.json
rm -rf data/chunked/*
rm -rf data/embedded/*
rm -rf data/metadata/*
rm -rf data/cache/*

# Verify cleanup
find data -type f | wc -l
```

---

## After Cleanup: Adding New Documents

### Step 1: Add Your New Documents

```bash
# Copy your new documents to the data/documents directory
cp /path/to/your/new/documents/* data/documents/
```

### Step 2: Rebuild the Vector Database

The system will automatically rebuild when you start it:

```bash
# Windows
.\start_web_app.bat

# Linux/Mac
./start_web_app.sh
```

Or manually trigger processing:

```bash
# Activate virtual environment first
# Windows
.\venv\Scripts\activate

# Linux/Mac
source venv/bin/activate

# Run processing pipeline
python scripts/process_documents.py
python scripts/chunk_documents.py
python scripts/generate_embeddings.py
python scripts/build_vector_db.py
```

---

## Safety Considerations

### Before Cleanup

1. **Backup important documents** - Copy `data/documents/` to a safe location
2. **Check for custom configurations** - Review any custom settings in `data/`
3. **Note your document count** - Know how many documents you're replacing

### Verification After Cleanup

```powershell
# Check that directories are empty (Windows)
Get-ChildItem -Path "data\documents" -Recurse
Get-ChildItem -Path "data\vector_db" -Recurse

# Should show empty directories
```

```bash
# Check that directories are empty (Linux/Mac)
ls -la data/documents/
ls -la data/vector_db/

# Should show no files
```

---

## Troubleshooting

### Issue: "Access Denied" or "File in Use"

**Solution:**
1. Close all applications using the RAG system
2. Stop any running web servers (check ports 5000, 7860, 8501)
3. Close your IDE if it has the folder open
4. Wait for OneDrive sync to complete
5. Try cleanup again

### Issue: Directories Not Recreating

**Solution:**
The cleanup script preserves directory structure. If directories are missing:

```bash
# Recreate directory structure
mkdir -p data/{documents,vector_db,chunked,embedded,metadata,cache}
```

### Issue: Old Data Still Appearing in Queries

**Solution:**
1. Verify `data/vector_db/` is completely empty
2. Delete `data/document_registry.json`
3. Restart the application
4. Check that new documents are being processed

---

## Cleanup Script Options

### Full Cleanup (Everything)

```bash
python cleanup_data.py --full
```

Deletes:
- All documents
- All embeddings
- All processing artifacts
- Document registry

### Partial Cleanup (Keep Documents)

```bash
python cleanup_data.py --keep-documents
```

Deletes:
- Embeddings only
- Processing artifacts
- Document registry

**Use case:** Reprocess existing documents with different settings

### Selective Cleanup

```bash
# Only delete vector database
python cleanup_data.py --vector-db-only

# Only delete processing artifacts
python cleanup_data.py --artifacts-only
```

---

## Best Practices

1. **Always backup** before cleanup if documents are irreplaceable
2. **Use dry-run first** to see what will be deleted
3. **Close all applications** that might have files open
4. **Verify cleanup** before adding new documents
5. **Document your changes** - note what data you're replacing and why

---

## Quick Reference

```bash
# Full cleanup with confirmation
python cleanup_data.py --confirm

# Dry run (safe preview)
python cleanup_data.py --dry-run

# Keep documents, rebuild embeddings
python cleanup_data.py --keep-documents

# Add new documents
cp /path/to/new/docs/* data/documents/

# Rebuild and start
.\start_web_app.bat  # Windows
./start_web_app.sh   # Linux/Mac
```

---

**Document Created:** May 7, 2026  
**Location:** C:\Users\639250\OneDrive - BOOZ ALLEN HAMILTON\Documents\repos\smithsonian\rag-app  
**Related Files:** `cleanup_data.py`, `start_web_app.bat`, `start_web_app.sh`
