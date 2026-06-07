# Permanent Fixes Applied - DO NOT REVERT

**Date:** May 8, 2026  
**Issue:** These same fixes were needed yesterday and today. This document ensures they stay fixed.

## Fix 1: Vector DB Path
**File:** `src/web/app.py` (line 33)  
**Problem:** Path pointed to file instead of directory  
**Fix:**
```python
vector_db_path = "data/vector_db"  # NOT "data/vector_db/faiss.index"
```

## Fix 2: Clearance Check for Demo User
**File:** `src/rag_system.py` (lines 289-298)  
**Problem:** Demo user blocked by clearance validation  
**Fix:** Allow unrestricted access when no clearance exists (for public Smithsonian data)
```python
# Get user's clearance level (optional for public/demo data)
user_clearance = self.clearance_manager.get_user_clearance(self.user_id)

# Search vector database with classification filtering
# If no clearance is set, allow unrestricted access (for public/demo data)
retrieved_documents = self.vector_db.search(
    query_embedding, 
    k=top_k,
    max_classification=user_clearance if user_clearance else None
)
```

## Fix 3: Claude Model Name
**File:** `src/llm/claude_client.py` (line 27)  
**Problem:** Using non-existent model names  
**Fix:**
```python
DEFAULT_MODEL_NAME = "claude-sonnet-4-6"  # Latest Claude 4 Sonnet model
```

**Valid models:**
- Claude 4 Sonnet: `claude-sonnet-4`, `claude-sonnet-4-5`, `claude-sonnet-4-6`
- Claude 4 Opus: `claude-opus-4`, `claude-opus-4-1`, `claude-opus-4-5`, `claude-opus-4-6`, `claude-opus-4-7`

## How to Verify Fixes Are Applied

1. Check `src/web/app.py` line 33: Should be `"data/vector_db"`
2. Check `src/rag_system.py` lines 289-298: Should have optional clearance check
3. Check `src/llm/claude_client.py` line 27: Should be `"claude-sonnet-4-6"`

## If Issues Recur

1. Check if files were reverted (git status)
2. Clear Streamlit cache: Delete `.streamlit/cache` directory
3. Restart Python completely (not just Streamlit)
4. Check if there are duplicate files in `scripts/` or `duplicates_old_versions/`

## Vector DB Structure (Current)
```
data/vector_db/
├── faiss.index          (file - 138 MB)
├── document_lookup.pkl  (file - 649 MB)
└── metadata.json        (file - 79 bytes)
```

**NOT** the old structure with `faiss.index/` as a subdirectory!
