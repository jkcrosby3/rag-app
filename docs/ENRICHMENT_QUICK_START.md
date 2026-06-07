# Pension File Enrichment - Quick Start Guide

## Current Situation

You have:
- ✅ **12,606 pension text files** already downloaded
- ✅ **pension_files_pdfs.json** with rich metadata (but only 2 records shown)
- ❌ Need to extract veteran info from all 12,606 files

## The JSON File Issue

The `pension_files_pdfs.json` file appears to only have 2 records because:
1. You may have run the download with `--limit 2` or similar
2. The download may have been interrupted
3. The file may be incomplete

**Good news:** The JSON has EXCELLENT metadata including:
- Veteran names in the `title` field
- Ranks and units in `extractedText`
- Service information
- NARA URLs and page counts

## Two Options

### Option 1: Re-download with Full Metadata (Recommended if you want fresh data)

```powershell
# Download ALL pension files with full metadata
python scripts\download_rev_war_pension.py --dataset pdfs --limit 0

# This will:
# - Download the full parquet file
# - Extract ALL records (not just 2)
# - Create text files for RAG
# - Create metadata JSON files with veteran info already extracted
```

**Pros:**
- Gets fresh data from source
- Metadata already structured
- No need for enrichment script

**Cons:**
- Takes time to download
- Requires network access

### Option 2: Enrich Existing Files (Recommended for your situation)

Since you already have 12,606 text files, just extract metadata from them:

```powershell
# Test on 10 files first (dry run)
python scripts\enrich_pension_files.py --dry-run --limit 10

# Test on 100 files (save metadata)
python scripts\enrich_pension_files.py --limit 100

# Process ALL 12,606 files
python scripts\enrich_pension_files.py
```

**Pros:**
- Works with existing files
- No download needed
- Fast (processes ~100 files/second)

**Cons:**
- Extraction may miss some fields
- Requires regex patterns (already built!)

## What the Enrichment Script Extracts

From each pension file text, it extracts:

### Basic Info
- Document ID (NARA record number)
- NARA URL
- Page count

### Veteran Information
- **Name**: "Joseph Torrey"
- **Rank**: "Major"
- **Unit**: "Col. Hazen's Regiment"
- **Pension Number**: "B.L. Wt. 2,153-400"
- **Service Dates**: "1776" - "1783"
- **State of Service**: "New York"
- **State of Residence**: "New York"

### Output Format

Creates `{filename}_metadata.json` for each text file:

```json
{
  "document_id": "111403815",
  "file_name": "111403815.txt",
  "collection": "pension_files",
  "source_system": "NARA",
  "nara_url": "https://catalog.archives.gov/id/111403815",
  "page_count": 5,
  "veteran_name": "Joseph Torrey",
  "rank": "Major",
  "unit": "Col. Hazen's Regiment",
  "pension_number": "B.L. Wt. 2,153-400",
  "service_start": "1776",
  "service_end": "1783",
  "state_of_service": "New York",
  "state_of_residence": "New York",
  "document_type": "pension_application",
  "language": "en",
  "enrichment_date": "2026-05-09T00:20:00Z",
  "enrichment_version": "1.0"
}
```

## Recommended Workflow

### Step 1: Test the Enrichment (5 minutes)

```powershell
# See what it would extract from 10 files (no files saved)
python scripts\enrich_pension_files.py --dry-run --limit 10
```

**Expected output:**
```
📜 PENSION FILE ENRICHMENT
======================================================================
Input directory: C:\...\pension_files\text_files
Dry run: True

✓  Found 12606 text files
✓  Processing first 10 files

⏳ Processing files...

--- Example 1: 111403760.txt ---
  Veteran: Asa Torrey
  Rank: None
  Unit: None
  Service: None - None
  Pension #: None

--- Example 2: 111403815.txt ---
  Veteran: Joseph Torrey
  Rank: Major
  Unit: Col. Hazen's Regiment
  Service: 1776 - 1783
  Pension #: B.L. Wt. 2,153-400

...

📊 ENRICHMENT SUMMARY
======================================================================
Total files: 10
Processed: 10
Errors: 0

Extraction Success Rates:
  Veteran names: 8 (80.0%)
  Ranks: 6 (60.0%)
  Units: 7 (70.0%)
  Service dates: 5 (50.0%)

✓  Dry run complete (no files saved)
```

### Step 2: Process a Sample (10 minutes)

```powershell
# Process first 100 files and save metadata
python scripts\enrich_pension_files.py --limit 100
```

This creates 100 `*_metadata.json` files alongside the text files.

### Step 3: Review Results

```powershell
# Check a metadata file
Get-Content "data\documents\smithsonian\pension_files\text_files\111403815_metadata.json"
```

### Step 4: Process All Files (30-60 minutes)

```powershell
# Process all 12,606 files
python scripts\enrich_pension_files.py
```

**Expected time:** ~10-15 minutes for 12,606 files

### Step 5: Use in RAG System

The metadata files will be automatically picked up when you rebuild the vector database:

```powershell
# Rebuild vector DB with enriched metadata
python scripts\process_documents.py
```

## Troubleshooting

### "No text files found"
**Problem:** Script can't find the files
**Solution:** Check the path
```powershell
# Verify files exist
Get-ChildItem "data\documents\smithsonian\pension_files\text_files\*.txt" | Measure-Object
```

### Low extraction rates
**Problem:** Only extracting 20% of veteran names
**Solution:** The regex patterns may need tuning. Share a few example files and I'll improve the patterns.

### Want to re-download everything
**Problem:** Want fresh data with original metadata
**Solution:** Use the download script
```powershell
# This will take a while but gets everything fresh
python scripts\download_rev_war_pension.py --dataset pdfs --limit 0
```

## Next Steps After Enrichment

1. **Rebuild Vector Database**
   ```powershell
   python scripts\process_documents.py
   ```

2. **Check Statistics Dashboard**
   - Start web app: `python start_web_app.py`
   - Go to "Statistics" tab
   - Should now show veteran counts, ranks, units

3. **Query with Metadata**
   - "Show me majors from New York"
   - "Find captains who served in 1776"
   - "List veterans from Pennsylvania"

## Summary

**Recommended approach for you:**
1. Run enrichment script on existing 12,606 files (fast, no download)
2. Test with `--dry-run --limit 10` first
3. Process all files with `python scripts\enrich_pension_files.py`
4. Rebuild vector database
5. Enjoy enriched metadata in your RAG system! 🚀

**If you want to re-download:**
- The download script command is: `python scripts\download_rev_war_pension.py --dataset pdfs --limit 0`
- But you already have the files, so enrichment is faster!
