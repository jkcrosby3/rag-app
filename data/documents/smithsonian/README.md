# Smithsonian Revolutionary War Data

This directory contains historical documents from the Smithsonian Institution related to the American Revolutionary War era.

## Data Structure

### 1. Newspapers Database (1770-1810)
**Files tracked with Git LFS:**
- `newspapers/newspapers_data.json` (1.18 GB) - Consolidated newspaper database
- `newspapers/newspapers_data.parquet` (797 MB) - Same data in Parquet format

**Contains:**
- 65 historical newspapers from Chronicling America
- OCR text from 18th-century printing
- Named entities (people, places, battles)
- Subject classification
- War relevance scoring
- OCR quality assessments

**Individual files (excluded from git):**
- `newspapers/text_files/` - Individual newspaper text files (can be regenerated from database)

---

### 2. Revolutionary War Pension Files
**Archive tracked with Git LFS:**
- `pension_files_archive.zip` (~303 MB compressed) - **25,216 pension files**

**Contains:**
- Complete Revolutionary War pension file collection
- Veteran names, ranks, units
- Service dates and locations
- Pension amounts
- Death information
- Family relationships

**To extract:**
```powershell
# Extract all files
Expand-Archive -Path pension_files_archive.zip -DestinationPath pension_files/

# Or use 7-Zip for faster extraction
7z x pension_files_archive.zip -opension_files/
```

**Enriched metadata (excluded from git):**
- `pension_files/text_files/*.json` - Metadata extraction (99.9% accuracy)
- Can be regenerated using `scripts/pension/enrich_pension_files.py`

---

### 3. American Revolutionary Era Collections
**Archive tracked with Git LFS:**
- `revolutionary_era_archive.zip` (~35 MB compressed) - **25,283 collection items**

**Contains:**
- Smithsonian American Art Museum artifacts
- Historical descriptions and metadata
- Revolutionary War-era cultural items

**To extract:**
```powershell
# Extract all files
Expand-Archive -Path revolutionary_era_archive.zip -DestinationPath american_revolutionary_era_collections/

# Or use 7-Zip
7z x revolutionary_era_archive.zip -oamerican_revolutionary_era_collections/
```

---

## Data Management Strategy

### What's in Git (via LFS):
✅ **3 Large database files:**
1. newspapers_data.json (1.18 GB)
2. newspapers_data.parquet (797 MB)
3. pension_files_archive.zip (303 MB)
4. revolutionary_era_archive.zip (35 MB)

**Total LFS storage:** ~2.3 GB in 4 files

### What's Excluded:
❌ Individual text files (50K+ files, included in archives)
❌ Enriched metadata JSON files (can be regenerated)
❌ Guggenheim books (can be re-downloaded)
❌ Vector databases (generated from source data)
❌ Processing artifacts (cache, chunked, embedded)

---

## Regenerating Data

### Extract Archives After Clone:
```powershell
# From smithsonian/rag-app root directory
cd data/documents/smithsonian

# Extract pension files
Expand-Archive -Path pension_files_archive.zip -DestinationPath .

# Extract revolutionary era collections
Expand-Archive -Path revolutionary_era_archive.zip -DestinationPath .
```

### Regenerate Enriched Metadata:
```powershell
# Enrich pension files (99.9% accuracy)
python scripts/pension/enrich_pension_files.py

# Enrich newspaper files
python scripts/newspaper/enrich_newspaper_files.py
```

### Rebuild Vector Database:
```powershell
# Process documents and build vector DB
python scripts/rebuild_pipeline.py
```

---

## Data Sources

### Original Sources:
- **Pension Files:** Smithsonian API - Revolutionary War Pension Files
- **Newspapers:** Library of Congress - Chronicling America (1770-1810)
- **Collections:** Smithsonian American Art Museum - Revolutionary Era Items

### Download Scripts:
- `scripts/download_rev_war_pension.py` - Download pension files
- `scripts/download_newspapers_*.py` - Download newspapers
- `scripts/download_rev_era_collections.py` - Download collections

---

## Storage Optimization

### Why Archives Instead of Individual Files?
1. **Git Efficiency:** 4 LFS files vs 50K+ individual files
2. **API Rate Limits:** GitHub LFS can't handle 50K files at once
3. **Transfer Speed:** Single compressed file downloads faster
4. **Repository Size:** Better compression than git's internal compression

### Why Keep Individual Files Locally?
1. **Development:** Easier to work with individual files
2. **Metadata Enrichment:** Processing works on individual files
3. **Vector DB:** Requires individual files for chunking
4. **Testing:** Can test on subsets of data

---

## Data Workflow

```
Clone Repo (with LFS)
    ↓
Extract Archives → data/documents/smithsonian/pension_files/text_files/
                   data/documents/smithsonian/american_revolutionary_era_collections/text_files/
    ↓
Generate Metadata → *.json files with enriched data (99.9% accuracy)
    ↓
Build Vector DB → data/vector_db/ (FAISS or Elasticsearch)
    ↓
Ready for Queries!
```

---

## File Size Reference

| Item | Count | Compressed | Uncompressed |
|------|-------|------------|--------------|
| Newspapers JSON | 1 | - | 1.18 GB |
| Newspapers Parquet | 1 | - | 797 MB |
| Pension Files | 25,216 | 303 MB (zip) | ~500 MB |
| Revolutionary Collections | 25,283 | 35 MB (zip) | ~50 MB |
| **Total LFS** | **4 files** | **~2.3 GB** | **~2.5 GB** |

---

## Maintenance

### Creating New Archives:
```powershell
# If you update the data, recreate archives
Compress-Archive -Path "pension_files\text_files" -DestinationPath "pension_files_archive.zip" -CompressionLevel Optimal -Force

Compress-Archive -Path "american_revolutionary_era_collections\text_files" -DestinationPath "revolutionary_era_archive.zip" -CompressionLevel Optimal -Force
```

### Verifying Archive Integrity:
```powershell
# Check archive contents
Expand-Archive -Path pension_files_archive.zip -DestinationPath temp_verify/ -Force
(Get-ChildItem temp_verify -Recurse -File).Count  # Should be 25,216
Remove-Item temp_verify -Recurse -Force
```

---

## Notes

- **Metadata JSON files** are excluded because they can be regenerated with 99.9% accuracy using enrichment scripts
- **Guggenheim books** are excluded because they can be easily re-downloaded from the source
- **Individual text files** are excluded from git but kept locally for development
- **Archives are versioned** in Git LFS for backup and collaboration
