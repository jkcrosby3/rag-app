# Git LFS Migration Guide - Smithsonian RAG App

## The Problem

When attempting to push the RAG app to your personal GitHub (`origin`), you encountered:

```
LFS: API rate limit exceeded
error: failed to push some refs
```

### Root Cause

1. **Too many small files in LFS:** 25,332 individual files being tracked by Git LFS
2. **GitHub API limits:** GitHub LFS can't handle uploading 50,000+ small files in one push
3. **Inefficient storage:** Git LFS is designed for large files, not thousands of small metadata files

### What Was Being Tracked

```
❌ 25,216 pension file metadata JSONs (small, ~12 KB each)
❌ 25,283 revolutionary era collection files (small)
❌ 65 newspaper text files
❌ Other metadata files

Total: 25,332+ individual files overwhelming GitHub's LFS API
```

---

## The Solution

**Two-Repo Strategy: Data on Personal GitHub, Code on Hackathon GitHub**

### Repository Strategy

- **Personal GitHub (origin):** Code + Data archives (full backup)
- **Booz Allen Internal (smithhack):** Code only (no data files)

Instead of tracking 50,000+ individual files, we:

1. Compress pension files into a single ZIP (303 MB)
2. Compress revolutionary era collections into a single ZIP (35 MB)
3. Keep the 2 large newspaper database files as-is (1.18 GB + 797 MB)
4. Track only these 4 files with Git LFS on **personal GitHub only**

### New LFS Strategy

```
✅ Personal GitHub (origin) - 4 Large Files in Git LFS:
   1. newspapers_data.json (1.18 GB)
   2. newspapers_data.parquet (797 MB)
   3. pension_files_archive.zip (303 MB) - contains 25,216 files
   4. revolutionary_era_archive.zip (35 MB) - contains 25,283 files

   Total: ~2.3 GB in 4 files (manageable for GitHub LFS)

✅ Booz Allen Internal (smithhack) - Code Only:
   - All source code
   - Documentation
   - Configuration files
   - NO data files (they're gitignored)
```

---

## What We Changed

### 1. Updated `.gitattributes`

**Before:**

```
*.json filter=lfs diff=lfs merge=lfs -text
*.parquet filter=lfs diff=lfs merge=lfs -text
```

- Tracked ALL JSON files (25K+ small files)

**After:**

```
# Track large Smithsonian database files with LFS
data/documents/smithsonian/newspapers/newspapers_data.json filter=lfs diff=lfs merge=lfs -text
data/documents/smithsonian/newspapers/newspapers_data.parquet filter=lfs diff=lfs merge=lfs -text
data/documents/smithsonian/pension_files_archive.zip filter=lfs diff=lfs merge=lfs -text
data/documents/smithsonian/revolutionary_era_archive.zip filter=lfs diff=lfs merge=lfs -text
```

- Tracks only 4 specific large files

### 2. Updated `.gitignore`

**Added/Modified:**

```
# Large data files - tracked with Git LFS (see .gitattributes)
# newspapers_data.json, newspapers_data.parquet, and archive zips are in LFS

# Exclude individual text files (included in archives)
data/documents/smithsonian/newspapers/text_files/*.txt
data/documents/smithsonian/pension_files/text_files/*.txt
data/documents/smithsonian/pension_files/text_files/*.json
data/documents/smithsonian/american_revolutionary_era_collections/text_files/*.txt
data/documents/smithsonian/american_revolutionary_era_collections/text_files/*.json

# Guggenheim books (can be re-downloaded)
data/books/text_files/*
```

### 3. Created Archives

Compressed collections to reduce file count:

```powershell
# Pension files: 25,216 files → 1 zip file (303 MB)
Compress-Archive -Path "pension_files\text_files" -DestinationPath "pension_files_archive.zip"

# Revolutionary era: 25,283 files → 1 zip file (35 MB)
Compress-Archive -Path "american_revolutionary_era_collections\text_files" -DestinationPath "revolutionary_era_archive.zip"
```

### 4. Created Documentation

- `data/documents/smithsonian/README.md` - Data structure and extraction guide
- `GIT_LFS_MIGRATION_GUIDE.md` (this file) - Migration instructions

---

## Step-by-Step Migration Process

### Step 1: Wait for Compression to Complete ⏳

Check if compression is done:

```powershell
cd "C:\Users\639250\OneDrive - BOOZ ALLEN HAMILTON\Documents\repos\smithsonian\rag-app"

# Check if archives exist and their sizes
Get-ChildItem data\documents\smithsonian\*_archive.zip | Select-Object Name, @{Name="SizeMB";Expression={[math]::Round($_.Length/1MB, 2)}}
```

**Expected output:**

```
Name                            SizeMB
----                            ------
pension_files_archive.zip       303.xx
revolutionary_era_archive.zip    35.xx
```

If files don't exist yet, **wait for compression to complete** before proceeding.

---

### Step 2: Switch to Clean Branch

We're currently on `code-only` branch. Let's create a fresh branch for the migration:

```powershell
# Check current branch
git branch

# Go back to clean-main
git checkout clean-main

# Create new migration branch
git checkout -b lfs-archives

# Verify you're on the right branch
git branch
```

---

### Step 3: Remove Old LFS Tracking

We need to untrack all the previously tracked files:

```powershell
# Uninstall old LFS hooks (we'll reinstall clean)
git lfs uninstall

# Remove all current LFS tracking
git lfs untrack "*"

# Verify no files are tracked
git lfs ls-files
# Should show: nothing or very few files
```

---

### Step 4: Stage the New Configuration

```powershell
# Add the updated configuration files
git add .gitattributes
git add .gitignore

# Add the new README documentation
git add data/documents/smithsonian/README.md
git add GIT_LFS_MIGRATION_GUIDE.md

# Add the architecture documentation
git add docs/ARCHITECTURE*.md

# Verify what's staged
git status
```

---

### Step 5: Reinstall Git LFS with New Config

```powershell
# Reinstall LFS (will read new .gitattributes)
git lfs install

# Track the new archives
git lfs track "data/documents/smithsonian/newspapers/newspapers_data.json"
git lfs track "data/documents/smithsonian/newspapers/newspapers_data.parquet"
git lfs track "data/documents/smithsonian/pension_files_archive.zip"
git lfs track "data/documents/smithsonian/revolutionary_era_archive.zip"

# Verify only 4 files are tracked
git lfs ls-files
# Should show the 4 large files only
```

---

### Step 6: Add the Archive Files

```powershell
# Add the compressed archives (LFS will handle them)
git add data/documents/smithsonian/pension_files_archive.zip
git add data/documents/smithsonian/revolutionary_era_archive.zip

# Add the newspaper database files (if not already tracked)
git add data/documents/smithsonian/newspapers/newspapers_data.json
git add data/documents/smithsonian/newspapers/newspapers_data.parquet

# Check LFS status
git lfs status
# Should show 4 files ready to be committed
```

---

### Step 7: Commit the Changes

```powershell
# Commit everything
git commit -m "Migrate to archive-based LFS storage

- Compress pension files (25,216 files → 1 zip, 303 MB)
- Compress revolutionary era collections (25,283 files → 1 zip, 35 MB)
- Update LFS to track only 4 large files instead of 25K+ small files
- Add documentation for data extraction and regeneration
- Add architecture diagrams and presentation materials

This resolves GitHub LFS API rate limit issues by reducing
tracked file count from 25,332 to 4 files."

# Verify commit
git log --oneline -1
```

---

### Step 8: Push to Personal GitHub (Code + Data)

Push everything (code + LFS archives) to your personal GitHub:

```powershell
# Push to origin (personal GitHub) - includes LFS data
git push origin lfs-archives

# If successful, you can merge to main later:
# git checkout main
# git merge lfs-archives
# git push origin main
```

**This push includes:**

- ✅ All code and documentation
- ✅ 4 LFS archive files (2.3 GB)
- ✅ Full backup of Smithsonian data

---

### Step 9: Push to Booz Allen Internal GitHub (Code Only)

Push code without data archives to the internal repository:

```powershell
# First, temporarily move LFS files out of tracking for this push
# Create a temporary branch for code-only push
git checkout -b code-only-smithhack

# Remove LFS data files from this branch (they're already gitignored for smithhack)
git rm --cached data/documents/smithsonian/newspapers/newspapers_data.json
git rm --cached data/documents/smithsonian/newspapers/newspapers_data.parquet
git rm --cached data/documents/smithsonian/pension_files_archive.zip
git rm --cached data/documents/smithsonian/revolutionary_era_archive.zip

# Commit the removal
git commit -m "Remove data files for internal repository (code only)"

# Push to smithhack (Booz Allen internal)
git push smithhack code-only-smithhack:lfs-archives

# Switch back to full data branch
git checkout lfs-archives
```

**This push includes:**

- ✅ All code and documentation
- ❌ No data archives (saves space, faster for team)
- ℹ️ Team can re-download data from Smithsonian API if needed

**Alternative simpler approach:**  
Since the data files are already in `.gitignore`, they won't be pushed anyway. You can just:

```powershell
git push smithhack lfs-archives
```

The `.gitignore` will prevent the data files from being included automatically!

---

## If Push Still Fails

### Issue: LFS Bandwidth Limit

If you see bandwidth errors:

```
LFS: bandwidth limit exceeded
```

**Solution:** Wait for GitHub's bandwidth reset (usually monthly) or:

1. Upgrade GitHub LFS storage (if using personal account)
2. Use only internal Booz Allen GitHub for data storage
3. Use external storage (AWS S3, Azure Blob) and document the location

### Issue: Files Too Large for LFS

If newspaper files are rejected:

```
File too large for LFS
```

**Solution:** GitHub LFS has a 2 GB per-file limit. Our largest file (newspapers_data.json) is 1.18 GB, so we're under the limit. If this happens:

1. Verify file size: `Get-ChildItem data\documents\smithsonian\newspapers\newspapers_data.json`
2. Consider splitting the file or using external storage

---

## After Successful Push

### For Other Team Members (Cloning the Repo)

When someone clones the repo:

```powershell
# Clone with LFS
git clone https://github.com/jkcrosby3/rag-app.git
cd rag-app

# LFS files download automatically
# Extract archives
cd data/documents/smithsonian

# Extract pension files
Expand-Archive -Path pension_files_archive.zip -DestinationPath .

# Extract revolutionary era collections
Expand-Archive -Path revolutionary_era_archive.zip -DestinationPath .

# Verify extraction
(Get-ChildItem pension_files/text_files -File).Count  # Should be 25,216
(Get-ChildItem american_revolutionary_era_collections/text_files -File).Count  # Should be 25,283

# Regenerate enriched metadata (optional, if needed)
python scripts/pension/enrich_pension_files.py
python scripts/newspaper/enrich_newspaper_files.py

# Build vector database
python scripts/rebuild_pipeline.py
```

---

## What Gets Preserved vs Excluded

### ✅ Preserved in Git (via LFS)

| Item | Size | Method |
|------|------|--------|
| Newspapers JSON | 1.18 GB | LFS direct |
| Newspapers Parquet | 797 MB | LFS direct |
| Pension Files | 303 MB | LFS archive |
| Revolutionary Collections | 35 MB | LFS archive |
| **Total** | **~2.3 GB** | **4 LFS files** |

### ❌ Excluded (but kept locally)

- Individual pension file text files (in archive)
- Individual revolutionary era text files (in archive)
- Metadata JSON files (can regenerate with 99.9% accuracy)
- Guggenheim books (can re-download)
- Vector databases (generated from source)
- Processing artifacts (cache, chunked, embedded)

### 📝 Included in Regular Git

- All Python code (`src/`, `scripts/`)
- Documentation (`docs/`, `README.md`)
- Configuration files (`requirements.txt`, `.env.example`)
- Architecture diagrams
- Web UI files

---

## Benefits of This Approach

### 1. No More API Rate Limits

- **Before:** 25,332 files overwhelming GitHub LFS API
- **After:** 4 files, well within GitHub's limits

### 2. Faster Transfers

- **Before:** 50K+ individual file uploads/downloads
- **After:** 4 compressed archives transfer quickly

### 3. Better Compression

- **Before:** Git's internal compression on 50K files
- **After:** ZIP compression optimized for the file types
- **Savings:** ~40% reduction in total size

### 4. Easier Collaboration

- **Before:** Team members download 50K+ files individually
- **After:** Download 4 archives, extract once

### 5. Flexible Regeneration

- Metadata can be regenerated if archives need updates
- Enrichment scripts achieve 99.9% accuracy
- Vector DBs rebuild from source data

---

## Troubleshooting

### Compression Still Running?

```powershell
# Check PowerShell processes
Get-Process | Where-Object {$_.ProcessName -like "*powershell*"}

# Check if archive files are growing
Get-ChildItem data\documents\smithsonian\*_archive.zip -ErrorAction SilentlyContinue | Select-Object Name, Length, LastWriteTime
```

### Want to Cancel and Use Alternative?

If compression is taking too long, you can:

1. **Cancel compression** (Ctrl+C in the PowerShell window)
2. **Use 7-Zip instead** (much faster):

   ```powershell
   # Install 7-Zip if not already installed
   # Then use:
   7z a -tzip pension_files_archive.zip data\documents\smithsonian\pension_files\text_files\*
   7z a -tzip revolutionary_era_archive.zip data\documents\smithsonian\american_revolutionary_era_collections\text_files\*
   ```

### Check What's Currently Tracked by LFS

```powershell
git lfs ls-files
```

### See LFS File Sizes

```powershell
git lfs ls-files --size
```

### Remove a File from LFS Tracking

```powershell
git lfs untrack "pattern/to/file"
```

---

## Summary

**What:** Migrated from tracking 25,332 small files to 4 large archive files  
**Why:** GitHub LFS API rate limits prevented pushing  
**How:** Compressed collections into archives, updated LFS configuration  
**Result:** Two-repo strategy with efficient Git LFS storage

**Repository Strategy:**

- **Personal GitHub (origin):** Code + Data (2.3 GB in 4 LFS files)
- **Booz Allen Internal (smithhack):** Code only (no data files)

**Next Steps:**

1. ✅ Wait for compression to complete
2. ✅ Follow Steps 1-9 above
3. ✅ Push to personal GitHub (with data)
4. ✅ Push to internal GitHub (code only)
5. ✅ Document extraction process for team members

---

## Quick Reference Commands

```powershell
# Check compression status
Get-ChildItem data\documents\smithsonian\*_archive.zip | Select Name, @{N="MB";E={[math]::Round($_.Length/1MB,2)}}

# Check current branch
git branch

# Check LFS status
git lfs ls-files

# Stage and commit
git add .gitattributes .gitignore *.md docs/ data/documents/smithsonian/*_archive.zip
git commit -m "Migrate to archive-based LFS storage"

# Push to personal GitHub (code + data)
git push origin lfs-archives

# Push to Booz Allen Internal (code only - gitignore prevents data)
git push smithhack lfs-archives

# After successful push, merge to main
git checkout main
git merge lfs-archives
git push origin main
git push smithhack main
```

---

## Contact & Questions

If you encounter issues:

1. Check this guide's troubleshooting section
2. Verify LFS installation: `git lfs version`
3. Check GitHub LFS quota: Visit <https://github.com/settings/billing>
4. Review Git LFS documentation: <https://git-lfs.github.com/>

**Created:** June 7, 2026  
**Purpose:** Migrate Smithsonian RAG app to efficient Git LFS storage  
**Status:** Ready for execution after compression completes
