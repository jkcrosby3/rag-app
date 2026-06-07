# Gutenberg Books for Smithsonian Hackathon RAG Application

**Purpose:** Supplement pension files and newspapers with historical books  
**Source:** Project Gutenberg (public domain)  
**Date Created:** May 12, 2026  
**Total Books Cataloged:** 47 books  
**Prioritized Books:** 16 books  
**Recommended for Download:** 10 books (~30-35 MB)

---

## 📁 Files in This Directory

### **Core Files:**

- **`book_catalog.json`** - Structured catalog with metadata and boolean flags
- **`PRIORITIZED_BOOK_LIST.md`** - Human-readable prioritization with full details
- **`list_of_american_revolution_books.txt`** - Original source list
- **`README.md`** - This file

### **Downloaded Books:**

- `*.txt` files - Downloaded Gutenberg books (created when you run download script)

---

## 🎯 Purpose & Reasoning

### **Why Add Books to the RAG Application?**

Your current dataset consists of:

1. **12,464 pension files** - Individual veteran applications (1830s-1850s)
2. **65 newspapers** - Local community newspapers (1770s-1810s)

**Gap:** These sources are:

- ❌ **Fragmented** - Individual stories, not comprehensive narratives
- ❌ **Limited scope** - Pension files focus on individual service, newspapers on local events
- ❌ **Missing context** - No comprehensive battle histories or strategic overviews
- ❌ **Underrepresented groups** - Few African American soldiers in pension files

**Books fill these gaps by providing:**

- ✅ **Comprehensive battle histories** - Full accounts of major engagements
- ✅ **Strategic context** - Why battles were fought, their significance
- ✅ **Diverse perspectives** - African American soldiers, loyalists, spies
- ✅ **Cross-referencing opportunities** - Names, dates, locations to match with pension files
- ✅ **Educational content** - Historical context for hackathon judges/users

---

## 🏆 Prioritization Methodology

### **Scoring Criteria (1-5 scale):**

**1. Veteran Names & Cross-Referencing (Weight: 40%)**

- ⭐⭐⭐⭐⭐ Contains extensive lists of veteran names, prisoners, rosters
- ⭐⭐⭐⭐ Contains some veteran names, officer lists
- ⭐⭐⭐ Contains few names, mostly famous figures
- ⭐⭐ Minimal names, mostly narrative
- ⭐ No names, purely contextual

**2. Battle Details & Specificity (Weight: 30%)**

- ⭐⭐⭐⭐⭐ Detailed accounts of specific battles with dates, locations, participants
- ⭐⭐⭐⭐ Good battle coverage, some details
- ⭐⭐⭐ General battle mentions
- ⭐⭐ Minimal battle content
- ⭐ No battle content

**3. Diversity & Unique Perspectives (Weight: 15%)**

- ⭐⭐⭐⭐⭐ Covers underrepresented groups (African Americans, loyalists, women)
- ⭐⭐⭐⭐ Unique perspective (spies, naval, prisoners)
- ⭐⭐⭐ Standard military history
- ⭐⭐ Political/diplomatic focus
- ⭐ General history

**4. Space Efficiency (Weight: 10%)**

- ⭐⭐⭐⭐⭐ Small file (<500 KB), high value
- ⭐⭐⭐⭐ Medium file (500 KB - 2 MB), high value
- ⭐⭐⭐ Large file (2-5 MB), good value
- ⭐⭐ Very large (>5 MB), moderate value
- ⭐ Multi-volume set, low value per MB

**5. Temporal Relevance (Weight: 5%)**

- ⭐⭐⭐⭐⭐ Covers 1775-1783 (Revolutionary War period)
- ⭐⭐⭐⭐ Covers 1770-1790 (includes lead-up and aftermath)
- ⭐⭐⭐ Covers broader period including Revolution
- ⭐⭐ Pre-war or post-war focus
- ⭐ Outside Revolutionary period

---

## 📊 Tier Breakdown

### **Tier 1: MUST HAVE (5 books)**

**Criteria:** Value score 5/5, high cross-reference potential, fills critical gaps

**Books:**

1. **American Prisoners of the Revolution** - Prisoner lists, names, locations
2. **The Campaign of 1776 around New York and Brooklyn** - Detailed battle accounts
3. **The Battle of April 19, 1775** - Lexington & Concord, first battles
4. **Colored Americans in the Wars of 1776 and 1812** - African American veterans
5. **A Study of Army Camp Life** - Daily soldier experiences

**Why These?**

- ✅ Directly complement pension files (prisoner lists, veteran names)
- ✅ Fill diversity gap (African American soldiers underrepresented)
- ✅ Provide battle context (newspapers lack detailed battle coverage)
- ✅ Small-medium size (space efficient)
- ✅ High cross-referencing value

---

### **Tier 2: HIGH VALUE (6 books)**

**Criteria:** Value score 4/5, strong additions, good cross-reference potential

**Books:**
6. **The Pictorial Field-Book of the Revolution, Vol. 1** - Encyclopedia-like reference
7. **The Pictorial Field-Book of the Revolution, Vol. 2** - Continuation
8. **The Navy of the American Revolution** - Naval veterans, sea battles
9. **Narrative and Critical History of America, Vol. 6** - Revolutionary period focus
10. **General Washington's spies** - Spy names, covert operations
11. **The Loyalists of Massachusetts** - Other side of the story

**Why These?**

- ✅ Comprehensive coverage (Field-Books are encyclopedic)
- ✅ Fill naval gap (pension files mostly army)
- ✅ Unique perspectives (spies, loyalists)
- ✅ Good cross-referencing value
- ⚠️ Some large files (Vol. 6 is large, set download=false by default)

---

### **Tier 3: GOOD TO HAVE (5 books)**

**Criteria:** Value score 3/5, useful but not critical

**Books:**
12. **The Command in the Battle of Bunker Hill** - Specific battle analysis
13. **Autobiography of Benjamin Franklin** - Founding father perspective
14. **The Life of George Washington, Vol. 1** - Washington's perspective
15. **The Eve of the Revolution** - Pre-war context
16. **The Diplomatic Correspondence, Vol. 1** - Political context

**Why Lower Priority?**

- ⚠️ Narrow focus (single battle) or famous figures (already well-known)
- ⚠️ Less veteran-focused (political/diplomatic)
- ⚠️ Large files (Washington biography is 5-volume set)
- ⚠️ Lower cross-referencing value

---

### **Tier 4: SKIP (8 books)**

**Criteria:** Value score 1/5, not recommended

**Reasons to Skip:**

- ❌ Too general (History of the United States - covers all periods)
- ❌ Too small (Declaration of Independence - tiny, well-known)
- ❌ Post-war focus (Republican Party, Susan B. Anthony)
- ❌ Pre-war focus (Colonial Virginia, Beginnings of American People)
- ❌ Theoretical (Anatomy of Revolution - not specific to American Revolution)
- ❌ Multi-volume overload (Jefferson Writings - 9 volumes, political focus)

---

## 🔍 Cross-Referencing Strategy

### **How Books Enhance Cross-Referencing:**

**1. Prisoner Lists → Pension Files**

- **Book:** American Prisoners of the Revolution
- **Contains:** Names of prisoners, locations, dates
- **Cross-reference:** Match prisoner names with pension applicants
- **Value:** Validates service claims, adds context to pension narratives

**2. Battle Participants → Newspapers**

- **Book:** The Campaign of 1776 around New York and Brooklyn
- **Contains:** Officer names, troop movements, battle details
- **Cross-reference:** Match with newspaper battle mentions
- **Value:** Fills gaps in newspaper coverage (newspapers mostly ads/local news)

**3. African American Veterans → Pension Files**

- **Book:** Colored Americans in the Wars of 1776 and 1812
- **Contains:** Names of African American soldiers
- **Cross-reference:** Find underrepresented veterans in pension files
- **Value:** Highlights diversity, fills historical gap

**4. Spy Networks → Historical Context**

- **Book:** General Washington's spies on Long Island
- **Contains:** Spy names, covert operations
- **Cross-reference:** Unique perspective not in pension files or newspapers
- **Value:** Interesting niche content for hackathon demo

**5. Loyalist Names → Balanced Perspective**

- **Book:** The Loyalists of Massachusetts
- **Contains:** Names of loyalists, their stories
- **Cross-reference:** Other side of the conflict
- **Value:** Balanced historical perspective

---

## 📥 Download Strategy

### **Recommended Approach:**

**Phase 1: Start with Tier 1 (5 books, ~10-15 MB)**

```powershell
python scripts/download_books.py --preset minimal
```

**Why:** Highest value, smallest size, immediate impact

**Phase 2: Add Tier 2 (10 books total, ~30-35 MB)**

```powershell
python scripts/download_books.py --preset standard
```

**Why:** Comprehensive coverage, good balance of size vs value

**Phase 3: Optional Tier 3 (15 books total, ~50-60 MB)**

```powershell
python scripts/download_books.py --preset comprehensive
```

**Why:** Only if space allows, diminishing returns

---

## 🎓 Educational Value for Hackathon

### **Why Judges Will Care:**

**1. Demonstrates Comprehensive Approach**

- ✅ Not just using provided datasets (pension files, newspapers)
- ✅ Proactively supplemented with historical context
- ✅ Shows research skills and historical knowledge

**2. Improves RAG Quality**

- ✅ More diverse sources = better answers
- ✅ Books provide context that pension files/newspapers lack
- ✅ Cross-referencing creates richer knowledge graph

**3. Fills Historical Gaps**

- ✅ African American soldiers (underrepresented in pension files)
- ✅ Naval veterans (mostly army in pension files)
- ✅ Battle context (newspapers mostly ads/local news)
- ✅ Loyalist perspective (balanced history)

**4. Enables Interesting Queries**

- ✅ "Tell me about African American soldiers at the Battle of Yorktown"
- ✅ "What happened to prisoners of war?"
- ✅ "How did spies operate during the Revolution?"
- ✅ "What was daily life like in army camps?"

---

## 🔧 Technical Implementation

### **JSON Catalog Structure:**

```json
{
  "id": 1,
  "title": "American Prisoners of the Revolution",
  "author": "Danske Dandridge",
  "gutenberg_id": "7829",
  "url": "https://www.gutenberg.org/cache/epub/7829/pg7829.txt",
  "filename": "american_prisoners_revolution.txt",
  "tier": 1,
  "priority": "must_have",
  "size_estimate": "medium",
  "value_score": 5,
  "download": true,              ← Boolean flag for download
  "downloaded": false,           ← Auto-updated when downloaded
  "downloaded_date": null,       ← ISO 8601 timestamp when downloaded
  "processed": false,            ← For tracking vector DB ingestion
  "processed_date": null,        ← ISO 8601 timestamp when processed
  "file_size_bytes": null,       ← Actual file size in bytes
  "tags": ["prisoners", "battles", "veteran_names", "locations"],
  "why_priority": "Names of prisoners, battles, locations - direct veteran connections",
  "cross_reference_value": "high"
}
```

### **Boolean Flags Explained:**

**`download` (boolean):**

- **Purpose:** Mark books for download
- **Default:** `true` for Tier 1-2, `false` for Tier 3-4
- **Usage:** Easily toggle which books to download
- **Example:** Set `download: false` for large files if space limited

**`downloaded` (boolean):**

- **Purpose:** Track download status
- **Default:** `false`
- **Auto-updated:** Set to `true` by download script when file saved
- **Usage:** Avoid re-downloading, show progress

**`downloaded_date` (string | null):**

- **Purpose:** Record when book was downloaded
- **Format:** ISO 8601 timestamp (e.g., "2026-05-12T16:45:30.123456")
- **Default:** `null`
- **Auto-updated:** Set by download script when file saved
- **Usage:** Track download history, audit trail

**`processed` (boolean):**

- **Purpose:** Track vector database ingestion
- **Default:** `false`
- **Usage:** Mark books as processed after adding to vector DB
- **Future:** Can re-process if embeddings change

**`processed_date` (string | null):**

- **Purpose:** Record when book was added to vector database
- **Format:** ISO 8601 timestamp (e.g., "2026-05-12T17:30:15.654321")
- **Default:** `null`
- **Manual update:** Use `mark_book_processed.py` script
- **Usage:** Track processing history, identify stale embeddings

**`file_size_bytes` (integer | null):**

- **Purpose:** Record actual file size
- **Format:** Integer (bytes)
- **Default:** `null`
- **Auto-updated:** Set by download script after file saved
- **Usage:** Verify downloads, calculate storage usage, compare with estimates

---

## 📋 Download Presets

### **Preset 1: Minimal (5 books, ~12 MB)**

**Book IDs:** 1, 2, 3, 4, 5  
**Use Case:** Limited space, want highest value books only  
**Command:** `python scripts/download_books.py --preset minimal`

### **Preset 2: Standard (10 books, ~35 MB)**

**Book IDs:** 1, 2, 3, 4, 5, 6, 7, 8, 10, 11  
**Use Case:** Good balance of coverage and size (recommended)  
**Command:** `python scripts/download_books.py --preset standard`  
**Note:** Excludes book ID 9 (Narrative History Vol. 6 - large file)

### **Preset 3: Comprehensive (15 books, ~55 MB)**

**Book IDs:** 1-8, 10-16  
**Use Case:** Maximum coverage, space not a concern  
**Command:** `python scripts/download_books.py --preset comprehensive`

---

## 🛠️ Management Commands

### **Check Status:**

```powershell
python scripts/download_books.py --status
```

Shows:

- Books by tier
- Download flags
- Downloaded status
- Total counts

### **Download All Flagged:**

```powershell
python scripts/download_books.py --download-all
```

Downloads all books where `download: true`

### **Toggle Individual Books:**

```powershell
# Flag book ID 9 for download
python scripts/download_books.py --flag 9

# Unflag book ID 12
python scripts/download_books.py --unflag 12
```

### **After Downloading:**

```powershell
# Check what was downloaded
python scripts/download_books.py --status

# Verify files exist
ls data/documents/books/*.txt
```

### **Mark Books as Processed:**

After adding books to vector database, mark them as processed:

```powershell
# Mark single book as processed
python scripts/mark_book_processed.py --id 1

# Or by filename
python scripts/mark_book_processed.py --filename american_prisoners_revolution.txt

# Mark all downloaded books as processed
python scripts/mark_book_processed.py --all

# Check processing status
python scripts/mark_book_processed.py --status
```

**What it does:**

- Sets `processed: true`
- Records `processed_date` timestamp
- Validates book was downloaded first
- Shows processing history

---

## 📊 Expected Outcomes

### **After Adding Books to RAG:**

**1. Improved Answer Quality**

- More comprehensive responses to battle questions
- Better context for pension file narratives
- Diverse perspectives (African American, loyalist, naval)

**2. Enhanced Cross-Referencing**

- Match veteran names across books, pension files, newspapers
- Validate service claims with prisoner lists
- Connect battles mentioned in newspapers with detailed book accounts

**3. Richer Knowledge Graph**

- Books → Battles → Veterans → Pension Files
- Books → People → Newspapers → Locations
- Books → Events → Context → Historical Significance

**4. Better Demo Queries**

- "Tell me about the Battle of Yorktown" → Comprehensive answer from books
- "Who were the African American soldiers?" → Names from Colored Americans book
- "What happened to prisoners of war?" → Details from Prisoners book
- "What was daily life like?" → Context from Army Camp Life book

---

## 🎯 Success Metrics

### **How to Measure Impact:**

**Before Books (Baseline):**

- Query: "Tell me about the Battle of Yorktown"
- Answer: Limited to newspaper mentions (if any)
- Quality: Low (newspapers mostly ads/local news)

**After Books:**

- Query: "Tell me about the Battle of Yorktown"
- Answer: Comprehensive from books + newspaper context + veteran connections
- Quality: High (detailed battle account + local perspective + veteran stories)

**Quantitative Metrics:**

- Number of veteran names cross-referenced: +500-1000 (from books)
- Battle coverage: +20 major battles (detailed accounts)
- Diversity: +100 African American soldiers (from Colored Americans book)
- Query satisfaction: Improved (more comprehensive answers)

---

## 🔮 Future Enhancements

### **Phase 2 (If Time Allows):**

**1. Add More Books**

- Download Tier 3 books (5 additional books)
- Consider multi-volume sets (Washington biography, Jefferson writings)

**2. Enhanced Metadata**

- Extract chapter titles, section headings
- Create book-specific indexes
- Tag paragraphs with topics (battles, people, locations)

**3. Advanced Cross-Referencing**

- Build knowledge graph: Books ↔ Pension Files ↔ Newspapers
- Entity linking: Same person mentioned across sources
- Timeline visualization: Events across all sources

**4. Quality Improvements**

- OCR improvement for scanned books (if needed)
- Named entity recognition (NER) on book text
- Sentiment analysis (victory/defeat, optimistic/pessimistic)

---

## 📚 Source Attribution

### **Original Source:**

`list_of_american_revolution_books.txt`

**Contains:**

- 47 Gutenberg book URLs
- Hugging Face dataset information
- API access information

**Processed Into:**

- `book_catalog.json` - Structured catalog with metadata
- `PRIORITIZED_BOOK_LIST.md` - Human-readable prioritization
- `README.md` - This documentation

**Date:** May 12, 2026  
**Purpose:** BAH WAI Smithsonian Hackathon  
**Team:** Smithsonian Hackathon Team

---

## ✅ Checklist

### **Before Downloading:**

- [ ] Review `book_catalog.json` and adjust `download` flags if needed
- [ ] Check available disk space (~35 MB for standard preset)
- [ ] Ensure internet connection is stable

### **Download Process:**

- [ ] Run `python scripts/download_books.py --status` to see what's flagged
- [ ] Run `python scripts/download_books.py --preset standard` to download
- [ ] Verify downloads: `ls data/documents/books/*.txt`

### **After Downloading:**

- [ ] Update `book_catalog.json` (auto-updated by script)
- [ ] Add books to vector database (separate ingestion script)
- [ ] Test RAG queries with book content
- [ ] Document improvements in hackathon presentation

---

## 🤝 Contributing

### **To Add New Books:**

1. **Add to `book_catalog.json`:**

```json
{
  "id": 25,
  "title": "New Book Title",
  "author": "Author Name",
  "gutenberg_id": "12345",
  "url": "https://www.gutenberg.org/cache/epub/12345/pg12345.txt",
  "filename": "new_book.txt",
  "tier": 2,
  "priority": "high_value",
  "size_estimate": "medium",
  "value_score": 4,
  "download": true,
  "downloaded": false,
  "processed": false,
  "tags": ["relevant", "tags"],
  "why_priority": "Reason for inclusion",
  "cross_reference_value": "high"
}
```

1. **Update metadata:**

```json
"metadata": {
  "total_books": 48,  ← Increment
  "prioritized_books": 17  ← Increment if Tier 1-3
}
```

1. **Test download:**

```powershell
python scripts/download_books.py --flag 25
python scripts/download_books.py --download-all
```

---

## 📞 Support

### **Issues?**

**Download fails:**

- Check internet connection
- Verify Gutenberg URL is still valid
- Try manual download: `wget <url>`

**JSON errors:**

- Validate JSON: `python -m json.tool book_catalog.json`
- Check for trailing commas, missing quotes

**Script errors:**

- Ensure Python 3.7+ installed
- Install dependencies: `pip install requests`
- Check file paths are correct

---

**Document Version:** 1.0  
**Last Updated:** May 12, 2026  
**Maintained By:** Smithsonian Hackathon Team  
**License:** Public Domain (Project Gutenberg books)
