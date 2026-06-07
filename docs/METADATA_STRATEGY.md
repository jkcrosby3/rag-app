# Metadata Enrichment Strategy for Your RAG System

## Current State Analysis

### ✅ What You Have (Good!)

**Three Collections:**
1. **American Revolutionary Era Collections**: 12,641 museum objects (NMAH, NPM, SAAM, NPG)
   - Already has rich metadata JSON files
   - Includes: dates, places, makers, object types, descriptions
   - **Already indexed**: dates, names, places, object types, topics

2. **Pension Files**: 12,606 Revolutionary War veteran records
   - Basic header metadata (Record ID, title, NARA URL, page count)
   - Full text content (service records, applications, correspondence)
   - **Needs enrichment**: veteran names, ranks, units, dates, locations

3. **Newspapers**: 65 Chronicling America issues (1770-1810)
   - Minimal metadata (just Document ID)
   - OCR text content (advertisements, articles, notices)
   - **Needs enrichment**: publication date, location, article topics

---

## Recommendation: DON'T Start Over!

### Why Your Current Data is Good:
1. ✅ **Rev Era Collections** already have excellent metadata
2. ✅ Text files are properly formatted for RAG
3. ✅ You have 25,282 + 12,606 + 65 = **37,953 documents** ready
4. ✅ Metadata JSON files exist alongside text files

### What You Should Do Instead:
**Add enrichment as a POST-PROCESSING step** after download, before embedding.

---

## Proposed Architecture

### Option 1: Unified Download System (Recommended)
```
scripts/
  ├── download_manager.py          # Main orchestrator
  ├── downloaders/
  │   ├── __init__.py
  │   ├── base_downloader.py       # Abstract base class
  │   ├── pension_downloader.py    # Pension-specific logic
  │   ├── newspaper_downloader.py  # Newspaper-specific logic
  │   └── collections_downloader.py # Rev Era-specific logic
  ├── enrichers/
  │   ├── __init__.py
  │   ├── base_enricher.py         # Abstract base class
  │   ├── pension_enricher.py      # Extract veteran info
  │   ├── newspaper_enricher.py    # Extract publication info
  │   └── collections_enricher.py  # Already has metadata!
  └── config/
      └── download_config.yaml     # Configuration file
```

### Configuration File (download_config.yaml)
```yaml
datasets:
  pension_files:
    enabled: true
    source: "NARA"
    url: "https://catalog.archives.gov/..."
    output_dir: "data/documents/smithsonian/pension_files"
    enrichment: "pension"
    
  newspapers:
    enabled: true
    source: "Chronicling America"
    url: "https://chroniclingamerica.loc.gov/..."
    output_dir: "data/documents/smithsonian/newspapers"
    enrichment: "newspaper"
    
  rev_era_collections:
    enabled: true
    source: "HuggingFace"
    dataset_id: "RevolutionCrossroads/si_us_revolutionary_era_collections"
    output_dir: "data/documents/smithsonian/american_revolutionary_era_collections"
    enrichment: "collections"
    
  images_textlabeling:
    enabled: false  # Not needed for text RAG
    source: "HuggingFace"
    dataset_id: "RevolutionCrossroads/si_images_textlabeling_bah"
```

---

## Metadata Structure

### Unified Metadata Schema
All documents should have a common base structure, plus collection-specific fields:

```json
{
  // COMMON FIELDS (all documents)
  "document_id": "111403760",
  "collection": "pension_files",
  "source_system": "NARA",
  "file_path": "data/documents/smithsonian/pension_files/text_files/111403760.txt",
  "file_size": 44096,
  "ingestion_date": "2026-05-08T23:50:00Z",
  "text_length": 25000,
  "word_count": 4200,
  
  // TEMPORAL
  "document_date": "1836-06-02",
  "historical_period": "post_revolutionary_war",
  "era": "1830s",
  
  // CONTENT
  "document_type": "pension_application",
  "language": "en",
  "ocr_quality": 0.85,
  
  // COLLECTION-SPECIFIC (varies by type)
  "collection_metadata": {
    // Pension files: veteran info
    // Newspapers: publication info
    // Collections: object info
  },
  
  // ENRICHMENTS (added post-processing)
  "entities": {
    "persons": [...],
    "places": [...],
    "organizations": [...]
  },
  
  // EMBEDDING INFO
  "embedding": {
    "model": "all-MiniLM-L6-v2",
    "chunk_index": 0,
    "total_chunks": 8
  }
}
```

---

## Collection-Specific Metadata

### 1. Pension Files (Needs Most Enrichment)
```json
{
  "collection_metadata": {
    "veteran_name": "Joseph Torrey",
    "rank": "Major",
    "unit": "Col. Hazen's Regiment",
    "branch": "Continental Army",
    "service_start": "1776",
    "service_end": "1783",
    "pension_number": "B.L. Wt. 2,153-400",
    "pension_type": "bounty_land_warrant",
    "state_of_service": "New York",
    "state_of_residence": "New York",
    "nara_url": "https://catalog.archives.gov/id/111403760",
    "page_count": 54
  }
}
```

### 2. Revolutionary Era Collections (Already Good!)
```json
{
  "collection_metadata": {
    "object_name": "Revenue Measures Set",
    "unit_code": "NMAH",
    "data_source": "National Museum of American History",
    "edan_id": "edanmdm:nmah_2434",
    "date_made": "ca. 1800",
    "maker": "Dring & Fage",
    "place_made": "United Kingdom: England, London",
    "measurements": "...",
    "description": "...",
    "collections_url": "https://collections.si.edu/search/detail/edanmdm:nmah_2434",
    "indexed_dates": ["1800s"],
    "indexed_names": ["Dring & Fage"],
    "indexed_places": ["England", "London", "United Kingdom"],
    "indexed_object_types": ["Revenue Measures Set"]
  }
}
```

### 3. Newspapers (Needs Enrichment)
```json
{
  "collection_metadata": {
    "lccn": "sn82014385",
    "newspaper_title": "The Virginia Gazette",
    "publication_date": "1776-07-04",
    "publication_place": "Williamsburg, Virginia",
    "publisher": "...",
    "page_number": 1,
    "issue_number": "...",
    "chronicling_america_url": "https://chroniclingamerica.loc.gov/lccn/sn82014385/",
    "ocr_quality": "fair"
  }
}
```

---

## Implementation Plan

### Phase 1: Create Enrichment Pipeline (Don't Re-Download!)
```python
# scripts/enrich_existing_documents.py

def enrich_pension_files():
    """Add veteran metadata to existing pension files."""
    pension_dir = Path("data/documents/smithsonian/pension_files/text_files")
    
    for txt_file in pension_dir.glob("*.txt"):
        # Read existing text
        text = txt_file.read_text()
        
        # Extract metadata
        metadata = extract_pension_metadata(text)
        
        # Save enriched metadata
        metadata_file = txt_file.with_suffix('.json')
        save_metadata(metadata, metadata_file)

def extract_pension_metadata(text):
    """Extract veteran information from pension file text."""
    metadata = {
        "document_id": extract_record_id(text),
        "collection": "pension_files",
        "veteran_name": extract_veteran_name(text),
        "rank": extract_rank(text),
        "unit": extract_unit(text),
        "service_dates": extract_service_dates(text),
        "pension_number": extract_pension_number(text),
        # ... more fields
    }
    return metadata
```

### Phase 2: Unified Metadata Format
Create a metadata standardizer that converts all three formats to a common schema:

```python
# scripts/standardize_metadata.py

def standardize_metadata(collection_type, raw_metadata):
    """Convert collection-specific metadata to unified format."""
    
    base_metadata = {
        "document_id": raw_metadata.get("document_id"),
        "collection": collection_type,
        "ingestion_date": datetime.now().isoformat(),
        # ... common fields
    }
    
    if collection_type == "pension_files":
        base_metadata["collection_metadata"] = format_pension_metadata(raw_metadata)
    elif collection_type == "american_revolutionary_era_collections":
        base_metadata["collection_metadata"] = format_collections_metadata(raw_metadata)
    elif collection_type == "newspapers":
        base_metadata["collection_metadata"] = format_newspaper_metadata(raw_metadata)
    
    return base_metadata
```

### Phase 3: Update Embedding Pipeline
Modify your embedding process to use the enriched metadata:

```python
# In your embedding script
def embed_document(text_file):
    # Load text
    text = text_file.read_text()
    
    # Load enriched metadata
    metadata_file = text_file.with_suffix('.json')
    if metadata_file.exists():
        metadata = json.load(open(metadata_file))
    else:
        # Fallback to basic metadata
        metadata = create_basic_metadata(text_file)
    
    # Generate embedding
    embedding = generate_embedding(text)
    
    # Store with enriched metadata
    store_in_vector_db(text, embedding, metadata)
```

---

## Specific Enrichment Strategies

### For Pension Files (High Priority)
```python
import re
from datetime import datetime

def extract_veteran_name(text):
    """Extract veteran name from pension file."""
    # Look for patterns like "Joseph Torrey" in title or header
    patterns = [
        r'title:\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',
        r'Pension.*?(?:File|Application)\s+(?:of\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
    return None

def extract_rank(text):
    """Extract military rank."""
    ranks = ['Private', 'Corporal', 'Sergeant', 'Lieutenant', 'Captain', 
             'Major', 'Colonel', 'General']
    for rank in ranks:
        if re.search(rf'\b{rank}\b', text, re.IGNORECASE):
            return rank
    return None

def extract_unit(text):
    """Extract military unit."""
    patterns = [
        r"(Col\.\s+[A-Z][a-z]+(?:'s)?\s+Regiment)",
        r"(\d+(?:st|nd|rd|th)\s+Regiment)",
        r"(Continental\s+Army)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    return None
```

### For Newspapers (Medium Priority)
```python
def extract_newspaper_metadata(text, filename):
    """Extract newspaper publication info."""
    # LCCN is in filename
    lccn = filename.replace('.txt', '')
    
    # Try to extract date from text
    date_pattern = r'(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})'
    date_match = re.search(date_pattern, text)
    
    return {
        "lccn": lccn,
        "publication_date": date_match.group(1) if date_match else None,
        "chronicling_america_url": f"https://chroniclingamerica.loc.gov/lccn/{lccn}/"
    }
```

### For Rev Era Collections (Low Priority - Already Good!)
```python
def enhance_collections_metadata(existing_metadata):
    """Minor enhancements to already-good metadata."""
    # Parse JSON strings into proper objects
    if 'date' in existing_metadata and isinstance(existing_metadata['date'], str):
        existing_metadata['date_parsed'] = json.loads(existing_metadata['date'])
    
    # Extract year from date
    if 'indexed_dates' in existing_metadata:
        existing_metadata['year'] = extract_year(existing_metadata['indexed_dates'][0])
    
    return existing_metadata
```

---

## Answer to Your Questions

### Q: Should metadata be in one JSON or separate?
**A: One unified JSON per document** (what you're already doing!)
- Keep `document_id.txt` for RAG text
- Keep `document_id_metadata.json` for full metadata
- During embedding, merge them into vector DB

### Q: Should I start over with downloading?
**A: NO! Post-process your existing data**
- Rev Era Collections: Already perfect
- Pension Files: Add enrichment script
- Newspapers: Add enrichment script

### Q: Should I have one unified download script?
**A: Yes, for future downloads, but not urgent**
- Current 3 scripts work fine
- Unified system is better long-term
- Focus on enrichment first

### Q: Should enrichment be a class?
**A: YES! Absolutely**
```python
# scripts/enrichers/base_enricher.py
class BaseEnricher:
    def enrich(self, text, metadata):
        raise NotImplementedError

# scripts/enrichers/pension_enricher.py
class PensionEnricher(BaseEnricher):
    def enrich(self, text, metadata):
        return {
            **metadata,
            "veteran_name": self.extract_veteran_name(text),
            "rank": self.extract_rank(text),
            # ...
        }
```

---

## Priority Action Items

### Immediate (This Weekend):
1. ✅ Create `scripts/enrich_pension_files.py`
2. ✅ Extract veteran names, ranks, units from pension files
3. ✅ Create standardized metadata JSON files
4. ✅ Test with 100 files first

### Short-term (Next Week):
1. Create `scripts/enrich_newspapers.py`
2. Extract publication dates and locations
3. Run enrichment on all collections
4. Re-build vector database with enriched metadata

### Long-term (Post-Hackathon):
1. Unified download manager
2. Advanced NER with spaCy
3. Relationship mapping between documents
4. Analytics dashboard

---

## Summary

**Your current approach is GOOD!** Don't start over. Instead:

1. ✅ Keep your existing 3 download scripts
2. ✅ Keep your existing text files and metadata
3. ✅ Add POST-PROCESSING enrichment scripts
4. ✅ Create unified metadata schema
5. ✅ Re-embed with enriched metadata

**The enrichment document I created is a REFERENCE** - you don't need to implement everything. Focus on:
- Pension files: veteran names, ranks, units (HIGH VALUE)
- Newspapers: dates, locations (MEDIUM VALUE)
- Collections: already good! (LOW PRIORITY)

This will give you a powerful, searchable RAG system for the hackathon! 🚀
