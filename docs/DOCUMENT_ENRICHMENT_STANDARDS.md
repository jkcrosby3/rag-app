# Document Enrichment & Metadata Standards for RAG Systems

## Industry Best Practices for Document Ingestion

### Overview
Modern RAG systems don't just embed raw text - they enrich documents with extensive metadata during ingestion to enable:
- Better retrieval filtering
- Faceted search
- Analytics and insights
- Quality control
- Provenance tracking

---

## 1. Basic Metadata (What You Have)

### Current Implementation ✅
```json
{
  "file_path": "path/to/document.txt",
  "file_name": "document.txt",
  "file_size": 12345,
  "relative_path": "pension_files/text_files/document.txt",
  "topic": "pension_files",
  "chunk_index": 0,
  "total_chunks": 5
}
```

---

## 2. Enhanced Metadata (Industry Standard)

### Document-Level Metadata

#### A. Provenance & Source Tracking
```json
{
  "source_system": "Smithsonian NARA",
  "collection_name": "Revolutionary War Pension Files",
  "record_id": "111403815",
  "catalog_url": "https://catalog.archives.gov/id/111403815",
  "original_format": "scanned_document",
  "digitization_date": "2020-03-15",
  "archive_location": "National Archives, Washington DC"
}
```

#### B. Temporal Metadata
```json
{
  "document_date": "1836-06-02",
  "document_date_parsed": "1836-06-02T00:00:00Z",
  "date_precision": "day",  // day, month, year, circa
  "time_period": "post_revolutionary_war",
  "era": "1830s",
  "historical_context": "Pension application period"
}
```

#### C. Content Classification
```json
{
  "document_type": "pension_application",
  "document_subtype": "service_record",
  "content_category": "legal_document",
  "language": "en",
  "language_confidence": 0.99,
  "ocr_quality_score": 0.85,  // 0-1 scale
  "readability_score": 65.2,  // Flesch reading ease
  "word_count": 450,
  "page_count": 5
}
```

#### D. Named Entity Recognition (NER)
```json
{
  "entities": {
    "persons": [
      {"name": "Joseph Torrey", "role": "applicant", "confidence": 0.95},
      {"name": "John Torrey", "role": "heir", "confidence": 0.92},
      {"name": "George Washington", "role": "commander", "confidence": 0.88}
    ],
    "organizations": [
      {"name": "Col. Hazen's Regiment", "type": "military_unit"},
      {"name": "Continental Army", "type": "military_branch"}
    ],
    "locations": [
      {"name": "New York City", "type": "city", "state": "New York"},
      {"name": "Valley Forge", "type": "military_camp"}
    ],
    "dates": [
      {"date": "1776-03-15", "event": "enlistment"},
      {"date": "1783-09-03", "event": "discharge"}
    ]
  }
}
```

#### E. Military-Specific Metadata (for your use case)
```json
{
  "military": {
    "veteran_name": "Joseph Torrey",
    "rank": "Major",
    "unit": "Col. Hazen's Regiment",
    "branch": "Continental Army",
    "service_start": "1776",
    "service_end": "1783",
    "battles": ["Saratoga", "Yorktown"],
    "pension_number": "B.L. Wt. 2,153-400",
    "pension_type": "bounty_land_warrant",
    "pension_amount": 400,
    "state_of_service": "New York",
    "state_of_residence": "New York"
  }
}
```

#### F. Content Quality Metrics
```json
{
  "quality": {
    "ocr_confidence": 0.85,
    "text_completeness": 0.92,  // % of expected content present
    "has_missing_pages": false,
    "has_damage": false,
    "legibility": "good",  // poor, fair, good, excellent
    "transcription_verified": false,
    "needs_review": false
  }
}
```

#### G. Semantic Metadata
```json
{
  "topics": ["military_service", "pension_benefits", "legal_proceedings"],
  "keywords": ["revolutionary_war", "veteran", "pension", "continental_army"],
  "themes": ["patriotism", "sacrifice", "government_benefits"],
  "sentiment": "neutral",
  "document_summary": "Pension application for Major Joseph Torrey...",
  "key_facts": [
    "Served as Major in Col. Hazen's Regiment",
    "Applied for pension in 1836",
    "Had 5 heirs listed"
  ]
}
```

#### H. Relationships & Links
```json
{
  "related_documents": [
    {"id": "111403816", "relationship": "same_veteran", "confidence": 1.0},
    {"id": "111403820", "relationship": "same_unit", "confidence": 0.75}
  ],
  "citations": [
    {"document_id": "act_of_congress_1776", "type": "legal_reference"}
  ],
  "cross_references": [
    {"type": "widow_claim", "document_id": "111403817"}
  ]
}
```

---

## 3. Chunk-Level Metadata (Beyond Document-Level)

### Enhanced Chunk Metadata
```json
{
  "chunk_id": "111403815_chunk_0",
  "parent_document_id": "111403815",
  "chunk_index": 0,
  "total_chunks": 5,
  "chunk_type": "header",  // header, body, signature, annotation
  "chunk_position": "beginning",  // beginning, middle, end
  "chunk_length": 512,
  "chunk_overlap": 50,
  "contains_entities": ["Joseph Torrey", "New York"],
  "contains_dates": ["1836-06-02"],
  "section_title": "Service Record",
  "page_number": 1,
  "semantic_density": 0.72,  // How information-rich is this chunk
  "embedding_model": "all-MiniLM-L6-v2",
  "embedding_version": "1.0",
  "embedded_at": "2026-05-08T20:00:00Z"
}
```

---

## 4. Computed Analytics (Post-Ingestion)

### Collection-Level Statistics
```json
{
  "collection_stats": {
    "total_documents": 12606,
    "total_veterans": 12606,
    "date_range": {
      "earliest": "1776-01-01",
      "latest": "1920-12-31"
    },
    "geographic_distribution": {
      "New York": 2345,
      "Pennsylvania": 1876,
      "Massachusetts": 1654
    },
    "rank_distribution": {
      "Private": 8500,
      "Corporal": 1200,
      "Sergeant": 800,
      "Lieutenant": 600,
      "Captain": 400,
      "Major": 80,
      "Colonel": 26
    },
    "units_represented": 450,
    "average_service_years": 4.2
  }
}
```

---

## 5. Implementation Approaches

### A. During Ingestion (Recommended)
```python
def enrich_document(document_text, metadata):
    """Enrich document with additional metadata during ingestion."""
    
    # 1. Extract named entities
    entities = extract_entities(document_text)
    
    # 2. Parse dates
    dates = extract_dates(document_text)
    
    # 3. Identify document type
    doc_type = classify_document_type(document_text, metadata)
    
    # 4. Extract military information
    military_info = extract_military_info(document_text)
    
    # 5. Calculate quality metrics
    quality_metrics = assess_quality(document_text)
    
    # 6. Generate summary
    summary = generate_summary(document_text)
    
    # 7. Extract keywords
    keywords = extract_keywords(document_text)
    
    # Merge all enrichments
    enriched_metadata = {
        **metadata,
        "entities": entities,
        "dates": dates,
        "document_type": doc_type,
        "military": military_info,
        "quality": quality_metrics,
        "summary": summary,
        "keywords": keywords
    }
    
    return enriched_metadata
```

### B. Post-Ingestion Analysis
```python
def analyze_collection(vector_db):
    """Analyze entire collection for insights."""
    
    # Load all documents
    all_docs = vector_db.get_all_documents()
    
    # Compute statistics
    stats = {
        "total_veterans": count_unique_veterans(all_docs),
        "rank_distribution": compute_rank_distribution(all_docs),
        "geographic_distribution": compute_geographic_distribution(all_docs),
        "temporal_distribution": compute_temporal_distribution(all_docs),
        "unit_analysis": analyze_military_units(all_docs)
    }
    
    # Save analytics
    save_analytics(stats, "data/analytics/collection_stats.json")
```

---

## 6. Benefits of Enhanced Metadata

### Improved Retrieval
- **Faceted Search**: Filter by date, location, rank, unit
- **Hybrid Search**: Combine semantic + metadata filters
- **Precision**: "Show me captains from Pennsylvania who served 1776-1778"

### Analytics & Insights
- Visualize veteran distribution by state
- Timeline of pension applications
- Unit composition analysis
- Service duration statistics

### Quality Control
- Identify low-quality OCR documents
- Flag incomplete records
- Prioritize documents for manual review

### User Experience
- Show document previews with key facts
- Display related documents
- Provide context (dates, people, places)

---

## 7. Tools & Libraries for Enrichment

### Named Entity Recognition
```python
# spaCy (fast, accurate)
import spacy
nlp = spacy.load("en_core_web_sm")
doc = nlp(text)
entities = [(ent.text, ent.label_) for ent in doc.ents]

# Hugging Face Transformers (more accurate)
from transformers import pipeline
ner = pipeline("ner", model="dslim/bert-base-NER")
entities = ner(text)
```

### Date Extraction
```python
# dateutil
from dateutil import parser
dates = parser.parse("June 2nd 1836")

# datefinder
import datefinder
dates = list(datefinder.find_dates(text))
```

### Document Classification
```python
# Zero-shot classification
from transformers import pipeline
classifier = pipeline("zero-shot-classification")
result = classifier(text, candidate_labels=["pension_application", "service_record", "correspondence"])
```

### Text Summarization
```python
# Extractive summarization
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer

parser = PlaintextParser.from_string(text, Tokenizer("english"))
summarizer = LsaSummarizer()
summary = summarizer(parser.document, 3)  # 3 sentences
```

---

## 8. Recommended Implementation for Your Project

### Phase 1: Essential Enrichments (Quick Wins)
1. **Extract veteran names** from pension file titles
2. **Parse dates** from document text
3. **Extract ranks and units** using regex patterns
4. **Add document summaries** (first 200 chars or LLM-generated)
5. **Compute quality scores** (OCR confidence, completeness)

### Phase 2: Advanced Enrichments
1. **Full NER** for all persons, places, organizations
2. **Relationship mapping** between documents
3. **Geographic analysis** with state/county extraction
4. **Timeline construction** for each veteran
5. **Unit roster building** from cross-references

### Phase 3: Analytics Dashboard
1. **Collection statistics** (total veterans, date ranges)
2. **Geographic visualizations** (maps, charts)
3. **Rank distribution** analysis
4. **Service duration** statistics
5. **Battle participation** tracking

---

## 9. Storage Considerations

### Metadata Storage Options

#### Option A: Embedded in Vector DB (Current)
- ✅ Simple, everything in one place
- ❌ Limited query capabilities
- ❌ Hard to update metadata without re-embedding

#### Option B: Separate Metadata Database
```python
# PostgreSQL with JSONB
CREATE TABLE document_metadata (
    document_id VARCHAR PRIMARY KEY,
    vector_id INTEGER REFERENCES vector_embeddings(id),
    metadata JSONB,
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);

# Enable fast queries
CREATE INDEX idx_metadata_rank ON document_metadata ((metadata->>'military'->>'rank'));
CREATE INDEX idx_metadata_state ON document_metadata ((metadata->>'military'->>'state_of_service'));
```

#### Option C: Hybrid (Recommended)
- Core metadata in vector DB (for retrieval)
- Extended metadata in separate DB (for analytics)
- Link via document_id

---

## 10. Example: Enhanced Pension File Metadata

```json
{
  "document_id": "111403815",
  "file_name": "111403815.txt",
  "file_path": "data/documents/smithsonian/pension_files/text_files/111403815.txt",
  "topic": "pension_files",
  
  "source": {
    "collection": "Revolutionary War Pension and Bounty-Land Warrant Application Files",
    "archive": "National Archives",
    "catalog_url": "https://catalog.archives.gov/id/111403815",
    "record_group": "RG 15",
    "series": "M804"
  },
  
  "veteran": {
    "name": "Joseph Torrey",
    "rank": "Major",
    "unit": "Col. Hazen's Regiment",
    "branch": "Continental Army",
    "service_period": {
      "start": "1776",
      "end": "1783",
      "duration_years": 7
    },
    "state_of_service": "New York",
    "pension_number": "B.L. Wt. 2,153-400",
    "pension_type": "bounty_land_warrant",
    "acres_granted": 400
  },
  
  "document": {
    "type": "pension_application",
    "date": "1836-06-02",
    "pages": 5,
    "word_count": 1250,
    "language": "en",
    "ocr_quality": 0.85
  },
  
  "entities": {
    "persons": ["Joseph Torrey", "John Torrey", "William Torrey", "James Campbell"],
    "places": ["New York City", "New York County"],
    "organizations": ["Col. Hazen's Regiment", "Continental Army", "Surrogate's Office"]
  },
  
  "heirs": [
    {"name": "John Torrey", "relationship": "grandson"},
    {"name": "William Torrey", "relationship": "grandson"},
    {"name": "Joseph Torrey Jr", "relationship": "grandson"},
    {"name": "James D Torrey", "relationship": "grandson"},
    {"name": "Edward P. Torrey", "relationship": "grandson"}
  ],
  
  "quality": {
    "ocr_confidence": 0.85,
    "completeness": 0.95,
    "legibility": "good",
    "needs_review": false
  },
  
  "embedding": {
    "model": "all-MiniLM-L6-v2",
    "dimension": 384,
    "created_at": "2026-05-08T20:00:00Z"
  },
  
  "chunks": {
    "total": 1,
    "strategy": "fixed_size",
    "size": 512,
    "overlap": 50
  }
}
```

---

## Summary

**What You Should Add:**
1. ✅ **Veteran names** - Extract from titles/text
2. ✅ **Ranks and units** - Parse military information
3. ✅ **Dates** - Service dates, application dates
4. ✅ **Geographic data** - States, counties, locations
5. ✅ **Document summaries** - Key facts extraction
6. ✅ **Quality metrics** - OCR confidence, completeness

**Tools to Use:**
- spaCy or Hugging Face for NER
- Regex patterns for military info
- dateutil for date parsing
- LLM (Claude) for summarization

**Storage:**
- Keep core metadata in vector DB
- Add analytics database for complex queries
- Create separate statistics/insights dashboard

This will transform your RAG system from basic retrieval to a powerful research tool! 🚀
