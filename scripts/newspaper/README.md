# Newspaper Metadata Enrichment Scripts

This folder contains scripts for extracting and enriching metadata from Revolutionary War era newspaper text files.

## Quick Start

```bash
# 1. Enrich all newspapers with metadata (fast, ~5 seconds)
python scripts/newspaper/enrich_newspaper_files.py

# 2. (Optional) Improve OCR quality - see OCR_SETUP.md for installation
python scripts/newspaper/improve_ocr.py --limit 5 --compare

# 3. (Optional) Cross-reference veterans with newspapers
python scripts/newspaper/cross_reference_veterans.py
```

**See also**:

- 📖 [OCR_SETUP.md](OCR_SETUP.md) - Tesseract installation guide
- 📊 [NEWSPAPER_ENRICHMENT_SUMMARY.md](NEWSPAPER_ENRICHMENT_SUMMARY.md) - Complete project summary

## Main Scripts

### `enrich_newspaper_files.py`

**Purpose**: Extract structured metadata from newspaper text files to enable better search and discovery.

**Extracts**:

- **Named Entities**:
  - People mentioned (military officers, politicians, civilians)
  - Places mentioned (cities, states, countries, battlefields)
  - Military units (regiments, militias, organizations)
  
- **War-Related Content**:
  - Battles/engagements mentioned (Yorktown, Saratoga, etc.)
  - War keywords by category (military, combat, personnel, supplies, operations)
  - War relevance score (0-1 scale)
  
- **Article Classification**:
  - Subject tags (military_news, political, economic, advertisement, etc.)
  - Entity count (for relevance ranking)
  - Has war content flag

**Usage**:

```bash
python scripts/newspaper/enrich_newspaper_files.py
```

**Output**: Updates existing `*_metadata.json` files with enriched metadata fields.

### `improve_ocr.py`

**Purpose**: Re-process newspaper images with modern OCR (Tesseract) to improve text quality compared to original OCR.

**Features**:

- Downloads newspaper images from Library of Congress URLs
- Preprocesses images (contrast enhancement, resizing)
- Runs Tesseract OCR with historical newspaper settings
- Compares quality with original OCR
- Saves improved text as `*_improved.txt`

**Requirements**:

```bash
pip install -r scripts/newspaper/requirements_ocr.txt
```

Also requires Tesseract OCR installed:

- **Windows**: <https://github.com/UB-Mannheim/tesseract/wiki>
- **Mac**: `brew install tesseract`
- **Linux**: `sudo apt-get install tesseract-ocr`

**Usage**:

```bash
# Process all newspapers
python scripts/newspaper/improve_ocr.py

# Process first 5 newspapers (test run)
python scripts/newspaper/improve_ocr.py --limit 5

# Process and compare with original OCR
python scripts/newspaper/improve_ocr.py --limit 5 --compare

# Process specific newspaper
python scripts/newspaper/improve_ocr.py --lccn sn85025609
```

**Output**: Creates `*_improved.txt` files with better OCR quality for metadata extraction.

**Note**: This is a **CPU-intensive** process. Each newspaper takes 1-2 minutes. Use `--limit` for testing.

### `cross_reference_veterans.py`

**Purpose**: Find connections between pension files and newspaper mentions by matching veteran names.

**Usage**:

```bash
python scripts/newspaper/cross_reference_veterans.py
```

**Output**: Creates `veteran_newspaper_cross_references.json` with matched veterans and their newspaper mentions.

## Metadata Schema

The enrichment adds the following fields to newspaper metadata:

```json
{
  "people_mentioned": ["George Washington", "General Howe"],
  "places_mentioned": ["Yorktown", "Virginia", "New York"],
  "battles_mentioned": ["Yorktown", "Saratoga"],
  "military_units": ["Continental Army", "5th Regiment"],
  "war_keywords": {
    "military": ["army", "troops", "militia"],
    "combat": ["battle", "victory"],
    "personnel": ["enlistment", "prisoner"]
  },
  "subject_tags": ["military_news", "political"],
  "entity_count": 15,
  "war_relevance_score": 0.65,
  "has_war_content": true
}
```

## Implementation Approach

**Hybrid Rule-Based System**:

- Regex patterns for war-specific terms (battles, military ranks, units)
- Gazetteer matching for place names
- Keyword frequency analysis
- Entity density calculation for relevance ranking

**Benefits**:

- Fast, CPU-only processing
- No external API dependencies
- Tuned for Revolutionary War era content
- Enables semantic search and filtering

## Use Cases

The enriched metadata enables:

1. **Semantic Search**: Find articles by person, place, or battle
2. **Filtering**: Filter by subject tags or war relevance
3. **Relevance Ranking**: Rank results by entity density
4. **Cross-Referencing**: Link pension files with newspaper mentions
5. **Topic Discovery**: Identify themes and patterns across articles

## OCR Quality Limitations

**Important**: Historical newspaper OCR quality is often poor due to:

- **18th century printing** - Faded ink, worn type, damaged paper
- **Old fonts** - Gothic/Old English fonts that modern OCR struggles with
- **Historical spelling** - "fhould" vs "should", long s (ſ) looks like f
- **Layout issues** - Multi-column layouts read incorrectly
- **Character confusion** - Common OCR errors: rn→m, cl→d, vv→w, ii→u

**Mitigation Strategies**:

1. **OCR Normalization**: Script normalizes common errors (long s, character substitutions)
2. **OCR Quality Assessment**: Each file gets a quality rating (good/fair/poor)
3. **Fuzzy Matching**: Name extraction uses flexible patterns to catch variations
4. **Extraction Note**: Metadata includes warning about OCR quality
5. **Cross-Reference Validation**: Use multiple sources to confirm veteran mentions

**Metadata Fields Added**:

- `ocr_quality`: "good", "fair", or "poor" based on text analysis
- `extraction_note`: Warning about OCR limitations

**Recommendation**: When cross-referencing veterans with newspapers, treat matches as **potential leads** requiring manual verification, not definitive proof.

## Customization

To add more entities or keywords, edit the constants at the top of `enrich_newspaper_files.py`:

- `BATTLES` - Revolutionary War battles
- `PLACES` - Colonial cities and states
- `MILITARY_UNITS` - Army/navy units
- `NOTABLE_PEOPLE` - Political and military figures
- `WAR_KEYWORDS` - War-related terms by category

## Comparison with Pension Enrichment

| Feature | Pension Files | Newspaper Files |
|---------|--------------|-----------------|
| **Focus** | Individual veteran data | Historical events/news |
| **Entities** | Veteran name, rank, unit | People, places, battles |
| **Structure** | Highly structured forms | Unstructured articles |
| **Approach** | Pattern matching on forms | Entity extraction from text |
| **Accuracy** | 99.9% (veteran names) | Variable (OCR quality) |

## Future Enhancements

Potential improvements:

- Named Entity Recognition (NER) with spaCy for better person/place extraction
- Sentiment analysis (victory/defeat, optimistic/pessimistic)
- Article type classification (news, proclamation, letter, advertisement)
- Date extraction for events mentioned
- Cross-document entity linking
