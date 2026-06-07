# Pension File Metadata Extraction Scripts

This folder contains all scripts related to extracting and enriching metadata from Revolutionary War pension files.

## Main Scripts

### `enrich_pension_files.py`
**Purpose**: Main enrichment script that extracts structured metadata from pension text files.

**Extracts**:
- Veteran name (99.9%+ accuracy)
- Military rank
- Military unit
- Service dates
- Pension amounts
- Death information
- Family information

**Usage**:
```bash
python scripts/pension/enrich_pension_files.py
```

**Output**: Creates `*_metadata.json` files alongside each `.txt` file with extracted metadata.

### `llm_enrich_pension_files.py`
**Purpose**: Alternative LLM-based enrichment approach (experimental).

**Note**: Requires LLM API access. More accurate but slower and requires API costs.

### `find_missing_names.py`
**Purpose**: Validation script to find pension files missing veteran names.

**Usage**:
```bash
python scripts/pension/find_missing_names.py
```

### `count_soldiers.py`
**Purpose**: Counts and analyzes soldier statistics from pension metadata.

## Testing Scripts

- `test_17_files.py` - Test extraction on specific subset of files
- `test_final_6.py` - Test extraction on edge cases
- `test_multi_veteran.py` - Test multi-veteran file detection
- `test_single_file.py` - Test extraction on single file
- `test_existing_patterns.py` - Test regex patterns

## Debug Scripts

- `debug_full_extraction.py` - Full extraction debugging
- `debug_pattern_order.py` - Pattern matching order debugging
- `debug_patterns.py` - Regex pattern testing
- `debug_remaining.py` - Debug remaining unmatched files
- `debug_single.py` - Single pattern debugging
- `debug_title_patterns.py` - Title line pattern debugging
- `debug_line80.py` - Specific line pattern debugging
- `analyze_edge_cases.py` - Edge case analysis

## Pattern Development

The extraction patterns evolved through multiple iterations to handle:
- Two-word names (John Smith)
- Three-word names (John Henry Smith)
- Multi-word surnames (Van Buskirk, De la Cruz)
- Letter prefixes (A Mack, Abner)
- Multi-veteran files (multiple warrant cards)
- Administrative documents (NARA sheets)
- Empty title lines
- Various title formats

## Results

**Final Statistics** (Run 10):
- Total files: 12,607
- Veteran names extracted: 12,606 (100.0%)
- Ranks: 9,541 (75.7%)
- Units: 10,952 (86.9%)
- Service dates: 5,171 (41.0%)
- Pension amounts: 5,654 (44.8%)
- Death info: 5,567 (44.2%)
- Family info: 7,344 (58.3%)

## Notes

These scripts are **pension-specific** and use regex patterns tailored to Revolutionary War pension file formats. For newspaper enrichment, see `scripts/newspaper/` (to be created).
