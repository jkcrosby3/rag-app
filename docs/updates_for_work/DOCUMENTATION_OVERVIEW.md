# Business Plan Document System Documentation

This document provides an overview of the business plan document system, including its components, relationships, and documentation.

## 1. Document Organization Structure

### Directory Structure
```
docs/updates_for_work/
├── business_plan_organization/
│   ├── DOCUMENT_ORGANIZATION.md         # Main organization structure documentation
│   ├── IMPLEMENTATION_STEPS.md          # Implementation guide
│   ├── Q1.json                          # Question mapping example
│   ├── create_sample_plan_structure.py  # Pipeline script
│   ├── plan_234456-main-DRAFT.txt       # Sample document
│   ├── plan_234456-main-DRAFT.metadata.json  # Document metadata
│   ├── plan_234456-relationships.json   # Document relationships
│   └── update_document_processor.py     # Document processing script
└── business_plan_organization.zip       # Archive of all documentation
```

## 2. Document Types and Relationships

### 2.1 Main Documentation Files

- **DOCUMENT_ORGANIZATION.md**: Describes the overall document organization structure, including:
  - Directory structure
  - Naming conventions
  - Enhanced features (version control, templates, outcome tracking)
  - Semantic categorization
  - Status tracking

- **IMPLEMENTATION_STEPS.md**: Provides step-by-step implementation guide:
  - Directory creation
  - Document naming
  - Metadata creation
  - Relationship mapping
  - Question-answer mapping

### 2.2 Example Files

- **Q1.json**: Example question mapping file
- **plan_234456-main-DRAFT.txt**: Sample document
- **plan_234456-main-DRAFT.metadata.json**: Example metadata file
- **plan_234456-relationships.json**: Example relationships mapping

## 3. Pipeline Documentation

### 3.1 Pipeline Scripts

- **create_sample_plan_structure.py**: Script for creating sample directory structure
  - Creates test directory structure
  - Generates placeholder files
  - Used for testing organization system

- **update_document_processor.py**: Document processing script
  - Handles document updates
  - Maintains relationships
  - Updates metadata

## 4. Document Relationships

### 4.1 Metadata Structure
```json
{
    "plan_number": "234456",
    "document_type": "main",
    "hierarchy_level": 1,
    "date": "2025-06-13",
    "status": "DRAFT",
    "keywords": ["strategic plan", "objectives", "timeline"],
    "author": "CEO Office",
    "department": "Executive",
    "revision": "1.0"
}
```

### 4.2 Relationships Mapping
```json
{
    "plan_id": "234456",
    "main_document": "plan_234456-main-DRAFT.txt",
    "status": "DRAFT",
    "annexes": [
        {
            "id": "A",
            "file": "plan_234456-annexA-DRAFT.txt",
            "department": "Finance",
            "appendices": [
                {
                    "id": "A",
                    "file": "plan_234456-annexA-appendixA-DRAFT.txt",
                    "answers_questions": ["Q1", "Q2", "Q3"]
                }
            ]
        }
    ]
}
```

## 5. File Preparation Process

### 5.1 File Renaming Convention
- All files must include full path structure in filename
- Replace directory separators with underscores
- Add .txt extension to all files
- Example: `src_tools_document_processor.py.txt`

### 5.2 Archive Creation
- Create zip archive of renamed files
- Use format: `updated_files_YYYY-MM-DD_renamed_with_paths.zip`
- Verify Gmail compatibility

## 6. Usage

### 6.1 Creating New Documents
1. Follow naming convention
2. Create metadata file
3. Update relationships mapping
4. Run pipeline scripts

### 6.2 Updating Documents
1. Update metadata
2. Update relationships
3. Run processing scripts
4. Create new archive

## 7. Maintenance

- Regularly review and update documentation
- Verify pipeline scripts
- Update example files
- Maintain compatibility with email systems

## 8. References
- Original documentation in `business_plan_organization/`
- Implementation guide in `IMPLEMENTATION_STEPS.md`
- Organization structure in `DOCUMENT_ORGANIZATION.md`
