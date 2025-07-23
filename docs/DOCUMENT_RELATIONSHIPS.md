# Document Relationship Management

This document explains how to use and maintain the document relationship management system in the RAG application. The system allows you to organize documents without renaming them in their original location (SharePoint).

## Overview

The relationship management system consists of two main components:

1. **Metadata Registry** - Stores document relationships (parent/child, related documents, versions)
2. **Tag-Based System** - Organizes documents using categories and tags

Both systems are integrated with the document manager and maintain version history to track changes over time.

## Automatic vs. Manual Maintenance

The relationship system supports both automatic and manual maintenance:

### Automatic Maintenance

- **Document Registration**: When documents are imported, they are automatically registered in the relationship system
- **Metadata Extraction**: Basic metadata is extracted from documents and stored
- **Version History**: All changes to relationships and tags are automatically versioned

### Manual Maintenance

- **Adding Relationships**: Users can manually define relationships between documents
- **Tagging**: Users can add/remove tags and categories to documents
- **Snapshots**: Users can create named snapshots before making significant changes

## Using the Relationship Manager

### Adding Document Relationships

```python
# Example: Mark document B as a child of document A
document_manager.add_document_relationship(
    source_doc_id="doc_A_id",
    target_doc_id="doc_B_id",
    relationship_type="parent",
    bidirectional=True  # This will also add doc_A as parent of doc_B
)

# Example: Mark documents as related
document_manager.add_document_relationship(
    source_doc_id="doc_A_id",
    target_doc_id="doc_C_id",
    relationship_type="related"
)
```

### Working with Tags

```python
# Add a tag with category
document_manager.add_document_tag(
    doc_id="doc_A_id",
    tag="quarterly_report",
    category="document_type"
)

# Add a tag without category
document_manager.add_document_tag(
    doc_id="doc_A_id",
    tag="important"
)

# Find documents by tag
docs = document_manager.get_documents_by_tag(
    tag="quarterly_report",
    category="document_type"
)
```

### Managing Versions

```python
# Create a snapshot before reorganizing
snapshot_file = document_manager.create_relationship_snapshot("before_reorganization")

# Make changes to relationships...

# If needed, revert to previous snapshot
document_manager.load_relationship_snapshot(snapshot_file)

# View available versions
versions = document_manager.get_relationship_versions()
print(versions)  # Shows {'relationships': [...], 'tags': [...]}
```

## Best Practices

### Relationship Types

Use consistent relationship types across your document collection:

- **parent/child**: Hierarchical relationships (e.g., annual report contains quarterly reports)
- **related**: Documents that are related but not hierarchically
- **version_of**: Different versions of the same document
- **supersedes**: A document that replaces an older one
- **references**: A document that references another

### Tag Categories

Organize tags into meaningful categories:

- **document_type**: Report, Presentation, Contract, etc.
- **business_unit**: Finance, HR, Marketing, etc.
- **project**: Project names
- **status**: Draft, Final, Approved, etc.
- **time_period**: Q1_2025, Q2_2025, etc.

### Versioning Strategy

- Create snapshots before major reorganizations
- Use descriptive names for snapshots
- Review version history periodically to track changes

## Technical Implementation

The relationship system is implemented in two main classes:

1. **RelationshipManager** (`src/document_management/relationship_manager.py`)
   - Core functionality for managing relationships and tags
   - Handles versioning and snapshots
   - Maintains the relationship and tag registries

2. **DocumentManager** (`src/document_management/document_manager.py`)
   - Integrates with the RelationshipManager
   - Provides high-level methods for document relationship management
   - Handles document registration and metadata

### Data Storage

Relationship data is stored in JSON files:

- `data/metadata/document_relationships.json` - Main relationship registry
- `data/metadata/document_tags.json` - Tag registry
- `data/metadata/versions/` - Version history and snapshots

## Handling Special Cases

### Document Renaming in SharePoint

If a document is renamed in SharePoint:

1. The system will maintain the relationship to the original document ID
2. When re-importing, you can use the `update_document_metadata` method to update the filename

### Document Deletion

If a document is deleted from SharePoint:

1. Its relationships remain in the registry
2. You can manually remove it or mark it as deleted using tags

### Bulk Updates

For bulk updates to document relationships:

1. Create a snapshot first
2. Use the Python API to make changes programmatically
3. Verify changes before committing

## Conclusion

This relationship management system allows you to maintain rich document organization without requiring changes to the original document names or locations in SharePoint. By using metadata, relationships, and tags, you can create a flexible organization system that meets your needs while preserving the original document structure.
