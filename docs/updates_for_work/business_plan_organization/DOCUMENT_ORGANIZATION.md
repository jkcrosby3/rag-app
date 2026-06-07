# Business Plan Document Organization

This document outlines the organization structure for business plan documents in the RAG application.

## Document Organization Structure

```
data/
  documents/
    plans/
      plan_234456/
        plan_234456-main.pdf
        plan_234456-annexT.pdf
        plan_234456-annexT-appendixG.pdf
        plan_234456-supporting/
          plan_234456-supporting-market_analysis.pdf
          plan_234456-supporting-financial_projections.pdf
      plan_567890/
        plan_567890-main.pdf
        ...
```

## Naming Convention

- **Primary identifier**: Plan number (e.g., `plan_234456`)
- **Document type**: Main document, annex, appendix (e.g., `-main`, `-annexT`, `-annexT-appendixG`)
- **Supporting documents**: Place in a subdirectory with descriptive names (e.g., `-supporting-market_analysis`)

## Enhanced Features

- **Version Control**: Track document versions in separate subdirectories
- **Templates**: Store high-quality examples as templates for future use
- **Outcome Tracking**: Track the results of implemented plans
- **Semantic Categorization**: Alternative views based on topic
- **Status Tracking**: Include document status in filenames
- **Relationship Mapping**: JSON/YAML files mapping document relationships
- **Question-Answer Mapping**: Map questions to documents containing answers

### Examples of Enhanced Features

#### Version Control Example
```
data/
  documents/
    plans/
      plan_234456/
        versions/
          v1.0/
            plan_234456-main-v1.0.pdf
          v1.1/
            plan_234456-main-v1.1.pdf
        current/
          plan_234456-main.pdf  # Symlink to latest version
```

#### Templates Example
```
data/
  documents/
    templates/
      annex_templates/
        financial_annex_template.pdf
        staffing_annex_template.pdf
      appendix_templates/
        budget_appendix_template.pdf
```

#### Outcome Tracking Example
```
data/
  documents/
    plans/
      plan_234456/
        outcomes/
          plan_234456-outcomes-6month.pdf
          plan_234456-outcomes-1year.pdf
          plan_234456-outcomes-metrics.json
```

#### Semantic Categorization Example
```
data/
  documents/
    semantic_categories/
      financial/
        plan_234456-annexT-appendixG.pdf  # Symlink
        plan_567890-annexB-appendixC.pdf  # Symlink
      staffing/
        plan_234456-annexB-appendixA.pdf  # Symlink
```

#### Status Tracking Example
```
plan_234456-annexT-appendixG-DRAFT.pdf
plan_234456-annexT-appendixG-APPROVED.pdf
plan_234456-annexT-appendixG-EXECUTED.pdf
```

#### Relationship Mapping Example
```json
{
  "plan_id": "234456",
  "main_document": "plan_234456-main.pdf",
  "annexes": [
    {
      "id": "T",
      "file": "plan_234456-annexT.pdf",
      "appendices": [
        {
          "id": "G",
          "file": "plan_234456-annexT-appendixG.pdf",
          "answers_questions": ["Q1", "Q3"],
          "supporting_documents": ["plan_234456-supporting-market_analysis.pdf"]
        }
      ]
    }
  ],
  "related_plans": ["123456", "345678"]
}
```

#### Question-Answer Mapping Example
```json
{
  "question_id": "Q1",
  "question_text": "How will we implement requirement AT1?",
  "answered_in": [
    {"plan": "234456", "document": "plan_234456-annexT-appendixG-DRAFT.pdf"},
    {"plan": "567890", "document": "plan_567890-annexT-appendixG-DRAFT.pdf"}
  ]
}
```

## Metadata Structure

Each document should include:

1. **Plan number**: The unique identifier for the plan
2. **Document type**: Main, annex, appendix, or supporting
3. **Hierarchy level**: 1 (main plan), 2 (annex), 3 (appendix), etc.
4. **Date**: Document creation/revision date
5. **Status**: Draft, approved, executable, etc.
6. **Keywords/Tags**: Specific topics covered
7. **Success Metrics**: Whether the plan/section was successfully implemented
8. **Author/Department**: Who created the document
9. **Revision History**: Track changes over time
10. **Cross-References**: Explicit links to related documents

## Sample Document Creation Plan

To implement and test this organization structure:

1. **Create Sample Plan Structure**
   - Create 2-3 sample plans with different numbers
   - For each plan, create a main document, 2-3 annexes, and 2-3 appendices per annex
   - Add supporting documents for each plan

2. **Document Types to Create**
   - Main plan documents outlining overall strategy
   - Annexes with detailed requirements for different departments
   - Appendices answering specific questions from annexes
   - Supporting documents with research, analysis, etc.

3. **Sample Content**
   - Use realistic business plan content
   - Ensure hierarchical relationships between documents
   - Include questions in annexes that are answered in appendices

4. **Implementation Steps**
   - Create the directory structure
   - Generate sample PDF documents
   - Create metadata JSON files for each document
   - Create relationship mapping files

5. **Testing**
   - Test document ingestion into the RAG system
   - Test retrieval based on plan number, document type, etc.
   - Test question answering using the hierarchical structure

## Next Steps

- [ ] Create directory structure for sample documents
- [ ] Generate sample PDF content for plans, annexes, appendices
- [ ] Implement metadata extraction in the document processing pipeline
- [ ] Update the RAG system to leverage the enhanced organization
