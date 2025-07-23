#!/usr/bin/env python
"""
Script to create a sample directory structure for business plan documents.
This creates the directory structure and placeholder files for testing the
document organization system described in docs/DOCUMENT_ORGANIZATION.md.
"""

import os
import json
from pathlib import Path
import logging
import argparse
import datetime
import shutil
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Base directory for documents
DEFAULT_BASE_DIR = Path("data/documents/plans")

# Sample plan numbers
PLAN_NUMBERS = ["234456", "567890", "789012"]

# Sample annexes (by letter)
ANNEXES = ["A", "B", "T"]

# Sample appendices (by letter) for each annex
APPENDICES = {
    "A": ["A", "B", "C"],
    "B": ["D", "E", "F"],
    "T": ["G", "H", "I"]
}

# Sample supporting document types
SUPPORTING_DOCS = [
    "market_analysis", 
    "financial_projections", 
    "risk_assessment",
    "stakeholder_feedback",
    "technical_specifications"
]

# Sample statuses
STATUSES = ["DRAFT", "APPROVED", "EXECUTED"]

# Sample departments
DEPARTMENTS = ["Finance", "Operations", "IT", "HR", "Marketing"]

def create_directory_structure(base_dir: Path) -> None:
    """
    Create the base directory structure for plans.
    
    Args:
        base_dir: Base directory for documents
    """
    # Create plans directory
    plans_dir = base_dir
    plans_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created base directory: {plans_dir}")
    
    # Create templates directory
    templates_dir = base_dir.parent / "templates"
    templates_dir.mkdir(parents=True, exist_ok=True)
    
    for template_type in ["main", "annex", "appendix"]:
        template_subdir = templates_dir / f"{template_type}_templates"
        template_subdir.mkdir(exist_ok=True)
    
    logger.info(f"Created templates directory: {templates_dir}")
    
    # Create semantic categories directory
    semantic_dir = base_dir.parent / "semantic_categories"
    semantic_dir.mkdir(parents=True, exist_ok=True)
    
    for category in ["financial", "staffing", "technology", "operations"]:
        category_dir = semantic_dir / category
        category_dir.mkdir(exist_ok=True)
    
    logger.info(f"Created semantic categories directory: {semantic_dir}")

def create_placeholder_file(file_path: Path, content: str = "This is a placeholder document.") -> None:
    """
    Create a placeholder text file.
    
    Args:
        file_path: Path to the file
        content: Content to write to the file
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    logger.info(f"Created placeholder file: {file_path}")

def create_metadata_file(file_path: Path, metadata: Dict[str, Any]) -> None:
    """
    Create a metadata JSON file.
    
    Args:
        file_path: Path to the file
        metadata: Metadata to write to the file
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Created metadata file: {file_path}")

def create_plan_documents(plan_number: str, base_dir: Path, status: str = "DRAFT") -> None:
    """
    Create documents for a single plan.
    
    Args:
        plan_number: Plan number
        base_dir: Base directory for documents
        status: Status of the plan
    """
    # Create plan directory
    plan_dir = base_dir / f"plan_{plan_number}"
    plan_dir.mkdir(exist_ok=True)
    
    # Create main plan document
    main_doc_path = plan_dir / f"plan_{plan_number}-main-{status}.txt"
    main_doc_content = f"""
# Strategic Plan {plan_number}

## Executive Summary
This is the main document for plan {plan_number}.

## Objectives
1. Objective 1
2. Objective 2
3. Objective 3

## Timeline
- Phase 1: Q1 2025
- Phase 2: Q2 2025
- Phase 3: Q3-Q4 2025

## Annexes
This plan includes the following annexes:
{', '.join([f'Annex {annex}' for annex in ANNEXES])}
"""
    create_placeholder_file(main_doc_path, main_doc_content)
    
    # Create metadata for main document
    main_metadata = {
        "plan_number": plan_number,
        "document_type": "main",
        "hierarchy_level": 1,
        "date": datetime.datetime.now().strftime("%Y-%m-%d"),
        "status": status,
        "keywords": ["strategic plan", "objectives", "timeline"],
        "author": "CEO Office",
        "department": "Executive",
        "revision": "1.0"
    }
    create_metadata_file(plan_dir / f"plan_{plan_number}-main-{status}.metadata.json", main_metadata)
    
    # Create relationship mapping file
    relationship_map = {
        "plan_id": plan_number,
        "main_document": f"plan_{plan_number}-main-{status}.txt",
        "status": status,
        "annexes": []
    }
    
    # Create annexes
    for annex in ANNEXES:
        annex_doc_path = plan_dir / f"plan_{plan_number}-annex{annex}-{status}.txt"
        department = DEPARTMENTS[ANNEXES.index(annex) % len(DEPARTMENTS)]
        
        annex_content = f"""
# Annex {annex}: {department} Implementation Plan

## Purpose
This annex details how the {department} department will implement Plan {plan_number}.

## Requirements
1. Requirement A{annex}1
2. Requirement A{annex}2
3. Requirement A{annex}3

## Questions to Address
{', '.join([f'Q{i+1}: How will we implement requirement A{annex}{i+1}?' for i in range(3)])}

## Appendices
This annex includes the following appendices:
{', '.join([f'Appendix {app}' for app in APPENDICES[annex]])}
"""
        create_placeholder_file(annex_doc_path, annex_content)
        
        # Create metadata for annex
        annex_metadata = {
            "plan_number": plan_number,
            "document_type": f"annex{annex}",
            "hierarchy_level": 2,
            "date": datetime.datetime.now().strftime("%Y-%m-%d"),
            "status": status,
            "keywords": [f"{department.lower()} implementation", "requirements"],
            "author": f"{department} Director",
            "department": department,
            "revision": "1.0"
        }
        create_metadata_file(plan_dir / f"plan_{plan_number}-annex{annex}-{status}.metadata.json", annex_metadata)
        
        # Add to relationship map
        annex_map = {
            "id": annex,
            "file": f"plan_{plan_number}-annex{annex}-{status}.txt",
            "department": department,
            "appendices": []
        }
        
        # Create appendices for this annex
        for appendix in APPENDICES[annex]:
            appendix_doc_path = plan_dir / f"plan_{plan_number}-annex{annex}-appendix{appendix}-{status}.txt"
            
            appendix_content = f"""
# Appendix {appendix} for Annex {annex}

## Purpose
This appendix answers questions raised in Annex {annex}.

## Answer to Q1
Detailed answer to how we will implement requirement A{annex}1.

## Answer to Q2
Detailed answer to how we will implement requirement A{annex}2.

## Answer to Q3
Detailed answer to how we will implement requirement A{annex}3.

## Supporting Documents
- Market Analysis
- Financial Projections
"""
            create_placeholder_file(appendix_doc_path, appendix_content)
            
            # Create metadata for appendix
            appendix_metadata = {
                "plan_number": plan_number,
                "document_type": f"annex{annex}-appendix{appendix}",
                "hierarchy_level": 3,
                "date": datetime.datetime.now().strftime("%Y-%m-%d"),
                "status": status,
                "keywords": ["implementation details", f"answers for annex {annex}"],
                "author": f"{department} Team Lead",
                "department": department,
                "revision": "1.0",
                "answers_questions": [f"Q{i+1}" for i in range(3)]
            }
            create_metadata_file(
                plan_dir / f"plan_{plan_number}-annex{annex}-appendix{appendix}-{status}.metadata.json", 
                appendix_metadata
            )
            
            # Add to relationship map
            appendix_map = {
                "id": appendix,
                "file": f"plan_{plan_number}-annex{annex}-appendix{appendix}-{status}.txt",
                "answers_questions": [f"Q{i+1}" for i in range(3)]
            }
            annex_map["appendices"].append(appendix_map)
        
        relationship_map["annexes"].append(annex_map)
    
    # Create supporting documents directory
    supporting_dir = plan_dir / f"plan_{plan_number}-supporting"
    supporting_dir.mkdir(exist_ok=True)
    
    # Create supporting documents
    for doc_type in SUPPORTING_DOCS:
        support_doc_path = supporting_dir / f"plan_{plan_number}-supporting-{doc_type}.txt"
        
        support_content = f"""
# {doc_type.replace('_', ' ').title()} for Plan {plan_number}

## Purpose
This document provides supporting information for Plan {plan_number}.

## Content
Detailed {doc_type.replace('_', ' ')} information.

## Related Plan Components
This document supports various annexes and appendices in Plan {plan_number}.
"""
        create_placeholder_file(support_doc_path, support_content)
        
        # Create metadata for supporting document
        support_metadata = {
            "plan_number": plan_number,
            "document_type": "supporting",
            "supporting_type": doc_type,
            "hierarchy_level": 0,  # Outside the main hierarchy
            "date": datetime.datetime.now().strftime("%Y-%m-%d"),
            "status": status,
            "keywords": [doc_type.replace('_', ' ')],
            "author": "Research Team",
            "department": "Research",
            "revision": "1.0"
        }
        create_metadata_file(
            supporting_dir / f"plan_{plan_number}-supporting-{doc_type}.metadata.json", 
            support_metadata
        )
    
    # Save relationship map
    create_metadata_file(plan_dir / f"plan_{plan_number}-relationships.json", relationship_map)

def create_question_mapping_file(base_dir: Path) -> None:
    """
    Create a question mapping file that maps questions to documents.
    
    Args:
        base_dir: Base directory for documents
    """
    question_dir = base_dir.parent / "question_mappings"
    question_dir.mkdir(parents=True, exist_ok=True)
    
    # Create sample question mappings
    questions = [
        {
            "question_id": "Q1",
            "question_text": "How will we implement requirement A1?",
            "answered_in": [
                {"plan": "234456", "document": "plan_234456-annexA-appendixA-DRAFT.txt"},
                {"plan": "567890", "document": "plan_567890-annexA-appendixA-DRAFT.txt"}
            ]
        },
        {
            "question_id": "Q2",
            "question_text": "How will we implement requirement B2?",
            "answered_in": [
                {"plan": "234456", "document": "plan_234456-annexB-appendixE-DRAFT.txt"},
                {"plan": "567890", "document": "plan_567890-annexB-appendixE-DRAFT.txt"}
            ]
        }
    ]
    
    for q in questions:
        q_file = question_dir / f"{q['question_id']}.json"
        create_metadata_file(q_file, q)

def main():
    """Main function to create the sample plan structure."""
    parser = argparse.ArgumentParser(description="Create sample plan directory structure")
    parser.add_argument("--base-dir", type=str, default=DEFAULT_BASE_DIR,
                       help="Base directory for documents")
    args = parser.parse_args()
    
    base_dir = Path(args.base_dir)
    
    # Create the directory structure
    create_directory_structure(base_dir)
    
    # Create documents for each plan
    for i, plan_number in enumerate(PLAN_NUMBERS):
        # Alternate statuses for different plans
        status = STATUSES[i % len(STATUSES)]
        create_plan_documents(plan_number, base_dir, status)
    
    # Create question mapping file
    create_question_mapping_file(base_dir)
    
    logger.info(f"Sample plan structure created successfully in {base_dir}")

if __name__ == "__main__":
    main()
