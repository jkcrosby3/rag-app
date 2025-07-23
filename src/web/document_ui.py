"""
Document Management UI for the RAG system.

This module provides a web interface for managing document ingestion,
listing, and processing from various sources.
"""
import os
import json
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

import gradio as gr
import pandas as pd

from src.document_management.document_manager import DocumentManager

# Initialize document manager
document_manager = DocumentManager()


def list_documents():
    """List all documents in the system.
    
    Returns:
        DataFrame with document information
    """
    documents = document_manager.get_document_list()
    
    if not documents:
        return pd.DataFrame(columns=["ID", "Name", "Source", "Size", "Processed", "Chunked"])
        
    # Convert to DataFrame
    df = pd.DataFrame([
        {
            "ID": doc["id"],
            "Name": doc["file_name"],
            "Source": doc["source"],
            "Size": f"{doc['size_bytes'] / 1024:.1f} KB",
            "Imported": doc["imported_at"].split("T")[0],
            "Processed": "✓" if doc["processed"] else "✗",
            "Chunked": "✓" if doc["chunked"] else "✗",
            "Embedded": "✓" if doc.get("embedded", False) else "✗"
        }
        for doc in documents
    ])
    
    return df


def upload_files(files):
    """Handle file uploads.
    
    Args:
        files: List of uploaded files
        
    Returns:
        Status message and updated document list
    """
    if not files:
        return "No files uploaded", list_documents()
        
    # Create temporary directory for uploaded files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Save uploaded files to temp directory
        for file in files:
            file_path = temp_path / os.path.basename(file.name)
            with open(file_path, "wb") as f:
                f.write(file.read())
                
        # Import files
        doc_ids = document_manager.import_from_local(temp_path)
        
    return f"Uploaded {len(doc_ids)} files", list_documents()


def import_from_directory(directory_path):
    """Import documents from a local directory.
    
    Args:
        directory_path: Path to directory
        
    Returns:
        Status message and updated document list
    """
    if not directory_path:
        return "No directory specified", list_documents()
        
    try:
        doc_ids = document_manager.import_from_local(directory_path)
        return f"Imported {len(doc_ids)} documents from {directory_path}", list_documents()
    except Exception as e:
        return f"Error importing documents: {str(e)}", list_documents()


def import_from_sharepoint(library_name, folder_path, file_extensions):
    """Import documents from SharePoint.
    
    Args:
        library_name: SharePoint library name
        folder_path: Folder path within library
        file_extensions: Comma-separated list of file extensions
        
    Returns:
        Status message and updated document list
    """
    if not library_name:
        return "No SharePoint library specified", list_documents()
        
    # Parse file extensions
    extensions = None
    if file_extensions:
        extensions = [ext.strip() for ext in file_extensions.split(",")]
        
    try:
        doc_ids = document_manager.import_from_sharepoint(
            library_name=library_name,
            folder_path=folder_path,
            file_extensions=extensions
        )
        
        return f"Imported {len(doc_ids)} documents from SharePoint", list_documents()
    except Exception as e:
        return f"Error importing from SharePoint: {str(e)}", list_documents()


def process_selected_documents(document_ids):
    """Process selected documents.
    
    Args:
        document_ids: Comma-separated list of document IDs
        
    Returns:
        Status message and updated document list
    """
    if not document_ids:
        return "No documents selected", list_documents()
        
    # Parse document IDs
    doc_ids = [id.strip() for id in document_ids.split(",")]
    
    # Process documents
    stats = document_manager.process_documents(doc_ids)
    
    return f"Processed {stats.get('processed', 0)} documents", list_documents()


def process_all_documents():
    """Process all unprocessed documents.
    
    Returns:
        Status message and updated document list
    """
    stats = document_manager.process_documents()
    
    return f"Processed {stats.get('processed', 0)} documents", list_documents()


def chunk_selected_documents(document_ids):
    """Chunk selected documents.
    
    Args:
        document_ids: Comma-separated list of document IDs
        
    Returns:
        Status message and updated document list
    """
    if not document_ids:
        return "No documents selected", list_documents()
        
    # Parse document IDs
    doc_ids = [id.strip() for id in document_ids.split(",")]
    
    # Chunk documents
    stats = document_manager.chunk_documents(doc_ids)
    
    return f"Chunked {stats.get('chunked', 0)} documents", list_documents()


def chunk_all_documents():
    """Chunk all processed but not chunked documents.
    
    Returns:
        Status message and updated document list
    """
    stats = document_manager.chunk_documents()
    
    return f"Chunked {stats.get('chunked', 0)} documents", list_documents()


def run_full_pipeline():
    """Run the full document processing pipeline.
    
    Returns:
        Status message and updated document list
    """
    stats = document_manager.run_full_pipeline()
    
    return (
        f"Processed {stats.get('processed', 0)} documents, "
        f"chunked {stats.get('chunked', 0)} documents",
        list_documents()
    )


def create_ui():
    """Create the document management UI.
    
    Returns:
        Gradio interface
    """
    with gr.Blocks(title="RAG Document Management") as interface:
        gr.Markdown("# Document Management")
        
        with gr.Tab("Document List"):
            documents_df = gr.DataFrame(
                value=list_documents(),
                label="Documents",
                interactive=False
            )
            refresh_btn = gr.Button("Refresh")
            refresh_btn.click(
                fn=lambda: list_documents(),
                outputs=[documents_df]
            )
            
        with gr.Tab("Import Documents"):
            gr.Markdown("## Upload Files")
            with gr.Row():
                upload_input = gr.File(
                    label="Upload Files",
                    file_count="multiple"
                )
                upload_btn = gr.Button("Upload")
                
            gr.Markdown("## Import from Directory")
            with gr.Row():
                dir_input = gr.Textbox(
                    label="Directory Path",
                    placeholder="Enter directory path"
                )
                dir_btn = gr.Button("Import")
                
            gr.Markdown("## Import from SharePoint")
            with gr.Row():
                sp_library = gr.Textbox(
                    label="Library Name",
                    placeholder="Enter SharePoint library name"
                )
                sp_folder = gr.Textbox(
                    label="Folder Path (optional)",
                    placeholder="Enter folder path within library"
                )
                sp_extensions = gr.Textbox(
                    label="File Extensions (optional)",
                    placeholder="pdf,docx,txt"
                )
                sp_btn = gr.Button("Import")
                
            import_status = gr.Textbox(
                label="Import Status",
                interactive=False
            )
            
            # Connect buttons
            upload_btn.click(
                fn=upload_files,
                inputs=[upload_input],
                outputs=[import_status, documents_df]
            )
            
            dir_btn.click(
                fn=import_from_directory,
                inputs=[dir_input],
                outputs=[import_status, documents_df]
            )
            
            sp_btn.click(
                fn=import_from_sharepoint,
                inputs=[sp_library, sp_folder, sp_extensions],
                outputs=[import_status, documents_df]
            )
            
        with gr.Tab("Process Documents"):
            gr.Markdown("## Process Selected Documents")
            with gr.Row():
                process_ids = gr.Textbox(
                    label="Document IDs (comma-separated)",
                    placeholder="Enter document IDs to process"
                )
                process_btn = gr.Button("Process Selected")
                
            gr.Markdown("## Process All Documents")
            process_all_btn = gr.Button("Process All Unprocessed")
            
            gr.Markdown("## Chunk Selected Documents")
            with gr.Row():
                chunk_ids = gr.Textbox(
                    label="Document IDs (comma-separated)",
                    placeholder="Enter document IDs to chunk"
                )
                chunk_btn = gr.Button("Chunk Selected")
                
            gr.Markdown("## Chunk All Documents")
            chunk_all_btn = gr.Button("Chunk All Processed")
            
            gr.Markdown("## Run Full Pipeline")
            pipeline_btn = gr.Button("Run Full Pipeline")
            
            process_status = gr.Textbox(
                label="Processing Status",
                interactive=False
            )
            
            # Connect buttons
            process_btn.click(
                fn=process_selected_documents,
                inputs=[process_ids],
                outputs=[process_status, documents_df]
            )
            
            process_all_btn.click(
                fn=process_all_documents,
                outputs=[process_status, documents_df]
            )
            
            chunk_btn.click(
                fn=chunk_selected_documents,
                inputs=[chunk_ids],
                outputs=[process_status, documents_df]
            )
            
            chunk_all_btn.click(
                fn=chunk_all_documents,
                outputs=[process_status, documents_df]
            )
            
            pipeline_btn.click(
                fn=run_full_pipeline,
                outputs=[process_status, documents_df]
            )
            
    return interface


def launch_ui(share=False):
    """Launch the document management UI.
    
    Args:
        share: Whether to create a public link
    """
    interface = create_ui()
    interface.launch(share=share)


if __name__ == "__main__":
    launch_ui()
