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

from ..document_management.document_manager import DocumentManager

# Initialize document manager
document_manager = DocumentManager()

# Run document initialization in a separate thread
def initialize_documents():
    """Initialize documents in the background"""
    try:
        # Wait a bit to ensure UI is fully initialized
        import time
        time.sleep(1)
        
        # Get existing documents
        existing_docs = set(document_manager.registry.get('documents', {}).keys())
        
        # Register new documents
        for file_path in Path('data/documents').glob("**/*.*"):
            if file_path.name not in existing_docs:
                try:
                    document_manager.register_document(
                        file_path=str(file_path),
                        source="existing",
                        metadata={"original_path": str(file_path)}
                    )
                    print(f"Registered existing document: {file_path}")
                except Exception as e:
                    print(f"Failed to register {file_path}: {e}")
    except Exception as e:
        print(f"Error during document registration: {e}")

# Start document initialization in background thread
import threading
threading.Thread(target=initialize_documents, daemon=True, name="document_initializer").start()

def list_documents():
    """List all documents in the system.
    
    Returns:
        DataFrame with document information
    """
    documents = document_manager.get_all_documents()
    
    if not documents:
        return pd.DataFrame(columns=["ID", "Name", "Source", "Size", "Processed", "Chunked"])
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
        
    results = []
    for file in files:
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(file.read())
                temp_path = Path(temp_file.name)
                
            # Import the document
            doc_id = document_manager.import_document(str(temp_path))
            results.append(f"Successfully imported {file.name} with ID {doc_id}")
            
            # Clean up temporary file
            temp_path.unlink()
            
        except Exception as e:
            results.append(f"Failed to import {file.name}: {str(e)}")
                
    status = "\n".join(results)
    return status, list_documents()


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
        # Convert to Path object
        dir_path = Path(directory_path)
        if not dir_path.exists() or not dir_path.is_dir():
            return f"Directory does not exist: {directory_path}", list_documents()
            
        # Import all files
        results = []
        for file_path in dir_path.glob("**/*.*"):
            if file_path.is_file():
                try:
                    doc_id = document_manager.import_document(str(file_path))
                    results.append(f"Successfully imported {file_path.name} with ID {doc_id}")
                except Exception as e:
                    results.append(f"Failed to import {file_path.name}: {str(e)}")
        
        status = "\n".join(results)
        return status, list_documents()
        
    except Exception as e:
        return f"Error importing from directory: {str(e)}", list_documents()
        
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


def search_documents(query: str) -> pd.DataFrame:
    """Search documents based on query."""
    results = document_manager.search_documents(query)
    return pd.DataFrame(results)


def get_notifications() -> pd.DataFrame:
    """Get system notifications."""
    return pd.DataFrame(document_manager.get_notifications())


def edit_metadata(doc_ids: str) -> tuple[str, pd.DataFrame]:
    """Edit document metadata."""
    status = document_manager.edit_metadata(doc_ids.split(','))
    return status, list_documents()


def manage_tags(doc_ids: str) -> str:
    """Manage document tags."""
    return ', '.join(document_manager.get_document_tags(doc_ids.split(',')))


def add_tag(tag_name: str, doc_ids: str) -> str:
    """Add a new tag to documents."""
    return document_manager.add_tag(tag_name, doc_ids.split(','))


def view_document_detail(doc_ids: str) -> str:
    """View document details."""
    return document_manager.get_document_detail(doc_ids.split(',')[0])


def create_ui():
    """Create and launch the Gradio UI interface."""
    with gr.Blocks() as demo:
        # Navigation Bar
        with gr.Row():
            gr.Markdown("# Document Tracking System")
            with gr.Row():
                search_input = gr.Textbox(
                    label="Search",
                    placeholder="Search documents...",
                    scale=2
                )
                search_btn = gr.Button("Search", scale=1)
                stats_btn = gr.Button("System Stats", scale=1)
        
        # Filter Section
        with gr.Row():
            with gr.Column():
                classification_filter = gr.Dropdown(
                    choices=["All", "Public", "Internal", "Confidential"],
                    label="Classification",
                    value="All"
                )
            with gr.Column():
                version_filter = gr.Dropdown(
                    choices=["All Versions", "Latest Only"],
                    label="Version",
                    value="All Versions"
                )
            with gr.Column():
                size_filter = gr.Dropdown(
                    choices=["All Sizes", "< 1MB", "1MB - 10MB", "> 10MB"],
                    label="Size",
                    value="All Sizes"
                )
            with gr.Column():
                date_filter = gr.Dropdown(
                    choices=["All Dates", "Last 7 Days", "Last 30 Days", "Custom"],
                    label="Upload Date",
                    value="All Dates"
                )
        
        # Document Management Column
        with gr.Column():
            # Document List with Actions
            gr.Markdown("### Document Management")
            with gr.Row():
                upload_btn = gr.Button("Upload New Document")
                export_btn = gr.Button("Export Selected")
                
            doc_list = gr.Dataframe(
                headers=["ID", "Name", "Source", "Size", "Processed", "Chunked"],
                value=list_documents,
                interactive=False
            )
            
            # File Upload
            upload_input = gr.File(
                label="Upload Files",
                file_types=["file"],
                file_count="multiple"
            )
            status_output = gr.Textbox(label="Status")
            
            # Document Operations
            gr.Markdown("### Batch Operations")
            with gr.Row():
                doc_ids = gr.Textbox(
                    label="Document IDs",
                    placeholder="Enter comma-separated document IDs"
                )
                process_btn = gr.Button("Process Selected")
                metadata_btn = gr.Button("Edit Metadata")
                tag_btn = gr.Button("Manage Tags")
                
            # Tag Management
            gr.Markdown("### Tag Management")
            with gr.Row():
                tag_input = gr.Textbox(
                    label="Add New Tag",
                    placeholder="Enter tag name"
                )
                add_tag_btn = gr.Button("Add Tag")
                
            # Selected Tags Display
            selected_tags = gr.Textbox(
                label="Selected Tags",
                interactive=False
            )
            
            # Document Detail Button
            doc_detail_btn = gr.Button("View Document Details")
            
            # Notifications
            gr.Markdown("### Notifications")
            notifications = gr.Dataframe(
                headers=["Type", "Message", "Timestamp"],
                value=get_notifications,
                interactive=False
            )
            
            # Connect all buttons
            upload_btn.click(
                fn=upload_files,
                inputs=[upload_input],
                outputs=[status_output, doc_list]
            )
            process_btn.click(
                fn=process_selected_documents,
                inputs=[doc_ids],
                outputs=[status_output, doc_list]
            )
            metadata_btn.click(
                fn=edit_metadata,
                inputs=[doc_ids],
                outputs=[status_output, doc_list]
            )
            tag_btn.click(
                fn=manage_tags,
                inputs=[doc_ids],
                outputs=[selected_tags]
            )
            add_tag_btn.click(
                fn=add_tag,
                inputs=[tag_input, doc_ids],
                outputs=[selected_tags]
            )
            doc_detail_btn.click(
                fn=view_document_detail,
                inputs=[doc_ids],
                outputs=[status_output]
            )
            
            # Directory Import
            gr.Markdown("### Import from Directory")
            dir_path = gr.Textbox(
                label="Directory Path",
                placeholder="Enter directory path"
            )
            dir_import_btn = gr.Button("Import from Directory")
            dir_import_btn.click(
                fn=import_from_directory,
                inputs=[dir_path],
                outputs=[status_output, doc_list]
            )
            
            # SharePoint Import
            gr.Markdown("### Import from SharePoint")
            with gr.Row():
                sp_library = gr.Textbox(
                    label="Library Name",
                    placeholder="Enter SharePoint library name"
                )
                sp_folder = gr.Textbox(
                    label="Folder Path",
                    placeholder="Enter folder path"
                )
                sp_extensions = gr.Textbox(
                    label="File Extensions",
                    placeholder="Enter file extensions (e.g., .pdf,.docx)"
                )
            sp_import_btn = gr.Button("Import from SharePoint")
            sp_import_btn.click(
                fn=import_from_sharepoint,
                inputs=[sp_library, sp_folder, sp_extensions],
                outputs=[status_output, doc_list]
            )
        
        # Query Interface Column
        with gr.Column():
            gr.Markdown("### Document Query Interface")
            query_input = gr.Textbox(
                label="Enter your query",
                placeholder="Ask questions about the documents"
            )
            query_btn = gr.Button("Search")
            results = gr.Textbox(
                label="Search Results",
                lines=10
            )
            
            # Connect query interface
            query_btn.click(
                fn=document_manager.search_documents,
                inputs=[query_input],
                outputs=[results]
            )
        
        # System Stats
        with gr.Column(visible=False) as stats_box:
            gr.Markdown("### System Statistics")
            stats_content = gr.Markdown("""
                Total Documents: 0
                Total Versions: 0
            """)
            
            # Visualizations
            with gr.Row():
                version_plot = gr.Plot()
                class_plot = gr.Plot()
        
        # Search functionality
        search_btn.click(
            fn=search_documents,
            inputs=[search_input],
            outputs=[doc_list]
        )
        
        # Stats toggle
        stats_btn.click(
            fn=lambda: stats_box.update(visible=True),
            outputs=[stats_box]
        )
    
    return demo

def launch_ui(share=False):
    """Launch the document management UI.
    
    Args:
        share: Whether to create a public link
        
    Returns:
        Gradio interface
    """
    ui = create_ui()
    ui.launch(
        share=share,
        inbrowser=True,
        show_error=True  # Show error messages in the browser
    )
    return ui


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
    """Create the document management UI.
    
    Returns:
        Gradio interface
    """
    with gr.Blocks(title="Document Management System") as demo:
        with gr.Row():
            with gr.Column():
                gr.Markdown("# Document Management")
                
                # Document list
                with gr.Row():
                    gr.Markdown("## Document List")
                    doc_list = gr.DataFrame()
                    
                # File upload section with enhanced file explorer
                with gr.Row():
                    gr.Markdown("## Upload Documents")
                    with gr.Column():
                        # File explorer button
                        file_explorer = gr.Button("Open File Explorer")
                        # File upload with file explorer
                        upload_button = gr.File(
                            file_types=[".pdf", ".txt", ".docx", ".xlsx"],
                            file_count="multiple",
                            label="Select files to upload"
                        )
                        
                # Directory import
                with gr.Row():
                    gr.Markdown("## Import from Directory")
                    dir_path = gr.Textbox(
                        placeholder="Enter directory path",
                        label="Directory Path"
                    )
                    import_button = gr.Button("Import Directory")
                    
                # Status output
                status = gr.Textbox(label="Status")
                
            # Query interface
            with gr.Column():
                gr.Markdown("# Query Interface")
                query_input = gr.Textbox(
                    placeholder="Enter your query here...",
                    label="Query"
                )
                query_button = gr.Button("Ask Question")
                response = gr.Textbox(label="Response")
                
        # Event handlers
        # Add file explorer click handler
        file_explorer.click(
            fn=lambda: None,  # This will trigger the native file explorer
            inputs=[file_explorer],
            outputs=[upload_button]
        )
        
        upload_button.change(
            fn=upload_files,
            inputs=[upload_button],
            outputs=[status, doc_list]
        )
        
        import_button.click(
            fn=import_from_directory,
            inputs=[dir_path],
            outputs=[status, doc_list]
        )
        
        query_button.click(
            fn=query_documents,
            inputs=[query_input],
            outputs=[response]
        )
        
        # Initial document list
        doc_list.update(list_documents())
        
    demo.launch()


def launch_ui(share=False):
    """Launch the document management UI.
    
    Args:
        share: Whether to create a public link
        
    Returns:
        Gradio interface
    """
    ui = create_ui()
    ui.launch(
        share=share,
        inbrowser=True,
        show_error=True
    )
    return ui

if __name__ == "__main__":
    launch_ui()
