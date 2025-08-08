import os
import shutil

def copy_and_rename_files():
    # List of files to copy and rename
    files_to_process = [
        "src/document_processing/chunker.py",
        "src/tools/document_processor.py",
        "src/tools/metadata_validator.py",
        "src/tools/pdf_processor.py",
        "src/vector_db/faiss_db.py",
        "app.py",
        "src/rag_system.py",
        "test_classification.py",
        "test_clearance.py",
        "data/clearances.json",
        "data/documents/classified_strategic_plan.pdf",
        "data/documents/test_classification.txt",
        "setup.py",
        "src/tools/abstract_pdf_processor.py",
        "src/tools/pdf_processor_alt.py",
        "src/tools/pdf_processor_factory.py",
        "src/tools/pdf_processor_example.py",
        "src/tools/abstract_document_processor.py",
        "src/tools/smart_document_processor.py",
        "examples/document_processing_example.py",
        "README.md",
        "requirements.txt",
        "run_ui.py",
        "run_ui.ps1",
        "run_ui.sh",
        "run.sh"
    ]

    # Create destination directory if it doesn't exist
    os.makedirs("to-email", exist_ok=True)
    
    # Process each file
    for original_path in files_to_process:
        # Get the base filename
        base_filename = os.path.basename(original_path)
        
        # Create new filename with full path structure and .txt extension
        # Replace backslashes with forward slashes for consistency
        path_with_slashes = original_path.replace("\\", "/")
        # Remove leading slash if present
        if path_with_slashes.startswith("/"):
            path_with_slashes = path_with_slashes[1:]
        # Replace all slashes with underscores
        new_filename = path_with_slashes.replace("/", "_") + ".txt"
        
        # Construct source and destination paths
        src_path = original_path
        dst_path = os.path.join("to-email", new_filename)
        
        # Only copy if the source file exists
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
            print(f"Copied and renamed: {base_filename} -> {new_filename}")
        else:
            print(f"Warning: Source file not found: {src_path}")

if __name__ == "__main__":
    copy_and_rename_files()
