import os
import shutil

def rename_files_in_to_email():
    # Define source and destination directories
    src_dir = "to_email/txt_files"
    dst_dir = "to_email"
    
    # Create destination directory if it doesn't exist
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    
    # Get list of files we want to rename
    files_to_rename = [
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
    
    # Process each file
    for original_path in files_to_rename:
        # Create new filename with path structure and .txt extension
        new_filename = original_path.replace("\\", "_").replace("/", "_") + ".txt"
        
        # Copy file with new name to destination directory
        src_path = os.path.join(src_dir, os.path.basename(original_path))
        dst_path = os.path.join(dst_dir, new_filename)
        
        # Only copy if the source file exists
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
            print(f"Renamed {os.path.basename(original_path)} to {new_filename}")
        else:
            print(f"Warning: Source file not found: {src_path}")

if __name__ == "__main__":
    rename_files_in_to_email()
