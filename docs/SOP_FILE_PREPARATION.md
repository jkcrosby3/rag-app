# Standard Operating Procedure: File Preparation for Email

## Purpose
This SOP outlines the process for preparing project files for email distribution, ensuring proper file naming and organization to prevent email system blocks.

## Scope
This procedure applies to all project files that need to be distributed via email.

## Procedure

### 1. File Preparation
1. Create a `to-email` directory in the project root if it doesn't exist:
   ```bash
   mkdir to-email
   ```

2. Copy relevant project files to the `to-email/txt_files` directory:
   - Source files should be copied from their original locations
   - Maintain the original file structure in the `txt_files` directory

### 2. File Renaming
1. Files must be renamed to include their full path structure:
   - Replace all directory separators (`\` or `/`) with underscores (`_`)
   - Add `.txt` extension to all files
   - Example: `src/tools/document_processor.py` becomes `src_tools_document_processor.py.txt`

2. The following file types should be renamed:
   - Python files (*.py)
   - JSON files (*.json)
   - PDF files (*.pdf)
   - Text files (*.txt)
   - Configuration files (*.json, *.yaml, *.toml)
   - Setup files (*.py, *.sh, *.ps1)
   - Documentation files (*.md, *.txt)

### 3. File Zipping
1. Create a zip archive of the renamed files:
   ```bash
   Compress-Archive -Path "to-email\*" -DestinationPath "to-email\updated_files_YYYY-MM-DD_renamed_with_paths.zip" -Force
   ```
   - Replace YYYY-MM-DD with the current date
   - Use the `-Force` flag to overwrite existing archives

### 4. Verification
1. Verify that all files have been:
   - Properly renamed with full path structure
   - Added `.txt` extension
   - Included in the zip archive
2. Test the zip file by:
   - Attempting to open it
   - Verifying file contents
   - Ensuring Gmail doesn't block the attachment

## File Naming Conventions
- All files must end with `.txt` extension
- Directory structure must be preserved in filenames using underscores
- Original file extensions must be preserved within the filename
- Example: `data_documents_classified_strategic_plan.pdf.txt`

## Exceptions
- Files that cannot be renamed (e.g., binary files) should be handled on a case-by-case basis
- Large files (>25MB) may need to be handled differently due to email size limits

## Maintenance
This SOP should be reviewed and updated whenever:
- New file types are added to the project
- Email system requirements change
- Project structure changes significantly

## References
- Project README.md
- Git repository history
- Email system documentation
