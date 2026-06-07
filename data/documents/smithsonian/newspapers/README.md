# Smithsonian Revolutionary War Newspapers Dataset

## Overview
This directory contains 65 Revolutionary War newspaper text files from the Smithsonian collection (1770-1810).

## How to Download the Dataset

### Option 1: Direct Download from Smithsonian
The newspaper files are available from the Smithsonian's digital collections. Use the provided download scripts:

```bash
python download_newspapers_simple.py
```

This will download all 65 newspaper text files to `text_files/` directory.

### Option 2: Manual Download
Visit the Smithsonian Open Access portal and search for Revolutionary War newspapers from 1770-1810.

## Processing the Data

After downloading the newspaper files, process them into the vector database:

```bash
python process_newspapers.py
```

This will:
1. Chunk the 65 newspaper text files
2. Generate embeddings for each chunk
3. Build the FAISS vector database
4. Store results in `data/vector_db/`

## Expected Files
- `text_files/*.txt` - 65 newspaper text files (not in git due to size)
- `newspapers_data.json` - Metadata (not in git due to size - 1.2GB)
- `newspapers_data.parquet` - Processed data (not in git due to size - 797MB)

## Note
The actual newspaper text files are NOT included in the git repository due to their size (>100MB total). You must download them using the scripts provided before running the RAG system.
