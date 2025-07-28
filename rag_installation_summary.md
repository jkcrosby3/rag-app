# RAG Application Installation Summary

## 1. Core Components

### Main Application Files
- `app.py`: Main entry point
- `templates/index.html`: Web interface
- `src/rag_system.py`: Core RAG system
- `src/vector_db/faiss_db.py`: Vector database
- `src/retrieval/retriever.py`: Document retrieval
- `src/llm/claude_client.py`: LLM integration

## 2. Installation Requirements

### System Requirements
- Python 3.13.0 or later
- GCC 7.0 or later
- Make utility
- System libraries:
  - libfreetype6-dev
  - zlib1g-dev
  - build-essential

### Python Packages
- PyMuPDF (1.25.0+)
- PyPDF2 (3.0.0+)
- FastAPI (0.100.0+)
- Uvicorn (0.22.0+)
- Typer (0.9.0+)
- Python-Dotenv (1.0.0+)
- Pydantic (2.0.0+)

## 3. Installation Order

1. **System Dependencies**
   ```bash
   apt-get install -y build-essential python3-dev python3-pip libfreetype6-dev zlib1g-dev
   ```

2. **MuPDF**
   ```bash
   cd /opt/mupdf_build/mupdf_source
   make HAVE_X11=no HAVE_GLFW=no HAVE_FREETYPE=yes
   make
   make install
   ```

3. **PyMuPDF**
   ```bash
   cd /opt/mupdf_build/pymupdf_source
   python3 setup.py install --mupdf-include=/opt/mupdf_build/mupdf_source/include --mupdf-lib=/opt/mupdf_build/mupdf_source/build/release
   ```

4. **Python Packages**
   ```bash
   pip install -r requirements.txt
   ```

## 4. Environment Variables

```bash
export MUPDF_INCLUDE=/opt/mupdf_build/mupdf_source/include
export MUPDF_LIB=/opt/mupdf_build/mupdf_source/build/release
export LD_LIBRARY_PATH=/opt/mupdf_build/mupdf_source/build/release:$LD_LIBRARY_PATH
```

## 5. Verification Steps

```bash
# Verify MuPDF
mutool -v

# Verify PyMuPDF
python3 -c "import fitz; print('PyMuPDF installed successfully')"

# Verify Python packages
pip list
```

## 6. Troubleshooting

### Common Issues
1. **Missing Dependencies**
   - Install missing packages
   - Verify versions

2. **Permission Errors**
   - Check file permissions
   - Run as root if needed

3. **Build Failures**
   - Check logs
   - Verify environment variables
   - Clean and rebuild

4. **Python Version**
   - Verify Python version
   - Use correct virtual environment

## 7. Maintenance

### Backup
```bash
# Create backup
mkdir -p /opt/mupdf_backup

cp -r /opt/mupdf_build /opt/mupdf_backup/
```

### Version Management
```bash
# Check versions
mutool -v
python3 -c "import fitz; print('PyMuPDF:', fitz.__version__)"
```

## 8. Notes

1. All operations should be performed in a closed network environment
2. Use the automated installation script for consistency
3. Verify each step before proceeding
4. Keep detailed logs of all operations
5. Maintain regular backups
