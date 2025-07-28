# MuPDF and PyMuPDF Installation Guide for Closed Network

This guide provides detailed instructions for installing MuPDF and PyMuPDF in a closed network environment without internet access.

## Prerequisites

### Required Software
1. **C Compiler**
   - GCC 7.0 or later
   - Make utility

2. **Python Development Environment**
   - Python 3.13.0 or later
   - Python development headers
   - pip (Python package installer)

3. **System Dependencies**
   - Standard C library headers
   - FreeType development headers
   - zlib development headers

## Step-by-Step Installation

### 1. Prepare the Build Environment

```bash
# Create working directory
mkdir -p /opt/mupdf_build

cd /opt/mupdf_build

# Create directories for each component
mkdir -p mupdf_source pymupdf_source dependencies
```

### 2. Install System Dependencies

```bash
# Install build tools
apt-get install -y build-essential

# Install Python development packages
apt-get install -y python3-dev python3-pip

# Install required libraries
apt-get install -y libfreetype6-dev zlib1g-dev
```

### 3. Build MuPDF

```bash
# Copy MuPDF source to build directory
cp -r /path/to/mupdf_source ./mupdf_source

cd mupdf_source

# Configure build
make HAVE_X11=no HAVE_GLFW=no HAVE_FREETYPE=yes

# Build MuPDF
make

# Install MuPDF
make install
```

### 4. Verify MuPDF Installation

```bash
# Check if MuPDF library is installed
ldconfig -p | grep mupdf

# If not found, update library cache
ldconfig

# Verify MuPDF version
mutool -v
```

### 5. Install PyMuPDF

```bash
# Copy PyMuPDF source
cp -r /path/to/pymupdf_source ./pymupdf_source

cd pymupdf_source

# Install PyMuPDF with local MuPDF
python3 setup.py install --mupdf-include=/opt/mupdf_build/mupdf_source/include \
                         --mupdf-lib=/opt/mupdf_build/mupdf_source/build/release
```

### 6. Verify PyMuPDF Installation

```bash
# Test PyMuPDF installation
python3 -c "import fitz; print('PyMuPDF installed successfully')"

# Check PyMuPDF version
python3 -c "import fitz; print('PyMuPDF version:', fitz.__version__)"
```

## Troubleshooting

### Common Issues and Solutions

1. **"fitz.h" not found**
   - Solution: Set `MUPDF_INCLUDE` environment variable
   ```bash
   export MUPDF_INCLUDE=/opt/mupdf_build/mupdf_source/include
   ```

2. **MuPDF library not found**
   - Solution: Update `LD_LIBRARY_PATH`
   ```bash
   export LD_LIBRARY_PATH=/opt/mupdf_build/mupdf_source/build/release:$LD_LIBRARY_PATH
   ```

3. **Missing dependencies**
   - Solution: Install missing packages
   ```bash
   apt-get install -y libfreetype6-dev zlib1g-dev
   ```

4. **Python version mismatch**
   - Solution: Ensure Python 3.13.0 or later is installed
   ```bash
   python3 --version
   ```

## Environment Variables

```bash
# Set environment variables
export MUPDF_INCLUDE=/opt/mupdf_build/mupdf_source/include
export MUPDF_LIB=/opt/mupdf_build/mupdf_source/build/release
export LD_LIBRARY_PATH=/opt/mupdf_build/mupdf_source/build/release:$LD_LIBRARY_PATH
```

## Post-Installation Verification

```bash
# Verify MuPDF
mutool version

# Verify PyMuPDF
python3 -c "import fitz; print('PyMuPDF installed successfully')"

# Test PDF processing
python3 -c "
import fitz

doc = fitz.open('/path/to/test.pdf')
print(f'Pages: {len(doc)}')
print('First page text:')
print(doc[0].get_text())
"
```

## Additional Configuration

### 1. System-wide Configuration

```bash
# Add MuPDF to system library path
sudo sh -c 'echo "/opt/mupdf_build/mupdf_source/build/release" > /etc/ld.so.conf.d/mupdf.conf'

# Update system library cache
sudo ldconfig
```

### 2. Python Virtual Environment

```bash
# Create virtual environment
python3 -m venv /opt/rag_env

# Activate virtual environment
source /opt/rag_env/bin/activate

# Install PyMuPDF in virtual environment
pip install .
```

## Security Considerations

1. **File Permissions**
   - Set appropriate permissions on installation directories
   ```bash
   chown -R user:group /opt/mupdf_build
   chmod -R 755 /opt/mupdf_build
   ```

2. **Environment Isolation**
   - Use virtual environments for Python packages
   - Run MuPDF processes with minimal privileges

3. **Regular Updates**
   - Keep MuPDF and PyMuPDF versions up to date
   - Monitor for security patches

## Maintenance

### Backup Configuration

```bash
# Backup configuration
mkdir -p /opt/mupdf_backup

cp -r /opt/mupdf_build /opt/mupdf_backup/

cp -r /opt/rag_env /opt/mupdf_backup/
```

### Version Management

```bash
# Check installed versions
mutool -v
python3 -c "import fitz; print('PyMuPDF:', fitz.__version__)"
```

## Error Reference

### Common Error Messages

1. "fitz.h not found"
   - Missing MuPDF include files
   - Solution: Set MUPDF_INCLUDE

2. "libmupdf.so not found"
   - Missing MuPDF library
   - Solution: Update LD_LIBRARY_PATH

3. "Python version mismatch"
   - Incorrect Python version
   - Solution: Install correct Python version

4. "Permission denied"
   - Insufficient file permissions
   - Solution: Set correct permissions
