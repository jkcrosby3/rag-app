#!/bin/bash

# Set script to exit on error
set -e

# Log file for troubleshooting
LOG_FILE="/var/log/mupdf_install.log"

# Function to log messages with timestamp
log_message() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

# Function to check command success
check_command() {
    local cmd="$1"
    local desc="$2"
    
    log_message "Checking $desc..."
    if ! command -v "$cmd" >/dev/null 2>&1; then
        log_message "ERROR: $cmd not found. Please install $desc."
        exit 1
    fi
    log_message "$desc found."
}

# Function to verify file exists
check_file() {
    local file="$1"
    local desc="$2"
    
    log_message "Checking $desc..."
    if [ ! -f "$file" ]; then
        log_message "ERROR: $desc not found at $file"
        exit 1
    fi
    log_message "$desc found at $file"
}

# Function to verify directory exists
check_dir() {
    local dir="$1"
    local desc="$2"
    
    log_message "Checking $desc..."
    if [ ! -d "$dir" ]; then
        log_message "ERROR: $desc not found at $dir"
        exit 1
    fi
    log_message "$desc found at $dir"
}

# Function to verify permissions
check_permissions() {
    local file="$1"
    local desc="$2"
    
    log_message "Checking permissions for $desc..."
    if [ ! -w "$file" ]; then
        log_message "WARNING: Write permissions missing for $desc"
    fi
    if [ ! -r "$file" ]; then
        log_message "WARNING: Read permissions missing for $desc"
    fi
}

# Function to verify Python version
check_python_version() {
    local min_version="3.13.0"
    local python_version=$(python3 --version 2>&1 | cut -d' ' -f2)
    
    log_message "Checking Python version..."
    if ! python3 -c "import sys; assert sys.version >= '$min_version'" 2>/dev/null; then
        log_message "ERROR: Python version $python_version is too old. Minimum required: $min_version"
        exit 1
    fi
    log_message "Python version $python_version is sufficient"
}

# Main installation script

# 1. Initial Checks
log_message "Starting MuPDF/PyMuPDF installation"

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    log_message "ERROR: Please run as root"
    exit 1
fi

# Check required commands
check_command "gcc" "C compiler"
check_command "make" "Make utility"
check_command "python3" "Python 3"
check_command "pip3" "Python package installer"

# Check Python version
check_python_version

# 2. Create working directories
WORK_DIR="/opt/mupdf_build"

log_message "Creating working directories..."
if [ ! -d "$WORK_DIR" ]; then
    mkdir -p "$WORK_DIR"
    if [ $? -ne 0 ]; then
        log_message "ERROR: Failed to create working directory"
        exit 1
    fi
fi

# 3. Verify source files
log_message "Verifying source files..."

# Check for MuPDF source
MUPDF_SOURCE_DIR="$WORK_DIR/mupdf_source"
check_dir "$MUPDF_SOURCE_DIR" "MuPDF source directory"

# Check for PyMuPDF source
PYMUPDF_SOURCE_DIR="$WORK_DIR/pymupdf_source"
check_dir "$PYMUPDF_SOURCE_DIR" "PyMuPDF source directory"

# 4. Build MuPDF
log_message "Starting MuPDF build..."

# Change to MuPDF directory
cd "$MUPDF_SOURCE_DIR" || {
    log_message "ERROR: Failed to change to MuPDF source directory"
    exit 1
}

# Configure build
log_message "Configuring MuPDF build..."
if ! make HAVE_X11=no HAVE_GLFW=no HAVE_FREETYPE=yes; then
    log_message "ERROR: Failed to configure MuPDF build"
    exit 1
fi

# Build MuPDF
log_message "Building MuPDF..."
if ! make; then
    log_message "ERROR: Failed to build MuPDF"
    exit 1
fi

# Install MuPDF
log_message "Installing MuPDF..."
if ! make install; then
    log_message "ERROR: Failed to install MuPDF"
    exit 1
fi

# 5. Verify MuPDF installation
log_message "Verifying MuPDF installation..."

if ! ldconfig -p | grep mupdf >/dev/null 2>&1; then
    log_message "WARNING: MuPDF library not found in ldconfig"
    log_message "Updating ldconfig cache..."
    ldconfig
fi

# 6. Install PyMuPDF
log_message "Starting PyMuPDF installation..."

cd "$PYMUPDF_SOURCE_DIR" || {
    log_message "ERROR: Failed to change to PyMuPDF source directory"
    exit 1
}

# Set environment variables
export MUPDF_INCLUDE="$MUPDF_SOURCE_DIR/include"
export MUPDF_LIB="$MUPDF_SOURCE_DIR/build/release"
export LD_LIBRARY_PATH="$MUPDF_LIB:$LD_LIBRARY_PATH"

# Install PyMuPDF
log_message "Installing PyMuPDF..."
if ! python3 setup.py install --mupdf-include="$MUPDF_INCLUDE" --mupdf-lib="$MUPDF_LIB"; then
    log_message "ERROR: Failed to install PyMuPDF"
    log_message "DEBUG: MUPDF_INCLUDE=$MUPDF_INCLUDE"
    log_message "DEBUG: MUPDF_LIB=$MUPDF_LIB"
    log_message "DEBUG: LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
    exit 1
fi

# 7. Verification
log_message "Verifying installation..."

# Verify PyMuPDF
if ! python3 -c "import fitz; print('PyMuPDF installed successfully')" >/dev/null 2>&1; then
    log_message "ERROR: PyMuPDF installation verification failed"
    exit 1
fi

# Verify MuPDF version
MU_VERSION=$(mutool -v 2>/dev/null || echo "not found")
PY_VERSION=$(python3 -c "import fitz; print(fitz.__version__)" 2>/dev/null || echo "not found")

log_message "Installation complete!"
log_message "MuPDF version: $MU_VERSION"
log_message "PyMuPDF version: $PY_VERSION"

# 8. Post-installation cleanup
log_message "Cleaning up..."

# Create backup
BACKUP_DIR="$WORK_DIR/backup_$(date '+%Y%m%d_%H%M%S')"
log_message "Creating backup at $BACKUP_DIR"
mkdir -p "$BACKUP_DIR"
cp -r "$MUPDF_SOURCE_DIR" "$BACKUP_DIR/mupdf_source"
cp -r "$PYMUPDF_SOURCE_DIR" "$BACKUP_DIR/pymupdf_source"

log_message "Installation completed successfully!"
log_message "Check $LOG_FILE for detailed logs"

# Exit successfully
exit 0
