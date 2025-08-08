#!/usr/bin/env python3
"""
This script sets up and runs the RAG application UI.

Usage:
- On Windows: python run.sh
- On Linux: chmod +x run.sh && ./run.sh

The script will:
1. Install all required Python packages from requirements.txt
2. Start the document management UI interface
"""

import os
import sys
import subprocess

# Function to install required packages
def install_requirements():
    """
    Install all Python packages needed for the application.
    
    This function reads the requirements.txt file and installs all packages using pip.
    Returns:
        bool: True if installation was successful, False otherwise
    """
    print("Installing requirements...")
    try:
        # Use sys.executable to ensure we're using the correct Python interpreter
        # -m pip tells Python to use pip as a module
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print(f"Error installing requirements: {result.stderr}")
            return False
            
        print("Requirements installed successfully")
        return True
        
    except Exception as e:
        print(f"Error during requirements installation: {e}")
        return False

def run_ui():
    """
    Launch the document management UI.
    
    This function starts the Gradio interface for managing documents.
    Returns:
        bool: True if UI launched successfully, False otherwise
    """
    try:
        from src.web.document_ui import launch_ui
        
        # Start the UI with in-browser=True
        launch_ui(share=False)
        return True
        
    except Exception as e:
        print(f"Error launching UI: {e}")
        print("You can manually open the browser at http://127.0.0.1:7860")
        return False

def main():
    """
    Main function that runs the setup and UI.
    
    This function installs requirements and launches the UI immediately.
    Document registration runs in the background.
    Returns:
        int: 0 if successful, 1 if any error occurred
    """
    print("Starting RAG Application Setup...")
    
    # Install requirements
    if not install_requirements():
        print("Failed to install requirements. Exiting...")
        return 1
        
    # Run UI
    if not run_ui():
        print("Failed to launch UI. Exiting...")
        return 1
        
    return 0

if __name__ == "__main__":
    """
    Entry point of the script.
    
    When you run this script directly (python run.sh), this block executes.
    """
    try:
        exit_code = main()
        sys.exit(exit_code if exit_code is not None else 0)
    except KeyboardInterrupt:
        print("\nScript interrupted by user. Cleaning up...")
        sys.exit(1)
    except Exception as e:
        print(f"\nFatal error: {e}")
        sys.exit(1)
