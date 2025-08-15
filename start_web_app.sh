#!/bin/bash
# Shell script to start the RAG web application

echo "Starting RAG Web Application..."
echo "===================================="

# Show available UIs
echo "Available UIs:"
echo "1. Unified Interface (default) - Combined query and document management"
echo "2. Query Interface - Just the query interface"
echo "3. Document Management - Just the document management UI"
echo "4. Flask Web App - Traditional web interface"
echo ""

# Get user input
read -p "Choose UI [1-4] (default: 1): " choice

# Set UI based on choice
case "$choice" in
    1|"") UI="unified" ;;
    2) UI="streamlit" ;;
    3) UI="gradio" ;;
    4) UI="flask" ;;
    *) echo "Invalid choice. Using default UI."; UI="unified" ;;
esac

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
else
    echo "Creating Python virtual environment..."
    python -m venv venv
    source venv/bin/activate
    echo "Virtual environment created and activated."
fi

# Install the package in development mode
echo "Installing package in development mode..."
pip install -e .

# Run the startup script with the selected UI
echo "Starting the $UI web application..."
python start_web_app.py --ui "$UI"
