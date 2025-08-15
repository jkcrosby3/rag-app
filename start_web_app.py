"""
Startup script for the RAG web application.

This script ensures all dependencies are installed and starts the web application.
"""
import os
import sys
import subprocess
import importlib.util
import time
import argparse
from pathlib import Path

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Start the RAG web application")
    parser.add_argument(
        "--ui", 
        type=str, 
        choices=["streamlit", "flask", "gradio", "unified"],
        default="unified",
        help="Which UI to start (default: unified)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8501,
        help="Port to run the application on (default: 8501)"
    )
    return parser.parse_args()

def check_and_install_package(package_name, install_name=None):    
    """Check if a package is installed, and install it if not."""
    try:
        importlib.import_module(package_name.split(".")[0] if "." in package_name else package_name)
        print(f"Found {package_name}")
        return True
    except ImportError:
        print(f"Installing {package_name}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", install_name or package_name])
            # Using [OK] instead of checkmark for Windows compatibility
            print(f"[OK] Installed {package_name}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Failed to install {package_name}: {e}")
            return False

def check_and_install_dependencies():
    """Check and install all required dependencies."""
    print("Checking dependencies...")
    
    # Core dependencies
    core_deps = [
        "python-dotenv",
        "numpy",
        "pandas",
        "tqdm",
        "requests",
        "PyMuPDF",
        "PyPDF2",
        "python-multipart"
    ]
    
    # UI-specific dependencies
    ui_deps = {
        "streamlit": ["streamlit>=1.24.0"],
        "flask": ["flask>=2.0.0", "flask-cors"],
        "gradio": ["gradio>=3.0.0"],
        "unified": ["streamlit>=1.24.0"]
    }
    
    # ML dependencies
    ml_deps = [
        "sentence-transformers>=2.2.2",
        "faiss-cpu",  # or faiss-gpu if CUDA is available
        "scikit-learn>=1.3.0",
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "anthropic>=0.9.0",
        "openai>=1.0.0"
    ]
    
    # Database and caching
    db_deps = [
        "SQLAlchemy>=2.0.0",
        "elasticsearch>=8.0.0",
        "diskcache>=5.6.1"
    ]
    
    all_deps = core_deps + ml_deps + db_deps
    
    # Add UI-specific deps based on args
    args = parse_arguments()
    all_deps.extend(ui_deps.get(args.ui, []))
    
    # Install all dependencies
    success = True
    for dep in all_deps:
        if not check_and_install_package(dep.split("==")[0].replace(">", "").replace("<", ""), dep):
            print(f"Warning: Failed to install {dep}")
            success = False
    
    return success

def check_vector_database():
    """Check if the vector database exists and is properly set up."""
    vector_db_path = Path("data/vector_db")
    if not vector_db_path.exists() or not any(vector_db_path.iterdir()):
        print("Vector database not found. Building it now...")
        try:
            subprocess.check_call([sys.executable, "scripts/build_vector_db.py"])
            print("✓ Vector database built successfully")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"Warning: Could not build vector database: {e}")
            print("You can manually build it later by running:")
            print("  python scripts/build_vector_db.py")
            return False
    return True

def start_web_app(ui_type, port=8501):
    """Start the web application with the specified UI."""
    print(f"\nStarting {ui_type.capitalize()} web application on port {port}...")
    
    if ui_type == "streamlit":
        return start_streamlit_app("app.py", port)
    elif ui_type == "flask":
        return start_flask_app(port)
    elif ui_type == "gradio":
        return start_gradio_app(port)
    elif ui_type == "unified":
        return start_streamlit_app("unified_app.py", port)
    else:
        print(f"Unknown UI type: {ui_type}")
        return False

def start_streamlit_app(app_file, port):
    """Start a Streamlit application."""
    os.environ["STREAMLIT_SERVER_PORT"] = str(port)
    os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
    os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    
    try:
        import streamlit.web.cli as st_cli
        from streamlit.web.cli import main as st_main
        
        sys.argv = [
            "streamlit", "run", 
            app_file,
            f"--server.port={port}",
            "--server.headless=true",
            "--browser.gatherUsageStats=false"
        ]
        
        print(f"\nRAG Web Application is running!")
        print(f"Open your browser and navigate to: http://localhost:{port}")
        print("Press Ctrl+C to stop the server\n")
        
        st_main()
        return True
        
    except KeyboardInterrupt:
        print("\nShutting down the web application...")
        return True
    except Exception as e:
        print(f"Error starting Streamlit application: {e}")
        return False

def start_flask_app(port):
    """Start the Flask application."""
    try:
        from flask import Flask
        app = Flask(__name__)
        
        @app.route('/')
        def home():
            return "Flask app is running!"
        
        print(f"\nFlask application is running!")
        print(f"Open your browser and navigate to: http://localhost:{port}")
        print("Press Ctrl+C to stop the server\n")
        
        app.run(port=port)
        return True
    except Exception as e:
        print(f"Error starting Flask application: {e}")
        return False

def start_gradio_app(port):
    """Start the Gradio application."""
    try:
        import gradio as gr
        
        def greet(name):
            return f"Hello {name}!"
        
        iface = gr.Interface(
            fn=greet, 
            inputs="text", 
            outputs="text",
            title="RAG System - Document Management"
        )
        
        print(f"\nGradio application is running!")
        print(f"Open your browser and navigate to: http://localhost:{port}")
        print("Press Ctrl+C to stop the server\n")
        
        iface.launch(server_port=port)
        return True
    except Exception as e:
        print(f"Error starting Gradio application: {e}")
        return False

def main():
    """Main entry point for the startup script."""
    print("=" * 50)
    print("RAG Web Application Startup")
    print("=" * 50)
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Check and install dependencies
    if not check_and_install_dependencies():
        print("\nWarning: Some dependencies failed to install. The application may not work correctly.")
    
    # Check vector database
    if not check_vector_database():
        print("\nWarning: Vector database not found. Some features may not work.")
    
    # Start the web application
    if not start_web_app(args.ui, args.port):
        print("\nFailed to start the web application.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
