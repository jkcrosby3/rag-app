import os
import sys

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

# Now run the document UI
if __name__ == "__main__":
    from src.web.document_ui import launch_ui
    launch_ui()
