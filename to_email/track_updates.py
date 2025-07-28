import os
import time
from datetime import datetime, timedelta
import hashlib

class FileTracker:
    def __init__(self):
        self.tracking_file = f"updated_files_{datetime.now().strftime('%Y-%m-%d')}.txt"
        self.updated_files = set()
        self.load_existing_files()
        
    def load_existing_files(self):
        """Load existing tracked files if the tracking file exists"""
        if os.path.exists(self.tracking_file):
            with open(self.tracking_file, 'r') as f:
                lines = f.readlines()
                self.updated_files = set(line.strip() for line in lines[1:] if line.strip())
    
    def track_file(self, file_path):
        """Track a new file update"""
        if file_path not in self.updated_files:
            self.updated_files.add(file_path)
            with open(self.tracking_file, 'a') as f:
                f.write(f"\n{file_path}")
    
    def archive_tracking_file(self):
        """Archive the tracking file at the end of the day"""
        archive_dir = "archive"
        if not os.path.exists(archive_dir):
            os.makedirs(archive_dir)
            
        archive_path = os.path.join(archive_dir, os.path.basename(self.tracking_file))
        if os.path.exists(self.tracking_file):
            os.rename(self.tracking_file, archive_path)

def main():
    tracker = FileTracker()
    
    # Check for file updates every minute
    while True:
        current_time = datetime.now()
        
        # Archive and exit at midnight
        if current_time.hour == 0 and current_time.minute == 0:
            tracker.archive_tracking_file()
            break
            
        # Get list of all files in the project directory
        for root, _, files in os.walk('.'): 
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    # Get file modification time
                    mod_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                    
                    # If file was modified today and not already tracked
                    if mod_time.date() == current_time.date():
                        tracker.track_file(os.path.relpath(file_path))
                        
                except Exception as e:
                    print(f"Error processing {file_path}: {str(e)}")
                    continue
                    
        # Wait for 1 minute before next check
        time.sleep(60)

if __name__ == "__main__":
    main()
