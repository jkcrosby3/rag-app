import os
import time
from datetime import datetime, timedelta
import hashlib
import logging
import sys
from pathlib import Path

class FileTracker:
    def __init__(self, last_run_date=None):
        """
        Initialize the file tracker with an optional last run date/time
        """
        self.project_root = Path(__file__).parent
        self.last_run_file = self.project_root / "last_run_time.txt"
        self.last_run = self.get_last_run_time(last_run_date)
        self.tracking_file = self.project_root / f"updated_files_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}.txt"
        self.updated_files = set()
        self.setup_logging()
        self.load_existing_files()
        
    def setup_logging(self):
        """Set up logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.project_root / 'tracking.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
    def get_last_run_time(self, default_date=None):
        """Get the last run time from file or use default"""
        try:
            if self.last_run_file.exists():
                with open(self.last_run_file, 'r') as f:
                    last_run_str = f.read().strip()
                    return datetime.strptime(last_run_str, "%Y-%m-%d %H:%M:%S")
            if default_date:
                return datetime.strptime(default_date, "%Y-%m-%d %H:%M:%S")
            return datetime.now() - timedelta(days=7)  # Default to 1 week ago
        except Exception as e:
            logging.error(f"Error reading last run time: {e}")
            if default_date:
                return datetime.strptime(default_date, "%Y-%m-%d %H:%M:%S")
            return datetime.now() - timedelta(days=7)
    
    def update_last_run_time(self):
        """Update the last run time file with current time"""
        try:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(self.last_run_file, 'w') as f:
                f.write(current_time)
            logging.info(f"Updated last run time to {current_time}")
        except Exception as e:
            logging.error(f"Error updating last run time: {e}")
    
    def load_existing_files(self):
        """Load existing tracked files if the tracking file exists"""
        try:
            if self.tracking_file.exists():
                with open(self.tracking_file, 'r') as f:
                    lines = f.readlines()
                    self.updated_files = set(line.strip() for line in lines[1:] if line.strip())
        except Exception as e:
            logging.error(f"Error loading existing files: {e}")
    
    def track_file(self, file_path):
        """Track a new file update"""
        try:
            abs_path = Path(file_path).resolve()
            if abs_path not in self.updated_files:
                self.updated_files.add(abs_path)
                with open(self.tracking_file, 'a') as f:
                    f.write(f"\n{abs_path}")
                logging.info(f"Tracked new file update: {abs_path}")
        except Exception as e:
            logging.error(f"Error tracking file {file_path}: {e}")
    
    def get_files_since_last_run(self):
        """Get all files modified since the last run"""
        modified_files = []
        try:
            for root, _, files in os.walk(self.project_root):
                for file in files:
                    file_path = Path(root) / file
                    try:
                        mod_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                        if mod_time > self.last_run:
                            modified_files.append(str(file_path.resolve()))
                    except Exception as e:
                        logging.error(f"Error checking modification time for {file_path}: {e}")
            return modified_files
        except Exception as e:
            logging.error(f"Error in get_files_since_last_run: {e}")
            return []

def main():
    try:
        # Parse command line arguments
        import argparse
        parser = argparse.ArgumentParser(description='Track file changes in the project')
        parser.add_argument('--last-run', default="2025-08-07 06:00:00",
                          help='Last run date/time in format YYYY-MM-DD HH:MM:SS')
        args = parser.parse_args()
        
        tracker = FileTracker(last_run_date=args.last_run)
        logging.info(f"File tracking started (last run: {args.last_run})")
        logging.info(f"Tracking file: {tracker.tracking_file}")
        
        # First check for files modified since last run
        modified_files = tracker.get_files_since_last_run()
        if modified_files:
            logging.info(f"Found {len(modified_files)} files modified since last run")
            for file in modified_files:
                tracker.track_file(file)
        else:
            logging.info("No files modified since last run")
        
        # Check for file updates every minute
        while True:
            current_time = datetime.now()
            
            try:
                # Get list of all files in the project directory
                for root, _, files in os.walk(tracker.project_root):
                    for file in files:
                        file_path = os.path.join(root, file)
                        try:
                            # Get file modification time
                            mod_time = os.path.getmtime(file_path)
                            # Track if modified in last minute
                            if time.time() - mod_time < 60:
                                tracker.track_file(file_path)
                        except Exception as e:
                            logging.error(f"Error processing file {file_path}: {e}")
                
                # Update last run time after each check
                tracker.update_last_run_time()
                
                # Sleep for 59 seconds to ensure we don't miss any updates
                time.sleep(59)
            except Exception as e:
                logging.error(f"Error in file processing loop: {e}")
                time.sleep(59)  # Still wait to avoid tight loop on error
    except KeyboardInterrupt:
        logging.info("Tracking interrupted by user")
        if 'tracker' in locals():
            tracker.update_last_run_time()
    except Exception as e:
        logging.error(f"Fatal error in main: {e}")
        return 1  # Return non-zero exit code on error
    finally:
        # Ensure last run time is updated before exit
        if 'tracker' in locals():
            tracker.update_last_run_time()

if __name__ == "__main__":
    sys.exit(main())
