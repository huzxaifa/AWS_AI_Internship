import os
import sys
from srv.ingestion.watcher import start_watching

# Ensure we can import from srv
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    # Define directories
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    INPUT_DIR = os.path.join(BASE_DIR, "data", "input")
    
    # Ensure input directory exists
    if not os.path.exists(INPUT_DIR):
        print(f"Directory {INPUT_DIR} not found. Creating it...")
        os.makedirs(INPUT_DIR)

    print(f"Starting PDF Pipeline Watcher...")
    start_watching()
