import time
import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from srv.extraction.pdf_extractor import extract_text_from_pdf
from srv.processing.cleaner import clean_text
# Import Classifier (lazy load might be bettefrom srv.processing.cleaner import clean_text
from srv.classification.classifier import DocumentClassifier
from srv.extraction.field_extractor import FieldExtractor
from srv.storage.csv_logger import CSVLogger

INPUT_DIR = os.path.join(os.getcwd(), "data", "input")

print("Initializing Classifier...")
doc_classifier = DocumentClassifier()

class PDFHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory:
            return
        if event.src_path.lower().endswith(".pdf"):
            print(f"New PDF detected: {event.src_path}")
            # Adding a small delay to ensure file write is complete
            time.sleep(1) 
            self.process_file(event.src_path)

    def process_file(self, filepath):
        try:
            filename = os.path.basename(filepath)
            print(f"\n[Processing] {filename}...")
            
            # 1. Extract Text
            text = extract_text_from_pdf(filepath)
            cleaned_text = clean_text(text)
            
            # 2. Classify
            category, score = doc_classifier.classify_text(cleaned_text)
            print(f"   -> Category: {category} (Conf: {score:.2f})")
            print(f"   -> Preview: {cleaned_text[:200]}...")
            
            # 3. Extract Key Fields
            metadata = extractor.extract_fields(cleaned_text, category)
            
            # 4. Log to CSV
            csv_logger.log_result(filename, category, score, metadata)
            print(f"   -> Data saved to CSV.")
            
        except Exception as e:
            print(f"Error processing {filepath}: {e}")

def start_watching():
    # 1. Initialize Components
    global doc_classifier, extractor, csv_logger
    print("Initializing Classifier...")
    doc_classifier = DocumentClassifier()
    extractor = FieldExtractor()
    csv_logger = CSVLogger()
    
    os.makedirs(INPUT_DIR, exist_ok=True)

    # 2. Bulk Process Existing Files
    print("\n--- Scanning for existing files ---")
    if os.path.exists(INPUT_DIR):
        for f in os.listdir(INPUT_DIR):
            if f.lower().endswith('.pdf'):
                path = os.path.join(INPUT_DIR, f)
                handler = PDFHandler()
                handler.process_file(path)
    print("--- Bulk processing complete. Watching for new files... ---\n")

    # 3. Start Watchdog
    event_handler = PDFHandler()
    observer = Observer()
    observer.schedule(event_handler, INPUT_DIR, recursive=False)
    observer.start()
    
    print(f"Monitoring {INPUT_DIR} for new PDFs...")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
