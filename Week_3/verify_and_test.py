import os
import sys
from reportlab.pdfgen import canvas
from srv.extraction.pdf_extractor import extract_text_from_pdf
from srv.processing.cleaner import clean_text

# Setup paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "data", "input")
TEST_PDF = os.path.join(INPUT_DIR, "sample_doc.pdf")

def create_sample_pdf(path):
    c = canvas.Canvas(path)
    c.drawString(100, 730, "Date: 2023-10-27")
    c.drawString(100, 710, "Total Amount: $500.00")
    c.drawString(100, 690, "   This is some   messy    whitespace   ")
    c.save()
    print(f"Created sample PDF at {path}")

def test_pipeline_logic():
    # 1. Create PDF
    if not os.path.exists(INPUT_DIR):
        os.makedirs(INPUT_DIR)
    create_sample_pdf(TEST_PDF)

    # 2. Extract
    print("Testing Extraction...")
    raw_text = extract_text_from_pdf(TEST_PDF)
    print(f"Raw Text:\n{raw_text}")

    # 3. Clean
    print("\nTesting Cleaning...")
    cleaned = clean_text(raw_text)
    print(f"Cleaned Text:\n'{cleaned}'")

    if "INVOICE # 12345" in cleaned and "whitespace" in cleaned:
        print("\nSUCCESS: Pipeline logic verified!")
    else:
        print("\nFAILURE: Text not extracted correctly.")
        
    # 4. Classify
    print("\nTesting Classification...")
    from srv.classification.classifier import DocumentClassifier
    clf = DocumentClassifier()
    cat, score = clf.classify_text(cleaned)
    print(f"Predicted Category: {cat} (Score: {score:.2f})")
    
    if cat.lower() == "invoice":
         print("SUCCESS: Classification verified!")
    else:
         print(f"WARNING: Expected 'invoice', got '{cat}'")

    # 5. Extract Fields
    print("\nTesting Field Extraction...")
    from srv.extraction.field_extractor import FieldExtractor
    extractor = FieldExtractor()
    data = extractor.extract_fields(cleaned, cat)
    
    print("Extracted Data:", data)
    
    # Check if we got something reasonable
    if data.get('total_amount') == "500.00":
        print("SUCCESS: Field Extraction verified!")
    else:
        print(f"FAILURE: Expected amount 500.00, got {data.get('total_amount')}")

if __name__ == "__main__":
    test_pipeline_logic()
