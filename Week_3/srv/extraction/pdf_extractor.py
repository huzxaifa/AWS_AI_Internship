import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_path):
    full_text = []
    try:
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text = page.get_text()
                if text:
                    full_text.append(text)
        return "\n".join(full_text)
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        return ""
