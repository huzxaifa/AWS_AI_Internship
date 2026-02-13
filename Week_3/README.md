# PDF Processing Pipeline

A local, serverless pipeline for **ingesting, classifying, and extracting data** from PDF documents.

## Features

*   **Real-Time Ingestion**: Monitors `data/input/` for new PDF files.
*   **Zero-Shot Classification**: Uses `joeddav/lm-roberta-large-xnli` to automatically categorize files into:
    *   Invoice, Resume, Technical Document, Email, Project Report.
*   **Key Field Extraction**: Uses Regex to pull metadata based on category:
    *   **Invoices**: Invoice #, Date, Total Amount ($).
    *   **General**: Emails, Phone Numbers.
*   **CSV Logging**: Automatically saves all results to `data/output/results.csv`.
*   **Bulk Processing**: Scans existing files on startup.

## Pipeline Flow

1.  **Ingestion (`watcher.py`)**: Detects file -> Triggers pipeline.
2.  **Extraction (`pdf_extractor.py`)**: Pulls raw text using `PyMuPDF`.
3.  **Cleaning (`cleaner.py`)**: Normalizes text (removes junk/whitespace).
4.  **Classification (`classifier.py`)**: AI Model determines document type.
5.  **Data Mining (`field_extractor.py`)**: Extracts specific fields (Date, Amount, etc.).
6.  **Storage (`csv_logger.py`)**: Appends structured data to CSV.

## Project Structure

```
PDF_Pipeline/
├── data/
│   ├── input/          # Drop PDFs here
│   └── output/         # Results saved here (results.csv)
├── srv/
│   ├── ingestion/      # Watchdog logic
│   ├── extraction/     # Text & Field extractors
│   ├── processing/     # Text cleaning
│   ├── classification/ # Zero-Shot Model
│   └── storage/        # CSV Logger
├── main.py             # Entry point
└── requirements.txt    # Dependencies
```

## Models Used and Tested

We experimented with several Zero-Shot models to find the right balance of speed and accuracy:

1.  **`facebook/bart-large-mnli`**: The original baseline. Good general accuracy but slower.
2.  **`cross-encoder/nli-deberta-v3-base`**: High accuracy (0.79+), but computationally heavy.
3.  **`typeform/distilbert-base-uncased-mnli`**: **Fastest & Most Accurate** for English (0.99 confidence).
4.  **`joeddav/xlm-roberta-large-xnli`** (Current): Selected for **Multilingual Support**.

## How to Run

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
2.  **Start the Pipeline**:
    ```bash
    python main.py
    ```
3.  **Use It**:
    *   Drop a PDF into `data/input`.
    *   Watch the terminal for real-time results.
    *   Check `data/output/results.csv` for the data.
