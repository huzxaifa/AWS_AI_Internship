import csv
import os
from datetime import datetime

class CSVLogger:
    def __init__(self, output_file="data/output/results.csv"):
        self.output_file = output_file
        self._ensure_file_exists()

    def _ensure_file_exists(self):
        # Create directory if needed
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        
        # Create file with header if it doesn't exist
        if not os.path.exists(self.output_file):
            with open(self.output_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                # Define standard columns
                header = [
                    "Timestamp", "Filename", "Category", "Confidence", 
                    "Invoice Number", "Date", "Total Amount", "Emails", "Phones"
                ]
                writer.writerow(header)

    def log_result(self, filename, category, confidence, data):
        """
        Appends a new record to the CSV.
        """
        # Flatten the list fields (emails/phones) for CSV
        emails = "; ".join(data.get('emails', []))
        phones = "; ".join(data.get('phones', []))
        
        row = [
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            filename,
            category,
            f"{confidence:.2f}",
            data.get('invoice_number', ''),
            data.get('date', ''),
            data.get('total_amount', ''),
            emails,
            phones
        ]
        
        with open(self.output_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(row)
        
        print(f"Logged to CSV: {filename}")
