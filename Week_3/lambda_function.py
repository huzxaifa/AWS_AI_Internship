import json
import boto3
import os
import uuid
from srv.extraction.pdf_extractor import extract_text_from_pdf
from srv.processing.cleaner import clean_text
from srv.classification.classifier import DocumentClassifier
from srv.extraction.field_extractor import FieldExtractor

# Initialize models outside handler (Warm Start)
print("Initializing models...")
doc_classifier = DocumentClassifier()
extractor = FieldExtractor()
s3 = boto3.client('s3')

def lambda_handler(event, context):
    try:
        # 1. Get Bucket & Key from S3 Event
        for record in event['Records']:
            bucket_name = record['s3']['bucket']['name']
            file_key = record['s3']['object']['key']
            
            print(f"Processing {file_key} from {bucket_name}")
            
            # Prevent infinite loop! (If we save .json to same bucket, don't trigger again)
            if not file_key.lower().endswith('.pdf'):
                print(f"Skipping non-PDF file: {file_key}")
                continue
            
            # 2. Download PDF to /tmp (Lambda's only writable path)
            download_path = f"/tmp/{os.path.basename(file_key)}"
            s3.download_file(bucket_name, file_key, download_path)
            
            # 3. Process Code (Reuse logic from watcher.py)
            # A. Extract
            text = extract_text_from_pdf(download_path)
            cleaned_text = clean_text(text)
            
            # B. Classify
            category, score = doc_classifier.classify_text(cleaned_text)
            print(f"Category: {category} ({score})")
            
            # C. Extract Fields
            metadata = extractor.extract_fields(cleaned_text, category)
            print(f"Extracted: {metadata}")
            
            # 4. Save/Export (For now just print, can save to S3/DynamoDB)
            # Example: Save result to JSON in the same bucket (or output bucket)
            result_key = f"results/{os.path.basename(file_key)}.json"
            s3.put_object(
                Bucket=bucket_name,
                Key=result_key,
                Body=json.dumps(metadata)
            )
            print(f"Saved results to s3://{bucket_name}/{result_key}")
            
        return {
            'statusCode': 200,
            'body': json.dumps('File processed successfully!')
        }
        
    except Exception as e:
        print(f"Error: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps(f"Error: {str(e)}")
        }
