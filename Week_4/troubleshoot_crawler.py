import boto3
import json
from datetime import datetime

# Configuration
CRAWLER_NAME = 'week4_dataeng_crawler'
DATABASE_NAME = 'week4_data_engineering_db'
REGION = 'us-east-1'

glue = boto3.client('glue', region_name=REGION)
s3 = boto3.client('s3', region_name=REGION)

def check_crawler():
    print(f"--- Checking Crawler: {CRAWLER_NAME} ---")
    try:
        response = glue.get_crawler(Name=CRAWLER_NAME)
        crawler = response['Crawler']
        
        print(f"Status: {crawler['State']}")
        if 'LastCrawl' in crawler:
            last_crawl = crawler['LastCrawl']
            print(f"Last Crawl Status: {last_crawl.get('Status')}")
            print(f"Last Crawl Message: {last_crawl.get('ErrorMessage', 'No error message')}")
            print(f"Log Group: {last_crawl.get('LogGroup')}")
            print(f"Log Stream: {last_crawl.get('LogStream')}")
        else:
            print("No last crawl info available.")
            
        # Check targets
        if 'Targets' in crawler and 'S3Targets' in crawler['Targets']:
            for target in crawler['Targets']['S3Targets']:
                path = target['Path']
                print(f"S3 Target Path: {path}")
                check_s3_path(path)
                
                # Also check processed_logs specifically if it's a root bucket path
                if path.endswith('/'):
                    check_s3_path(path + 'processed_logs/')
        else:
            print("No S3 targets found!")
            
    except glue.exceptions.EntityNotFoundException:
        print(f"Crawler {CRAWLER_NAME} not found.")
    except Exception as e:
        print(f"Error checking crawler: {e}")

def check_s3_path(s3_path):
    print(f"--- Checking S3 Path: {s3_path} ---")
    try:
        if s3_path.startswith('s3://'):
            s3_path = s3_path[5:]
        
        parts = s3_path.split('/', 1)
        bucket = parts[0]
        prefix = parts[1] if len(parts) > 1 else ''
        
        print(f"Bucket: {bucket}")
        print(f"Prefix: {prefix}")
        
        response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
        if 'Contents' in response:
            print(f"Found {response['KeyCount']} files in path:")
            for obj in response['Contents']:
                print(f" - {obj['Key']} (Size: {obj['Size']} bytes)")
        else:
            print("No files found in this S3 path! This is likely the issue.")
            
    except Exception as e:
        print(f"Error checking S3 path: {e}")

def check_database():
    print(f"--- Checking Database: {DATABASE_NAME} ---")
    try:
        glue.get_database(Name=DATABASE_NAME)
        print(f"Database {DATABASE_NAME} exists.")
        
        response = glue.get_tables(DatabaseName=DATABASE_NAME)
        if 'TableList' in response and len(response['TableList']) > 0:
            print(f"Found {len(response['TableList'])} tables:")
            for table in response['TableList']:
                print(f" - Table Name: {table['Name']}")
                print(f"   - Description: {table.get('Description', 'N/A')}")
                print(f"   - TableType: {table.get('TableType', 'N/A')}")
                
                # Check Columns
                columns = table['StorageDescriptor']['Columns']
                if columns:
                    print(f"   - Columns ({len(columns)}):")
                    for col in columns:
                        print(f"     * {col['Name']} ({col['Type']})")
                else:
                    print(f"   - [WARNING] No columns found in table definition!")

                # Check Serialization/Format
                serde_info = table['StorageDescriptor']['SerdeInfo']
                print(f"   - Serialization Lib: {serde_info.get('SerializationLibrary', 'N/A')}")
                print(f"   - Parameters: {table.get('Parameters', {})}")
        else:
            print("No tables found in database.")
            
    except glue.exceptions.EntityNotFoundException:
        print(f"Database {DATABASE_NAME} not found.")
    except Exception as e:
        print(f"Error checking database: {e}")

if __name__ == '__main__':
    check_crawler()
    check_database()
