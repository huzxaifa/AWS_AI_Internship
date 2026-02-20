import sys
import boto3
import json
import datetime
import os
from awsglue.utils import getResolvedOptions

# This script runs INSIDE AWS Glue


try:
    args = getResolvedOptions(sys.argv, ['BUCKET_NAME', 'LOG_GROUP'])
    BUCKET_NAME = args['BUCKET_NAME']
    LOG_GROUP = args['LOG_GROUP']
except:
    BUCKET_NAME = 'huzaifas-input-crawler-data'
    LOG_GROUP = 'week4/app-logs'

REGION = 'us-east-1'
s3 = boto3.client('s3', region_name=REGION)
logs = boto3.client('logs', region_name=REGION)

def process_logs():
    print(f"Starting ETL Job for Log Group: {LOG_GROUP}")
    print(f"Output Bucket: {BUCKET_NAME}")
    
    # 1. Read Logs from CloudWatch
    end_time = int(datetime.datetime.now().timestamp() * 1000)
    start_time = end_time - (86400 * 1000) # 24 hours ago
    
    print(f"Querying logs from {start_time} to {end_time}")
    
    query = "fields @timestamp, @message | sort @timestamp desc | limit 1000"
    
    print("Running CloudWatch Logs Query...")
    start_query_response = logs.start_query(
        logGroupName=LOG_GROUP,
        startTime=start_time,
        endTime=end_time,
        queryString=query,
    )
    
    query_id = start_query_response['queryId']
    
    status = 'Running'
    while status in ['Running', 'Scheduled']:
        print("Waiting for query execution...")
        response = logs.get_query_results(queryId=query_id)
        status = response['status']
        
    results = response['results']
    print(f"Found {len(results)} log entries.")
    
    # 2. Process and Filter Logs
    processed_data = []
    
    for row in results:
        # row is a list of fields [{'field': '@timestamp', 'value': '...'}, ...]
        # Convert to dict for easier access
        data = {item['field']: item['value'] for item in row}
        
        message = data.get('@message', '')
        timestamp = data.get('@timestamp', '')
        
        # Parse timestamp to get date for partitioning
        # CloudWatch returns string timestamp usually
        try:
            # Example: 2023-10-27 10:00:00.000
            dt = datetime.datetime.strptime(timestamp.split('.')[0], '%Y-%m-%d %H:%M:%S')
            date_str = dt.strftime('%Y-%m-%d')
        except:
             date_str = datetime.datetime.now().strftime('%Y-%m-%d')
        
        tag = 'UNKNOWN'
        if '[INFO]' in message or 'INFO' in message:
            tag = 'INFO'
        elif '[ERROR]' in message or 'ERROR' in message:
            tag = 'ERROR'
            
        if tag in ['INFO', 'ERROR']:
            processed_data.append({
                'log_timestamp': timestamp,
                'message': message,
                'tag': tag,
                'date': date_str
            })
            
    print(f"Filtered down to {len(processed_data)} relevant records.")
    
    # 3. Write to S3 (Partitioned by Date)
    # Key format: output_logs/date=YYYY-MM-DD/logs.json
    
    grouped_data = {}
    for item in processed_data:
        date_key = item['date']
        if date_key not in grouped_data:
            grouped_data[date_key] = []
        grouped_data[date_key].append(item)
        
    for date_key, items in grouped_data.items():
        # Create NDJSON content (Newline Delimited JSON)
        # IMPORTANT: Remove 'date' from the content because it's already the partition key.
        # Otherwise, Athena sees it as a duplicate column.
        clean_items = []
        for item in items:
            item_copy = item.copy()
            if 'date' in item_copy:
                del item_copy['date']
            clean_items.append(item_copy)
            
        content = '\n'.join([json.dumps(item) for item in clean_items])
        
        file_key = f"processed_logs/date={date_key}/logs_{int(datetime.datetime.now().timestamp())}.json"
        
        print(f"Writing {len(items)} records to s3://{BUCKET_NAME}/{file_key}")
        s3.put_object(
            Bucket=BUCKET_NAME,
            Key=file_key,
            Body=content
        )
        
    print("ETL Job Completed Successfully.")

if __name__ == '__main__':
    process_logs()
