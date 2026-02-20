import boto3
import time
import random
from datetime import datetime

# Configuration
LOG_GROUP = 'week4/app-logs'
LOG_STREAM = 'stream-1'
REGION = 'us-east-1'

logs = boto3.client('logs', region_name=REGION)

def create_log_group():
    try:
        logs.create_log_group(logGroupName=LOG_GROUP)
        print(f"Created Log Group: {LOG_GROUP}")
    except logs.exceptions.ResourceAlreadyExistsException:
        print(f"Log Group {LOG_GROUP} already exists.")

def create_log_stream():
    try:
        logs.create_log_stream(logGroupName=LOG_GROUP, logStreamName=LOG_STREAM)
        print(f"Created Log Stream: {LOG_STREAM}")
    except logs.exceptions.ResourceAlreadyExistsException:
        print(f"Log Stream {LOG_STREAM} already exists.")

def generate_logs(count=50):
    print(f"Generating {count} log events...")
    
    messages = [
        ("INFO", "User logged in successfully"),
        ("INFO", "Data processing started"),
        ("ERROR", "Database connection failed"),
        ("INFO", "File uploaded to S3"),
        ("ERROR", "Timeout waiting for API response"),
        ("INFO", "Health check passed"),
        ("ERROR", "NullPointerException in module X"),
        ("INFO", "Job completed successfully")
    ]
    
    
    log_events = []
    for _ in range(count):
        level, msg = random.choice(messages)
        # Use current time for each log to ensure they are picked up by the 1-hour window query
        timestamp = int(time.time() * 1000)
        log_events.append({
            'timestamp': timestamp,
            'message': f"[{level}] {msg} - RequestID: {random.randint(1000,9999)}"
        })
        time.sleep(0.01) # Small delay to vary timestamps slightly

    try:
        logs.put_log_events(
            logGroupName=LOG_GROUP,
            logStreamName=LOG_STREAM,
            logEvents=log_events
        )
        print("Successfully pushed logs to CloudWatch.")
    except Exception as e:
        print(f"Error pushing logs: {e}")

if __name__ == '__main__':
    create_log_group()
    create_log_stream()
    generate_logs()
