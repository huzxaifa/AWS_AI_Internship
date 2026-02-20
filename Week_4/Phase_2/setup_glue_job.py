import boto3
import json
import time

# Configuration
REGION = 'us-east-1'

BUCKET_NAME_PREFIX = 'huzaifas-etl-glue-script'
ROLE_NAME = 'AWSGlueServiceRole-crawler'
JOB_NAME = 'week4_log_processing_job'
SCRIPT_LOCAL_PATH = 'glue_etl_script.py'
SCRIPT_S3_KEY = 'scripts/glue_etl_script.py'
LOG_GROUP = 'week4/app-logs'

s3 = boto3.client('s3', region_name=REGION)
iam = boto3.client('iam', region_name=REGION)
glue = boto3.client('glue', region_name=REGION)

def get_bucket_name():
    # Find the bucket we created earlier
    response = s3.list_buckets()
    for bucket in response['Buckets']:
        if bucket['Name'].startswith(BUCKET_NAME_PREFIX):
            return bucket['Name']
    
    # Fallback: ask user to hardcode if not found
    print("Could not likely find the bucket from Task 1.")
    return input("Please enter your S3 Bucket Name: ")

def upload_script(bucket_name):
    print(f"Uploading Glue script to s3://{bucket_name}/{SCRIPT_S3_KEY}...")
    s3.upload_file(SCRIPT_LOCAL_PATH, bucket_name, SCRIPT_S3_KEY)

def create_iam_role():
    print(f"Creating IAM Role: {ROLE_NAME}...")
    assume_role_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"Service": "glue.amazonaws.com"},
                "Action": "sts:AssumeRole"
            }
        ]
    }
    
    try:
        try:
            role = iam.get_role(RoleName=ROLE_NAME)
            print(f"Role {ROLE_NAME} already exists.")
        except iam.exceptions.NoSuchEntityException:
            role = iam.create_role(
                RoleName=ROLE_NAME,
                AssumeRolePolicyDocument=json.dumps(assume_role_policy)
            )
            print("Role created.")

        # Attach policies
        # Need S3 Full Access (to write) and CloudWatch Logs Read
        iam.attach_role_policy(RoleName=ROLE_NAME, PolicyArn='arn:aws:iam::aws:policy/service-role/AWSGlueServiceRole')
        iam.attach_role_policy(RoleName=ROLE_NAME, PolicyArn='arn:aws:iam::aws:policy/AmazonS3FullAccess') 
        iam.attach_role_policy(RoleName=ROLE_NAME, PolicyArn='arn:aws:iam::aws:policy/CloudWatchLogsReadOnlyAccess')
        
        print("Attached policies: GlueServiceRole, S3FullAccess, CloudWatchLogsReadOnlyAccess")
        time.sleep(10) # Wait for propagation
        return role['Role']['Arn']
        
    except Exception as e:
        print(f"Error creating role: {e}")
        return None

def create_glue_job(role_arn, bucket_name):
    print(f"Creating Glue Job: {JOB_NAME}...")
    
    try:
        glue.create_job(
            Name=JOB_NAME,
            Description='Week 4 Task 2: Process CloudWatch Logs',
            Role=role_arn,
            Command={
                'Name': 'pythonshell', # Using Python Shell for cost/simplicity
                'ScriptLocation': f's3://{bucket_name}/{SCRIPT_S3_KEY}',
                'PythonVersion': '3.9'
            },
            DefaultArguments={
                '--BUCKET_NAME': bucket_name,
                '--LOG_GROUP': LOG_GROUP
            },
            MaxCapacity=0.0625 # Minimal capacity for Python Shell (cheaper)
        )
        print(f"Glue Job {JOB_NAME} created successfully.")
    except glue.exceptions.AlreadyExistsException:
        print(f"Glue Job {JOB_NAME} already exists. Updating...")
        glue.update_job(
            JobName=JOB_NAME,
            JobUpdate={
                'Role': role_arn,
                'Command': {
                    'Name': 'pythonshell',
                    'ScriptLocation': f's3://{bucket_name}/{SCRIPT_S3_KEY}',
                    'PythonVersion': '3.9'
                },
                'DefaultArguments': {
                    '--BUCKET_NAME': bucket_name,
                    '--LOG_GROUP': LOG_GROUP
                },
                'MaxCapacity': 0.0625
            }
        )
        print("Glue Job updated.")
    except Exception as e:
        print(f"Error creating Glue Job: {e}")

def main():
    bucket_name = get_bucket_name()
    print(f"Using Bucket: {bucket_name}")
    
    upload_script(bucket_name)
    
    role_arn = create_iam_role()
    if role_arn:
        create_glue_job(role_arn, bucket_name)
        print("\n--- Setup Complete ---")
        print(f"You can now run the job '{JOB_NAME}' from the Glue Console.")
    else:
        print("Failed to create role. Exiting.")

if __name__ == '__main__':
    main()
