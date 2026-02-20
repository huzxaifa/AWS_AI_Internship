import boto3
import json
import time
import os

# Configuration
REGION = 'us-east-1' # Change if needed
BUCKET_NAME = f'week4-glue-crawler-source-{int(time.time())}' # Unique bucket name
FILE_NAME = 'employees.csv'
FILE_PATH = 'employees.csv' # Assumes script and file are in the same directory
S3_FOLDER = 'week4_data/'
DATABASE_NAME = 'week4_data_engineering_db'
CRAWLER_NAME = 'week4_crawler'
ROLE_NAME = 'AWSGlueServiceRole-Week4'

# Initialize clients
s3 = boto3.client('s3', region_name=REGION)
glue = boto3.client('glue', region_name=REGION)
iam = boto3.client('iam', region_name=REGION)

def create_s3_bucket():
    try:
        if REGION == 'us-east-1':
            s3.create_bucket(Bucket=BUCKET_NAME)
        else:
            s3.create_bucket(
                Bucket=BUCKET_NAME,
                CreateBucketConfiguration={'LocationConstraint': REGION}
            )
        print(f"Created S3 bucket: {BUCKET_NAME}")
    except Exception as e:
        print(f"Error creating bucket: {e}")

def upload_file():
    try:
        s3.upload_file(FILE_PATH, BUCKET_NAME, f"{S3_FOLDER}{FILE_NAME}")
        print(f"Uploaded {FILE_NAME} to s3://{BUCKET_NAME}/{S3_FOLDER}")
    except Exception as e:
        print(f"Error uploading file: {e}")
        # Try finding the file in current directory if relative path fails
        try:
             s3.upload_file(FILE_NAME, BUCKET_NAME, f"{S3_FOLDER}{FILE_NAME}")
             print(f"Uploaded {FILE_NAME} to s3://{BUCKET_NAME}/{S3_FOLDER}")
        except Exception as e2:
             print(f"Error uploading file (retry): {e2}")

def create_iam_role():
    try:
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
            role = iam.get_role(RoleName=ROLE_NAME)
            print(f"IAM Role {ROLE_NAME} already exists.")
            return role['Role']['Arn']
        except iam.exceptions.NoSuchEntityException:
            role = iam.create_role(
                RoleName=ROLE_NAME,
                AssumeRolePolicyDocument=json.dumps(assume_role_policy)
            )
            print(f"Created IAM Role: {ROLE_NAME}")
            
            # Attach policies
            iam.attach_role_policy(
                RoleName=ROLE_NAME,
                PolicyArn='arn:aws:iam::aws:policy/service-role/AWSGlueServiceRole'
            )
            iam.attach_role_policy(
                RoleName=ROLE_NAME,
                PolicyArn='arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess'
            )
            print("Attached policies to role.")
            # Verify role propagation
            time.sleep(10) 
            return role['Role']['Arn']
            
    except Exception as e:
        print(f"Error managing IAM role: {e}")
        return None

def create_glue_database():
    try:
        glue.create_database(
            DatabaseInput={
                'Name': DATABASE_NAME,
                'Description': 'Database for Week 4 Data Engineering assignment'
            }
        )
        print(f"Created Glue Database: {DATABASE_NAME}")
    except glue.exceptions.AlreadyExistsException:
        print(f"Glue Database {DATABASE_NAME} already exists.")
    except Exception as e:
        print(f"Error creating Glue Database: {e}")

def create_glue_crawler(role_arn):
    try:
        glue.create_crawler(
            Name=CRAWLER_NAME,
            Role=role_arn,
            DatabaseName=DATABASE_NAME,
            Targets={
                'S3Targets': [
                    {'Path': f's3://{BUCKET_NAME}/{S3_FOLDER}'}
                ]
            },
            SchemaChangePolicy={
                'UpdateBehavior': 'UPDATE_IN_DATABASE',
                'DeleteBehavior': 'DEPRECATE_IN_DATABASE'
            }
        )
        print(f"Created Glue Crawler: {CRAWLER_NAME}")
    except glue.exceptions.AlreadyExistsException:
        print(f"Glue Crawler {CRAWLER_NAME} already exists. Updating...")
        glue.update_crawler(
             Name=CRAWLER_NAME,
            Role=role_arn,
            DatabaseName=DATABASE_NAME,
            Targets={
                'S3Targets': [
                    {'Path': f's3://{BUCKET_NAME}/{S3_FOLDER}'}
                ]
            }
        )
    except Exception as e:
        print(f"Error creating Glue Crawler: {e}")

def run_crawler():
    try:
        glue.start_crawler(Name=CRAWLER_NAME)
        print(f"Started Glue Crawler: {CRAWLER_NAME}")
        
        print("Waiting for crawler to complete...")
        while True:
            response = glue.get_crawler(Name=CRAWLER_NAME)
            state = response['Crawler']['State']
            print(f"Crawler state: {state}")
            
            if state in ['READY', 'STOPPING']:
                if 'LastCrawl' in response['Crawler']:
                    status = response['Crawler']['LastCrawl']['Status']
                    print(f"Crawler finished with status: {status}")
                break
            
            time.sleep(10)
            
    except Exception as e:
        print(f"Error running Glue Crawler: {e}")

def main():
    print("Starting Week 4 Setup...")
    create_s3_bucket()
    upload_file()
    
    role_arn = create_iam_role()
    if role_arn:
        create_glue_database()
        create_glue_crawler(role_arn)
        run_crawler()
    else:
        print("Failed to get Role ARN. Exiting.")

if __name__ == '__main__':
    main()
