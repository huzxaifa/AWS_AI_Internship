# Task 1: AWS Glue Crawler — Catalog CSV Data from S3

## Objective
Create an AWS Glue Crawler to automatically detect the schema of a CSV file stored in Amazon S3 and populate the AWS Glue Data Catalog. The resulting table is queryable in Amazon Athena.

---

## Architecture Overview

```
employees.csv  →  Amazon S3  →  AWS Glue Crawler  →  Glue Data Catalog  →  Amazon Athena
```

---

## Dataset

`employees.csv` contains employees data with 10 columns:

| Column | Type |
|---|---|
| employee_id | bigint |
| first_name | string |
| last_name | string |
| email | string |
| phone_number | string |
| hire_date | string |
| job_id | string |
| salary | bigint |
| manager_id | bigint |
| department_id | bigint |

---

## Steps Taken

### 1. Prepare the S3 Bucket
- Created bucket `huzaifas-input-crawler-data`.
- Uploaded `employees.csv` to `s3://huzaifas-input-crawler-data/employees/employees.csv`.
  > **Important**: Place the CSV inside a dedicated subfolder (e.g., `employees/`) so the crawler doesn't merge schemas from different datasets in the same root path.

### 2. Create IAM Role
Created role `AWSGlueServiceRole-crawler` with the following managed policies:
- `AWSGlueServiceRole` — grants Glue service permissions.
- `AmazonS3ReadOnlyAccess` — grants Glue read access to S3.

### 3. Create Glue Database
Created database `week4_data_engineering_db` in the AWS Glue Data Catalog.

### 4. Create table
Created table specifically defined for employees dataset with added schema.

### 5. Create & Configure the Glue Crawler
- **Name**: `week4_dataeng_crawler`
- **Data Source**: `s3://huzaifas-input-crawler-data/employees/`
- **IAM Role**: `AWSGlueServiceRole-crawler`
- **Output Database**: `week4_data_engineering_db`
- **Schema Change Policy**: Update in database

### 6. Run the Crawler
Ran the crawler from the AWS Glue Console. It completed successfully and created table `huzaifas_employees_crawler` with all 11 columns detected automatically.

### 7. Attached the created table with crawler
Attached the created table in the same database with the crawler.

---

## Verification

Queried the table in Amazon Athena:

```sql
SELECT * FROM "week4_data_engineering_db"."huzaifas_employees_crawler" LIMIT 10;
```

Result: All 10 employee records returned with correct column names and data types.

---

## Issues Encountered & Fixes

| Issue | Cause | Fix |
|---|---|---|
| `AccessDenied` on S3 | IAM Role missing S3 policy | Attached `AmazonS3ReadOnlyAccess` to the role |
| `Insufficient Lake Formation permission` | Lake Formation not granting access to role | Granted `Super` permission to Glue role in Lake Formation |
| `COLUMN_NOT_FOUND` in Athena | User account had no Lake Formation permissions | Granted `Select + Describe` to IAM user in Lake Formation |
| `compressionType` mismatch warn | Stale table metadata from old crawl | Deleted old table, re-ran crawler to create fresh one |
| `BAD_DATA: NumberFormatException` | Header row included in data (data type mismatch) | Deleted table, re-ran crawler; new table had `skip.header.line.count=1` |
| Multiple tables skipped | CSV and JSON under same root S3 prefix | Moved CSV into `employees/` subfolder; pointed crawler to that prefix |
