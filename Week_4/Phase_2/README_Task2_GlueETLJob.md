# Task 2: AWS Glue ETL Job — Process CloudWatch Logs

## Objective
Create an AWS Glue ETL job (Python Shell) that retrieves logs from Amazon CloudWatch, filters them by `ERROR` and `INFO` tags, and stores the processed output in Amazon S3 partitioned by date. The processed data is then cataloged and queried via Amazon Athena.

---

## Architecture Overview

```
generate_logs.py  →  CloudWatch Logs  →  Glue ETL Job  →  S3 (processed_logs/)  →  Glue Crawler  →  Athena
```

---

## Files

| File | Description |
|---|---|
| `generate_logs.py` | Generates dummy INFO/ERROR logs into CloudWatch |
| `glue_etl_script.py` | Glue Job script — reads CloudWatch, filters & writes to S3 |
| `setup_glue_job.py` | Deploys the Glue Job (uploads script, creates IAM role & job) |
| `troubleshoot_crawler.py` | Diagnoses crawler, S3, and table schema issues |

---

## Steps Taken

### 1. Generate Dummy Logs
Ran `generate_logs.py` in AWS Cloud Shell to create the CloudWatch Log Group and push 50 sample log events.

- **Log Group**: `week4/app-logs`
- **Log Stream**: `stream-1`
- **Log Format**: `[INFO] <message>` or `[ERROR] <message>`

```bash
python3 generate_logs.py
```

### 2. Deploy the Glue Job
Ran `setup_glue_job.py` in Cloud Shell. This script:
1. Uploaded `glue_etl_script.py` to `s3://huzaifas-etl-glue-script/scripts/glue_etl_script.py`.
2. Created IAM Role `AWSGlueServiceRole-crawler` with:
   - `AWSGlueServiceRole`
   - `AmazonS3FullAccess`
   - `CloudWatchLogsReadOnlyAccess`
3. Created Glue Job `week4_log_processing_job` (Python Shell, `MaxCapacity=0.0625`).

```bash
python3 setup_glue_job.py
```

### 3. ETL Job Logic (`glue_etl_script.py`)
The job performs three steps:

**A. Read** — Queries CloudWatch Logs Insights for the past 24 hours:
```
fields @timestamp, @message | sort @timestamp desc | limit 1000
```

**B. Transform** — Filters each log entry:
- Keeps only lines containing `INFO` or `ERROR`.
- Extracts the `date` for partitioning.
- Removes `date` from the JSON body (avoids Hive duplicate column error).

**C. Write** — Saves to S3 partitioned by date:
```
s3://huzaifas-etl-glue-script/processed_logs/date=YYYY-MM-DD/logs_<timestamp>.json
```

### 4. Run the Job
- Went to **Glue Console** → **ETL Jobs** → `week4_log_processing_job` → **Run**.
- Job status changed to **Succeeded**.

### 5. Re-run Crawler
Added `s3://huzaifas-etl-glue-script/processed_logs/` as a second target to the existing crawler and re-ran it. This created a new table `huzaifas_glue_job` in the Glue Data Catalog.

### 6. Grant Lake Formation Permissions
Granted **Select + Describe** to the IAM user on the new table via Lake Formation.

---

## Verification

**All logs:**
```sql
SELECT * FROM "week4_data_engineering_db"."huzaifas_glue_job" LIMIT 10;
```

**Filter ERROR logs only:**
```sql
SELECT * FROM "week4_data_engineering_db"."huzaifas_glue_job"
WHERE tag = 'ERROR'
LIMIT 10;
```

**Filter by date:**
```sql
SELECT * FROM "week4_data_engineering_db"."huzaifas_glue_job"
WHERE date = '2026-02-19'
LIMIT 10;
```

---

## Issues Encountered & Fixes

| Issue | Cause | Fix |
|---|---|---|
| `ParamValidationError: Unknown parameter LogGroupName` | CloudWatch Logs Boto3 API uses camelCase params | Changed to `logGroupName`, `logStreamName`, `logEvents` |
| Job found 0 log entries | 1-hour query window too narrow / timezone mismatch | Extended query window to 24 hours |
| `HIVE_INVALID_METADATA: duplicate columns` | `date` field in JSON body AND as partition key | Removed `date` key from JSON records before writing to S3 |
| `COLUMN_NOT_FOUND` in Athena | Lake Formation permissions not granted on new table | Granted `Select + Describe` to IAM user for the table |
| `huzaifas_glue_job` table skipped in crawler | Multiple schemas under same root S3 prefix | Separated datasets into distinct subfolders; added separate crawler targets |
