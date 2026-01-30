# Week 1 – AWS AI & ML Foundations

This folder contains the **Week 1 tasks** completed as part of my internship at **Cloud Elligent**.  
The focus of this week was to get hands-on exposure to **AWS AI services**, understand how they integrate with S3 and Python (Boto3), and explore the **SageMaker Studio environment**.

---

## Task 1: Image Resizing with AWS Lambda
In this task, an automated image processing workflow was built using **AWS Lambda and Amazon S3**.

- Images uploaded to an S3 bucket are automatically resized
- The resized images are stored in a separate destination bucket
- AWS Lambda handles the image processing logic
- IAM roles and permissions were configured with least-privilege access
- A scheduled cleanup mechanism was later added using EventBridge

**Goal:** Understand serverless image processing and event-driven architecture on AWS.

---

## Task 2: Image Analysis with Amazon Rekognition
This task focused on using **Amazon Rekognition** for image analysis.

- Images were stored in an S3 bucket
- A Python script using **Boto3** called `detect_labels`
- Detected labels and confidence scores were extracted
- Results were saved locally for review

**Goal:** Learn how AWS Rekognition performs automatic image labeling and how to integrate it with Python.

---

## Task 3: Sentiment Analysis using Amazon Comprehend
In this task, **Amazon Comprehend** was used for natural language processing.

- Tweets or sample text were analyzed using `detect_sentiment`
- Sentiment results (Positive, Negative, Neutral, Mixed) were printed and saved
- AWS CloudShell and Boto3 were used for execution

**Goal:** Understand basic NLP capabilities offered by AWS and how sentiment analysis works.
