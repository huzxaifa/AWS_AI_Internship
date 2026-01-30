import boto3
import os

bucket_name = "huzaifas-rekognition-test"
image_folder = "images"

s3 = boto3.client("s3")

for image in os.listdir(image_folder):
    if image.lower().endswith((".jpg", ".jpeg", ".png", ".jfif")):
        s3.upload_file(
            os.path.join(image_folder, image),
            bucket_name,
            image
        )
        print(f"Uploaded {image}")
