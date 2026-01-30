import boto3
from PIL import Image
import io

s3 = boto3.client('s3')
DEST_BUCKET = 'huzaifas-images-resized'

def lambda_handler(event, context):
    for record in event['Records']:
        bucket = record['s3']['bucket']['name']
        key = record['s3']['object']['key']

        response = s3.get_object(Bucket=bucket, Key=key)
        image = Image.open(response['Body'])
        image.thumbnail((300, 300))

        buffer = io.BytesIO()
        image.save(buffer, 'JPEG')
        buffer.seek(0)

        s3.put_object(
            Bucket=DEST_BUCKET,
            Key=key,
            Body=buffer,
            ContentType='image/jpeg'
        )

    return {"status": "success"}
