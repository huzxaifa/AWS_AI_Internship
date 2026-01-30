import boto3
import json

bucket_name = "huzaifas-rekognition-test"
images = ["1748370012264.jpg", "Earbuds MPOW.jpeg", "juice.png", "mountain_rocks.jfif"]

rekognition = boto3.client("rekognition")

results = {}

for image in images:
    response = rekognition.detect_labels(
        Image={
            "S3Object": {
                "Bucket": bucket_name,
                "Name": image
            }
        },
        MaxLabels=10,
        MinConfidence=70
    )

    results[image] = [
        {
            "Label": label["Name"],
            "Confidence": round(label["Confidence"], 2)
        }
        for label in response["Labels"]
    ]

# Save results
with open("rekognition_results.json", "w") as f:
    json.dump(results, f, indent=4)

print("Label detection complete. Results saved.")
