import boto3
import json

s3 = boto3.client('s3')
bucket_name = 'huzaifas-tweets'
file_key = 'tweets.json'

obj = s3.get_object(Bucket=bucket_name, Key=file_key)
content = obj['Body'].read().decode('utf-8')

data = json.loads(content)
tweets_list = data['tweets']

comprehend = boto3.client('comprehend', region_name='us-east-1')


for tweet in tweets_list:
    text = tweet['text']
    sentiment = comprehend.detect_sentiment(Text=text, LanguageCode='en')
    print(f"Tweet: {text}\nSentiment: {sentiment['Sentiment']}, Confidence: {sentiment['SentimentScore']}\n")
