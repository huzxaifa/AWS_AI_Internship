import boto3

comprehend = boto3.client(
    service_name='comprehend',
    region_name='us-east-1'
)

# Sample tweets
tweets = [
    "Pakistan’s military junta led by psychopath Field Marshal Asim Munir forced an emergency medical procedure on the eyes of abducted former Prime Minister Imran Khan in the middle of the night - WITHOUT INFORMING HIS FAMILY - transporting the former PM to the capital’s PIMS medical facility from Adiyala where the former PM has been kept in isolation for 120 days - out of 900 days incommunicado!",
    "I’m deeply sad to learn of the passing of Dr. Bill Foege. Bill was a towering figure in global health—a man who saved the lives of literally hundreds of millions of people. He was also a friend and mentor who gave me a deep grounding in the history of global health and inspired me with his conviction that the world could do more to alleviate suffering.​",
    "A big step forward in the fight against Alzheimer’s: The FDA approved the first blood test to help diagnose the disease. Breakthroughs like this will make earlier, easier diagnosis possible—bringing us closer to better treatments and, someday, a cure.",
    "In a rare sight, a female vocalist named Noreen Afzal leads a qawwal party & performs a Kafi 'ki jaana main kaun' of Hazrat Bulleh Shah. Miss Noreen hails from Gujranwala; started from reciting naats, manqabats, marsiyas before eventually focusing on qawwali."
]

for tweet in tweets:
    response = comprehend.detect_sentiment(
        Text=tweet,
        LanguageCode='en'
    )

    print("Tweet:", tweet)
    print("Sentiment:", response['Sentiment'])
    print("Confidence Scores:", response['SentimentScore'])
    print("-" * 50)

