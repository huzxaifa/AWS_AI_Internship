import os
# Force cache to /tmp for AWS Lambda (Read-Only fix)
os.environ['HF_HOME'] = '/tmp'
os.environ['TRANSFORMERS_CACHE'] = '/tmp'

from transformers import pipeline

class DocumentClassifier:
    def __init__(self):
        # Zero-Shot classification
        # We use DistilBERT for speed and English accuracy (0.99+)
        # XLM-RoBERTa is too heavy (2.2GB) for quick local testing
        print("Loading Flexible Business Classifier (Zero-Shot: distilbert-base-uncased-mnli)...")
        self.classifier = pipeline("zero-shot-classification", model="typeform/distilbert-base-uncased-mnli")
        
        self.labels = [
            "Invoice",
            "Resume", 
            "Scientific document",
            "Email",
            "Project report"
        ]

    def classify_text(self, text):
        if not text or len(text.strip()) < 10:
            return "Unknown", 0.0

        # Truncate text to fit model (DistilBERT limit ~1024 tokens)
        truncated_text = text[:1024] 
        
        result = self.classifier(truncated_text, self.labels)
        
        top_label = result['labels'][0]
        top_score = result['scores'][0]
        
        return top_label, top_score
