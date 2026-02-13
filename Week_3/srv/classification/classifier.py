from transformers import pipeline

class DocumentClassifier:
    def __init__(self):
        # Zero-Shot classification
        print("Loading Flexible Business Classifier (Zero-Shot: joeddav/xlm-roberta-large-xnli)...")
        self.classifier = pipeline("zero-shot-classification", model="joeddav/xlm-roberta-large-xnli")
        
        self.labels = [
            "Invoice",
            "Resume", 
            "Technical document",
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
