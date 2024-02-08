from transformers import pipeline

class Sentiment:
    def __init__(self, model_name="distilbert-base-uncased-finetuned-sst-2-english"):
        self.sentiment = pipeline('sentiment-analysis', model=model_name)

    def get_sentiment(self, text):
        return self.sentiment(text)
