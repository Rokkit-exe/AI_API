from transformers import pipeline

class Sentiment:
    def __init__(self, model_name="distilbert-base-uncased-finetuned-sst-2-english"):
        self.model_name = model_name
        self.task = "sentiment-analysis"

    def get_sentiment(self, text):
        pipe = pipeline(
            task=self.task, 
            model=self.model_name
        )
        return pipe(text)

    
