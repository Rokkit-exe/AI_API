from transformers import pipeline

class Summarizer:
    def __init__(self, model_name="t5-small", tokenizer='t5-small', model_path="./models", framework='pt'):
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.model_path = model_path
        self.framework = framework
        self.summarize = pipeline(
            task='summarization', 
            model=self.model_name, 
            tokenizer=self.tokenizer, 
            framework=self.framework
        )

    def get_summarize(self, text):
        return self.summarize(text)

