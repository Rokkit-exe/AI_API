import os
from transformers import pipeline
from utils import is_model_installed

models = {
    "Falconsai/text_summarization"
}

class Summarization:
    def __init__(self, model_name="t5-small", tokenizer='t5-small', model_path="./models", framework='pt'):
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.model_path = model_path
        self.framework = framework
        self.pipeline = pipeline(
            task='summarization', 
            model= os.path.join(model_path, model_name) if is_model_installed(self.model_path, self.model_name) else self.model_name, 
            tokenizer=self.tokenizer, 
            framework=self.framework
        )

    def get_summarization(self, text):
        return self.pipeline(text)
    
    
