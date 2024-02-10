from transformers import pipeline, BertConfig, BertModel, BertTokenizer
from utils import download_model, save_model

class Sentiment:
    def __init__(self, model_name="distilbert-base-uncased-finetuned-sst-2-english"):
        self.__sentiment = pipeline('sentiment-analysis', model=model_name)
        self.__config = BertConfig.from_pretrained(model_name)
        self.__model = BertModel.from_pretrained(model_name)
        self.__tokenizer = BertTokenizer.from_pretrained(model_name)

    def get_sentiment(self, text):
        return self.__sentiment(text)
    
    def get_config(self):
        return self.__config.get_config_dict()
    
    def update_config(self, config_dict):
        self.__config = BertConfig.from_dict(config_dict)
        self.__model = BertModel(self.__config)
        self.__tokenizer = BertTokenizer(self.__config)
    
    def get_model(self):
        return self.__model
    
    def get_tokenizer(self):
        return self.__tokenizer
    
    def tokenize_input(self, text):
        return self.__tokenizer(text, return_tensors="pt")
    
    def get_input_ids(self, text):
        return self.tokenize_input(text)["input_ids"]
    
    def get_output(self, input_ids):
        return self.__model(input_ids)
