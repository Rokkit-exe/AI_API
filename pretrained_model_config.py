from transformers import AutoConfig

class PretrainedModelConfig:
    def __init__(self, model_name="t5-small", tokenizer='t5-small', model_path="./models", framework='pt'):
        self.model_name = model_name
        self.model_path = model_path
        self.framework = framework
        self.tokenizer = tokenizer
        self.config = AutoConfig.from_pretrained(model_name)

    def get_config(self):
        return self.config