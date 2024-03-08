from transformers import pipeline

class Fill_Mask:
    def __init__(self, model_name="bert-base-multilingual-uncased"):

        self.task = "fill-mask"
        self.model_name = model_name
        self.mask = "[MASK]"
    
    def fill_mask(self, prompt):
        pipe = pipeline(
            task=self.task, 
            model=self.model_name,
        )
        return pipe(prompt)

test = {
    "model": "bert-base-multilingual-uncased",
    "text": "what does my mom [MASK] when she is happy?"
}

