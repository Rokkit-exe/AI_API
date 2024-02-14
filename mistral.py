from transformers import MistralConfig, MistralForCausalLM, AutoTokenizer
from utils import is_model_installed, install_model, MODEL_BASE_PATH, get_model_info, install_model
import torch
import os

class Mistral:
    def __init__(self, model_name="cognitivecomputations/dolphin-2.6-mistral-7b"):

        self.model_path = self.build_model_path(model_name)

        if not is_model_installed(self.model_path):
            install_model(model_name=self.model_path)
        self.config = MistralConfig.from_pretrained(self.model_path)
        self.tokenizer = None
        self.model = None
        #self.model_info = get_model_info(self.model_path)
        if not torch.cuda.is_available():
            # raise error
            raise ValueError("No GPU found")
        else:
            self.device = "cuda"
    
    def generate(self, prompt):
        model_inputs = self.tokenizer([prompt], return_tensors="pt").to(self.device)
        self.model.to(self.device)

        generated_ids = self.model.generate(
            **model_inputs, 
            max_new_tokens=3000, 
            do_sample=True, 
            pad_token_id=self.tokenizer.eos_token_id, 
            eos_token_id=self.tokenizer.eos_token_id
        )
        return self.tokenizer.batch_decode(generated_ids)[0]
    
    def load_model(self):
        print(f"Loading model from {self.model_path} to {self.device}")
        print("Loading model...")

        self.model = MistralForCausalLM.from_pretrained(self.model_path, torch_dtype=torch.float16, attn_implementation="flash_attention_2").to(self.device)
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

    def build_model_path(self, model_name):
        return os.path.join(MODEL_BASE_PATH, model_name)
        