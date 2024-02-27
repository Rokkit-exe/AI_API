from transformers import LlamaConfig, LlamaForCausalLM, AutoTokenizer
import torch
import os
from utils import is_model_installed, install_model, install_model, build_model_path

class Llama:
    def __init__(self, model_name="cognitivecomputations/TinyDolphin-2.8.2-1.1b-laser"):
        self.model_path = build_model_path(model_name)
        if not is_model_installed(self.model_path):
            install_model(model_name=self.model_path)

        self.model_name = model_name
        self.config = LlamaConfig.from_pretrained(self.model_name)
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def generate(self, prompt):
        model_inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        generated_ids = self.model.generate(
            **model_inputs, 
            max_new_tokens=3000, 
            do_sample=True, 
            pad_token_id=self.tokenizer.eos_token_id, 
            eos_token_id=self.tokenizer.eos_token_id
        )
        return self.tokenizer.batch_decode(generated_ids)[0]
    
    def load_model(self):
        print(f"Loading model from {self.model_name} to {self.device}")
        self.model = LlamaForCausalLM.from_pretrained(self.model_name, torch_dtype=torch.float16, attn_implementation="flash_attention_2").to(self.device)
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True, device_map=0)



