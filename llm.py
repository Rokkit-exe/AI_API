from transformers import AutoModelForCausalLM, AutoTokenizer, MistralConfig, MistralModel, MistralForCausalLM
import torch
from utils import install_model, MODEL_BASE_PATH
import os
import warnings

model1 = "cognitivecomputations/dolphin-2.6-mistral-7b"
model2 = "cognitivecomputations/TinyDolphin-2.8-1.1b"

# Download the model and tokenizer from Hugging Face Hub
#install_model(model_name="huggingface/cognitivecomputations/dolphin-2.6-mistral-7b")

# Load the model and tokenizer
model_path = os.path.join(MODEL_BASE_PATH, model1)

# Initializing a Mistral 7B style configuration
#configuration = MistralConfig()

# Initializing a model from the Mistral 7B style configuration
#model = MistralModel(configuration)

# Accessing the model configuration
#configuration = model.config

def generate():
    device = "cuda" # the device to load the model onto
    print(f"Loading model from {model_path} to {device}")
    print("Loading model...")

    model = MistralForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, attn_implementation="flash_attention_2").to(device)
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    exit = False

    while not exit:
        prompt = input("Enter a prompt: ")

        if prompt == "exit":
            exit = True
            continue

        model_inputs = tokenizer([prompt], return_tensors="pt").to(device)
        model.to(device)

        generated_ids = model.generate(**model_inputs, max_new_tokens=3000, do_sample=True, pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id)
        print(tokenizer.batch_decode(generated_ids)[0])

if __name__ == '__main__':
    generate()
    