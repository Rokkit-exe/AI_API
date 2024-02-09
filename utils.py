import os
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import huggingface_hub


def install_model(model_name, model_path="./models"):
    if not is_model_installed(model_path):
        create_model_dir(model_path)
        try:
            huggingface_hub.hf_hub_download(repo_id=model_name, repo_type="model", path=model_path)
            print(f"Model {model_name} installed in {model_path}")
        except Exception as e:
            print(f"Unable to install model {model_name} in {model_path}\n\n{e}")        
    else:
        print(f"Model {model_name} already installed")


def load_model(model_path="./models"):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    return model, tokenizer

def is_model_installed(model_path="./models", model_name="gpt2"):
    return os.path.exists(os.path.join(model_path, model_name))
    
def create_model_dir(model_path="./models"):
    if not os.path.exists(model_path):
        os.makedirs(model_path)
        print(f"Directory {model_path} created")
    else:
        print(f"Directory {model_path} already exists")