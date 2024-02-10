import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import hf_hub_download, HfApi
import torch
import json

# load environment variables
# from dotenv import load_dotenv
# load_dotenv()
# MODEL_BASE_PATH = os.getenv("MODEL_BASE_PATH")

MODEL_BASE_PATH = "./models"

# install model from huggingface hub in local directory
def install_model(model_name, model_path=MODEL_BASE_PATH):
    if not is_model_installed(model_path):
        create_model_dir(model_path)
        try:
            hf_hub_download(repo_id=model_name, repo_type=MODEL_BASE_PATH, path=model_path)
            print(f"Model {model_name} installed in {model_path}")
        except Exception as e:
            print(f"Unable to install model {model_name} in {model_path}\n\n{e}")        
    else:
        print(f"Model {model_name} already installed")

# load model from local directory
def load_model(model_path=MODEL_BASE_PATH):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    return model, tokenizer

# check if model is installed
def is_model_installed(model_path=MODEL_BASE_PATH, model_name="gpt2"):
    return os.path.exists(os.path.join(model_path, model_name))
    
# create model directory
def create_model_dir(model_path=MODEL_BASE_PATH):
    if not os.path.exists(model_path):
        os.makedirs(model_path)
        print(f"Directory {model_path} created")
    else:
        print(f"Directory {model_path} already exists")

# set device
def set_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def get_model_task(model_name):
    api = HfApi()
    model_info = api.model_info("distilbert-base-uncased")

    # The url to the model card (README file)
    print(model_info["id"])
    print(model_info["author"])
    #print(model_info["sha"])
    #print(model_info["library_name"])
    #print(model_info["tags"])
    #print(model_info["pipeline_tag"])
    #print(model_info["mask_token"])
    #print(json.dumps(model_info[0], indent=2))
