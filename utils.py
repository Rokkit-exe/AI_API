import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import hf_hub_download, HfApi, snapshot_download
import torch
import json
import subprocess

# load environment variables
from dotenv import load_dotenv
load_dotenv()

MODEL_BASE_PATH = "./models"

# huggingface hub login with hu
def hf_hub_login(token):
    try:
        subprocess.run(["huggingface-cli", "login", "--token", token])
        print("Logged in to huggingface hub")
    except Exception as e:
        print(f"Unable to login to huggingface hub\n\n{e}")

# install model from huggingface hub in local directory
def install_model(model_name, model_path=MODEL_BASE_PATH):
    # login to huggingface hub
    hf_hub_login(os.getenv("HUGGINGFACE_TOKEN"))

    # create model directory
    create_model_dir(model_name=model_name)

    local_dir = os.path.join(model_path, model_name)

    # download model from huggingface hub
    if not is_model_installed(model_path):
        try:
            snapshot_download(repo_id=model_name, local_dir=local_dir, revision="main")
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
def is_model_installed(model_name="gpt2", model_path=MODEL_BASE_PATH):
    return os.path.exists(os.path.join(model_path, model_name))
    
# create model directory
def create_model_dir(model_name, model_path=MODEL_BASE_PATH):
    model_path = os.path.join(model_path, model_name)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
        print(f"Directory {model_path} created")
    else:
        print(f"Directory {model_path} already exists")

# set device
def set_device():
    print("device info: ", json.dumps(get_device_info(), indent=4))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"Setting default device to {device}")
        torch.set_default_device(device)
    return device

# get device info
def get_device_info():
    return {
        "cuda_version": torch.version.cuda,
        "torch_version": torch.__version__,
        "device_available": torch.cuda.is_available()
    }

def get_model_task(model_name):
    api = HfApi()
    model_info = api.model_info("distilbert-base-uncased")
    print(model_info)
    # The url to the model card (README file)
    #print(model_info["id"])
    #print(model_info["author"])
    #print(model_info["sha"])
    #print(model_info["library_name"])
    #print(model_info["tags"])
    #print(model_info["pipeline_tag"])
    #print(model_info["mask_token"])
    #print(json.dumps(model_info[0], indent=2))

def is_pipeline_supported(pipeline):
    with open("pipelines.json", "r") as f:
        PIPELINES = json.load(f)
    for p in PIPELINES:
        if p["task"] == pipeline:
            return True
    return False

def get_models_from_pipeline(pipeline):
    with open("pipelines.json", "r") as f:
        PIPELINES = json.load(f)
    for p in PIPELINES:
        if p["task"] == pipeline:
            return p["models"]
