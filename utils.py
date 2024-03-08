import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import hf_hub_download, HfApi, snapshot_download, hf_api
import torch
import json
import subprocess
import shutil

# load environment variables
from dotenv import load_dotenv
load_dotenv()

MODELS_PATH = os.getenv("MODELS_PATH")

# huggingface hub login with hu
def hf_hub_login(token):
    try:
        subprocess.run(["huggingface-cli", "login", "--token", token])
        print("Logged in to huggingface hub")
    except Exception as e:
        print(f"Unable to login to huggingface hub\n\n{e}")

def build_model_path(model_name, model_path=MODELS_PATH):
    return os.path.join(model_path, model_name)

# install model from huggingface hub in local directory
def install_model(model_name, model_path=MODELS_PATH):
    # login to huggingface hub
    hf_hub_login(os.getenv("HUGGINGFACE_TOKEN"))

    # create model directory
    create_model_dir(model_name=model_name)

    local_dir = os.path.join(model_path, model_name)

    # download model from huggingface hub
    try:
        snapshot_download(repo_id=model_name, local_dir=local_dir, revision="main")
        update_model_info(action="add", category="installed", model_name=model_name)
        print(f"Model {model_name} installed in {model_path}")
    except Exception as e:
        print(f"Unable to install model {model_name} in {model_path}\n\n{e}")

# install dataset from huggingface hub in local directory
def install_dataset(dataset_name, dataset_path=MODELS_PATH):
    print(f"Installing dataset {dataset_name} in {dataset_path}")
    new_repo = os.path.join(dataset_path, dataset_name)
    subprocess.run(["git", "clone", f"https://huggingface.co/datasets/{dataset_name}", new_repo])
# check if model is installed
def is_model_installed(model_name, model_path=MODELS_PATH):
    return os.path.exists(os.path.join(model_path, model_name))
    
# create model directory
def create_model_dir(model_name, model_path=MODELS_PATH):
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

def get_model_info(model_name="distilbert-base-uncased"):
    api = HfApi()
    model = hf_api.model_info(model_name)
    model = api.model_info(model_name)
    model_info = {
        "id": model.id,
        "author": model.author,
        "tags": model.tags,
        "pipeline_tag": model.pipeline_tag,
        "transformers_info": model.transformers_info,
    }
    return json.dumps(model_info, indent=4)

def uninstall_model(model_name, model_path=MODELS_PATH):
    local_dir = os.path.join(model_path, model_name)
    try:
        shutil.rmtree(local_dir)
        print(f"Model {model_name} uninstalled")
    except Exception as e:
        print(f"Unable to uninstall model {model_name}\n\n{e}")


def get_installed_models(models_path=MODELS_PATH):
    installed_models = []
    authors = os.listdir(models_path)
    for author in authors:
        models = os.listdir(os.path.join(models_path, author))
        for model in models:
            installed_models.append(os.path.join(author, model))
    return installed_models
