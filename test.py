from utils import get_model_task, set_device, install_model, load_model, is_model_installed, create_model_dir, MODEL_BASE_PATH, is_pipeline_supported, get_models_from_pipeline

# get model task
#model_task = get_model_task("distilbert-base-uncased")

# set device
#device = set_device()
#print(device)

print(is_pipeline_supported("summarization"))

print(get_models_from_pipeline("summarization"))

