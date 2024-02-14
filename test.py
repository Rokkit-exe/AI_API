from utils import get_model_info
import torch
# get model task
#model_task = get_model_task("distilbert-base-uncased")

# set device
#device = set_device()
#print(device)

#print(is_pipeline_supported("summarization"))

#print(get_models_from_pipeline("summarization"))

if __name__ == "__main__":
    model_info = get_model_info("cognitivecomputations/dolphin-2.6-mistral-7b")
    print(model_info)

