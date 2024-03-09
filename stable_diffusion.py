import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from utils import install_model, is_model_installed, build_model_path
import gc

class SD_Pipeline:
    def __init__(self):
        self.model_id = None
        self.pipeline = None
        self.image = None

    def load_model(self, model_id):
        if not is_model_installed(model_id):
            raise Exception("Model not installed")
        self.model_id = build_model_path(model_id)
        self.pipeline = StableDiffusionPipeline.from_pretrained(self.model_id, torch_dtype=torch.float16)
        self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(self.pipeline.scheduler.config)
        self.pipeline = self.pipeline.to("cuda")
        self.pipeline.enable_attention_slicing()

    def unload_model(self):
        print("Unloading model...")
        del self.pipeline
        gc.collect()
        self.pipeline = None
        print("Model unloaded successfully!")

    def generate_image(self, prompt):
        self.image = self.pipeline(prompt).images[0]
        return self.image

    def save_image(self):
        self.image.save("generated_images/image.png")