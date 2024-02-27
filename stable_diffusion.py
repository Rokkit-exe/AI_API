import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from utils import install_model



# Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead

class SD_Pipeline:
    def __init__(self):
        self.model_id = "models/stabilityai/stable-diffusion-2-1"
        self.pipeline = None
        self.image = None

    def load_model(self):
        self.pipeline = StableDiffusionPipeline.from_pretrained(self.model_id, torch_dtype=torch.float16)
        self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(self.pipeline.scheduler.config)
        self.pipeline = self.pipeline.to("cuda")
        self.pipeline.enable_attention_slicing()

    def generate_image(self, prompt):
        self.image = self.pipeline(prompt).images[0]
        return self.image

    def save_image(self):
        self.image.save("generated_images/image.png")