import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from utils import install_model

model_id = "models/stabilityai/stable-diffusion-2-1"

# Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")
pipe.enable_attention_slicing()

prompt = "A logo representing an otter's face."
image = pipe(prompt).images[0]
    
image.save("generated_images/image.png")