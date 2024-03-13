import scipy
from transformers import AutoProcessor, MusicgenForConditionalGeneration, pipeline
import torch
import os
from utils import is_model_installed
import torch.nn.utils.parametrizations as WN
from rich.console import Console
from rich.spinner import Spinner

class Musicgen:
    def __init__(self, model_name="facebook/musicgen-large"):
        if not is_model_installed(model_name):
            raise ValueError(f"Model {model_name} not installed")
        self.model_name = model_name
        self.processor = None
        self.model = None
        self.model_loaded = False
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.audio_values = None
        self.console = Console()

    def synthesize(self, prompt="lo-fi music with a soothing melody", file_path="./generated_audio", file_name="audio.wav"):
        synthesiser = pipeline("text-to-audio", "facebook/musicgen-large", device=self.device, max_new_tokens=512, padding=True, return_tensors="pt")

        music = synthesiser(prompt, forward_params={"do_sample": True})
        file = os.path.join(file_path, file_name)
        scipy.io.wavfile.write(file, rate=music["sampling_rate"], data=music["audio"])


    def load_model(self):
        try:
            with self.console.status("[bold green]Loading model...", spinner="dots") as status:
                status.update("[bold green]Loading processor...")
                self.processor = AutoProcessor.from_pretrained(self.model_name)
                status.update("[bold green]Loading model...")
                self.model = MusicgenForConditionalGeneration.from_pretrained(self.model_name).to(self.device)
                self.model_loaded = True
                status.update("[bold green]Model loaded successfully")
        except Exception as e:
            print(f"Unable to load model from {self.model_name}\n\n{e}")

    def generate_audio(self, prompt=["lo-fi music with a soothing melody"]):
        with self.console.status("[bold green]Generating inputs...", spinner="dots") as status:
            inputs = self.processor(
                text=prompt,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            status.update("[bold green]Generating audio...")
            self.audio_values = self.model.generate(**inputs, guidance_scale=3, max_new_tokens=256)
        return self.audio_values
    
    def save_audio(self, file_path, file_name):
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        with self.console.status("[bold green]Saving audio...", spinner="dots") as status:
            status.update(f"[bold green]Saving audio to {file_path}/{file_name}")
            sampling_rate = self.model.config.audio_encoder.sampling_rate
            file = os.path.join(file_path, file_name)
            # Move the tensor to CPU before converting to numpy
            audio_data = self.audio_values[0, 0].cpu().numpy()
            scipy.io.wavfile.write(file, rate=sampling_rate, data=audio_data)




