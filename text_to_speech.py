from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan, pipeline
from datasets import load_dataset
import torch
import torch.nn as nn
import soundfile as sf
import os
from rich import print
from rich.console import Console
from utils import is_model_installed, MODELS_PATH

class TextToSpeech:
    def __init__(self, model_id="microsoft/speecht5_tts", vocoder_id="Matthijs/cmu-arctic-xvectors"):
        self.console = Console()
        with self.console.status("[bold green]Initialising...", spinner="dots") as status:
            if not is_model_installed(model_id):
                raise Exception(f"Model {model_id} is not downloaded. Please download the model first.")
            #if not is_model_installed(vocoder_id):
            #    raise Exception(f"Vocoder {vocoder_id} is not downloaded. Please download the vocoder first.")
            self.model_id = model_id
            self.model_path = self.build_model_path(model_id)
            self.vocoder_id = vocoder_id
            self.vocoder_path = self.build_model_path(vocoder_id)
            self.processor = None
            self.model = None
            self.vocoder = None
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def synthesise(self, text):
        with self.console.status("[bold green]Creating pipeline...", spinner="dots") as status:
            synthesiser = pipeline("text-to-speech", self.model_path, device=self.device)
            
            status.update("[bold green]Loading speaker embeddings...")
            embeddings_dataset = load_dataset(self.vocoder_path, split="validation", trust_remote_code=True)

            speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
            # You can replace this embedding with your own as well.
            status.update("[bold green]Synthesising speech...")
            speech = synthesiser(text, forward_params={"speaker_embeddings": speaker_embedding})
            return speech
            

    def load_model(self):
        with self.console.status("[bold green]Loading processor...", spinner="dots") as status:
            self.processor = SpeechT5Processor.from_pretrained(self.model_path)

            status.update("[bold green]Loading model...")
            self.model = SpeechT5ForTextToSpeech.from_pretrained(self.model_path).to(self.device)

            status.update("[bold green]Loading vocoder...")
            self.vocoder = SpeechT5HifiGan.from_pretrained(self.vocoder_path).to(self.device)

            print("[bold green]Model loaded successfully!")

    def generate_speech(self, text, speaker_id):
        with self.console.status("[bold green]Generating inputs...", spinner="dots") as status:
            inputs = self.processor(text=text, return_tensors="pt").to(self.device)

            status.update("[bold green]speaker embeddings...")
            embeddings_dataset = load_dataset(self.vocoder_path, split="validation")
            speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

            status.update("[bold green]Generating speech...")
            speech = self.model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=self.vocoder).to(self.device)
            return speech
    
    def save_speech(self, speech, file_path, file_name):
        with self.console.status("[bold green]Saving speech...", spinner="dots") as status:
            if not os.path.exists(file_path):
                status.update(f"[bold green]Creating directory {file_path}...")
                os.makedirs(file_path)
            path = os.path.join(file_path, file_name)
            sf.write(path, speech["audio"], samplerate=speech["sampling_rate"])

    def build_model_path(self, model_name):
        return os.path.join(MODELS_PATH, model_name)

