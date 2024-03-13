import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from utils import install_model, is_model_installed, MODELS_PATH, build_model_path
import os
from datasets import load_dataset

class SpeechToText:
    def __init__(self, model_id="openai/whisper-tiny"):
        if not is_model_installed(model_id):
            raise Exception(f"Model {model_id} not installed")
        self.model_path = build_model_path(model_id)
        self.model_id = model_id
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model = None
        self.processor = None
        self.pipe = None

    def load_model(self, max_new_tokens=128, chunk_length_s=30, batch_size=16):
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_path, torch_dtype=self.torch_dtype, low_cpu_mem_usage=True, use_safetensors=True, attn_implementation="flash_attention_2"
        ).to(self.device)

        self.processor = AutoProcessor.from_pretrained(self.model_id)

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            max_new_tokens=max_new_tokens,
            chunk_length_s=chunk_length_s,
            batch_size=batch_size,
            return_timestamps=True,
            torch_dtype=self.torch_dtype,
            device=self.device,
        )

    def transcribe(self, sample, language="english", translate=False):
        task = "translation" if translate else "transcribe"
        result = self.pipe(sample, generate_kwargs={"language": language, "task": task})
        return result