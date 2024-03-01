from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import is_model_installed, install_model, MODELS_PATH, get_model_info, install_model
import torch
import os
from rich.console import Console
from rich import print

class LLM:
    def __init__(self, model_name="cognitivecomputations/dolphin-2.6-mistral-7b"):
        print(f"[bold green]Initialysing model: {model_name}")
        self.model_name = model_name
        self.model_path = self.build_model_path(model_name)
        if not is_model_installed(self.model_name):
            raise ValueError(f"Model {model_name} not installed")
        self.tokenizer = None
        self.model = None
        self.model_loaded = False
        self.chat_history = [{"role": "system", "content": "You are a usefull assistant!"}]
        self.console = Console()
        #self.model_info = get_model_info(self.model_path)
        if not torch.cuda.is_available():
            # raise error
            raise ValueError("No GPU found")
        else:
            self.device = torch.device("cuda:0")
    
    def generate(self, prompt):
        with self.console.status("[bold green]Generating inputs...", spinner="dots") as status:
            self.chat_history.append({"role": "user", "content": prompt})
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt"
            ).to(self.device)

            status.update("[bold green]Generating response...")
            generated_ids = self.model.generate(
                **inputs, 
                max_new_tokens=3000, 
                do_sample=True, 
                pad_token_id=self.tokenizer.eos_token_id, 
                eos_token_id=self.tokenizer.eos_token_id
            )
            status.update("[bold green]Decoding response...")
            response = self.tokenizer.batch_decode(generated_ids)[0]
            
            self.chat_history.append({"role": "assistant", "content": response})
            return response
    
    def load_model(self):
        with self.console.status("[bold green]Loading model...", spinner="dots") as status:
            try:
                self.model = AutoModelForCausalLM.from_pretrained(self.model_path, torch_dtype=torch.float16, attn_implementation="flash_attention_2").to(self.device)
                status.update("[bold green]Loading tokenizer...")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                self.model_loaded = True
                print("[bold green]\nModel loaded successfully\n\n")
            except Exception as e:
                print(f"Unable to load model from {self.model_path}\n\n{e}")

    def print_chat_history(self):
        for message in self.chat_history:
            if message["role"] == "user":
                print(f"[bold blue]{message['role']}: {message['content']}")
            elif message["role"] == "assistant":
                print(f"[bold green]{message['role']}: {message['content']}")

    def print_last_user_message(self):
        for message in reversed(self.chat_history):
            if message["role"] == "user":
                print(f"[bold blue]User: {message['content']}\n\n")
                break

    def print_last_assistant_message(self):
        for message in reversed(self.chat_history):
            if message["role"] == "assistant":
                print(f"[bold green]Assistant: {message['content']}\n\n")
                break

    def chat(self):
        prompt = None
        while prompt != "exit":
            prompt = self.console.input("[bold cyan]talk to me or type 'exit' to quit: ")
            if prompt == "exit":
                break
            self.generate(prompt)
            self.print_last_user_message()
            self.print_last_assistant_message()

    def build_model_path(self, model_name):
        return os.path.join(MODELS_PATH, model_name)

