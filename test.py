from utils import install_model, install_dataset
from llm import LLM
from speech_to_text import SpeechToText
from musicgen import Musicgen
from text_to_speech import TextToSpeech

if False:
    max_seq_length = 2048

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "models/mistra_test_save", # Supports Llama, Mistral - replace this!
        max_seq_length = max_seq_length,
        dtype = None,
        load_in_4bit = True,
    )

    tokenizer = get_chat_template(
        tokenizer,
        chat_template = "mistral", # Supports zephyr, chatml, mistral, llama, alpaca, vicuna, vicuna_old, unsloth
        mapping = {"role" : "from", "content" : "value", "user" : "human", "assistant" : "gpt"}, # ShareGPT style
        map_eos_token = True, # Maps <|im_end|> to </s> instead
    )

    FastLanguageModel.for_inference(model) # Enable native 2x faster inference

    messages = [
        {"from": "human", "value": "how to declare a function in python?"},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize = True,
        add_generation_prompt = True, # Must add for generation
        return_tensors = "pt",
    ).to("cuda")

    outputs = model.generate(input_ids = inputs, max_new_tokens = 64, use_cache = True)
    print(tokenizer.batch_decode(outputs))

# speech to text
if False:
    stt = SpeechToText()
    stt.load_model()
    print(stt.transcribe("./speech_sample/Recording.mp3"))

# llm
if False:
    llm = LLM()
    llm.load_model()
    llm.chat()

# musicgen
if False:
    musicgen = Musicgen()
    musicgen.load_model()
    audio = musicgen.generate_audio()
    musicgen.save_audio("./generated_audio", "audio.wav")

#install_dataset("Matthijs/cmu-arctic-xvectors")


text = "my mom is in bad shape because of her back pain. I am worried about her."
tts = TextToSpeech()
tts.load_model()
speech = tts.generate_speech(text, 0)
tts.save_speech(speech, "./generated_speech", "speech.wav")


