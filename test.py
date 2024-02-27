from utils import install_model

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


install_model("cognitivecomputations/TinyDolphin-2.8.2-1.1b-laser")
install_model("cognitivecomputations/dolphin-2.6-mistral-7b")
