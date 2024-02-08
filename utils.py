from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


def install_model(model_name, model_path="./models"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    print(f"Model {model_name} installed at {model_path}")


def load_model(model_path="./models"):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    return model, tokenizer