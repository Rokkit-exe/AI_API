from transformers import pipeline

task = "fill-mask"
model_name = "bert-base-multilingual-uncased"
mask = "[MASK]"

prompt = f"The quick brown {mask} jumps over the lazy dog."

unmasker = pipeline(task=task, model=model_name)
sequences = unmasker(prompt)

for sequence in sequences:
    print(sequence['score'])
    print(sequence['token'])
    print(sequence['token_str'])
    print(sequence['sequence'])
    print("\n")