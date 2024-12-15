import os
from huggingface_hub import hf_hub_download

HUGGING_FACE_API_KEY = ("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")

model_id = "openai-community/gpt2"
filenames = [
        ".gitattributes", "64-8bits.tflite", "64-fp16.tflite", "64.tflite", 
        "config.json", "flax_model.msgpack", "generation_config.json", "merges.txt",
        "model.safetensors", "pytorch_model.bin", "rust_model.ot", "tf_model.h5",
        "tokenizer.json", "tokenizer_config.json", "vocab.json"
]

from transformers import GPT2Tokenizer, GPT2Model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2Model.from_pretrained('gpt2')
text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)

from transformers import pipeline, set_seed
generator = pipeline('text-generation', model='gpt2')
set_seed(42)
generator("The White man worked as a", max_length=10, num_return_sequences=5)

