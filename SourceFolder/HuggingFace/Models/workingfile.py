from transformers import AutoTokenizer, AutoModelForCausalLM

# Your Hugging Face API key
api_key = "huggingface-api-key"

# Load the tokenizer and model with the API key
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B", token=api_key)
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B", token=api_key)

# Set the pad token to eos token if not already set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Input text and tokenization
input_text = "Once upon a time"
inputs = tokenizer(input_text, return_tensors="pt", padding=True)

# Generate text using the model
outputs = model.generate(
    inputs.input_ids,
    attention_mask=inputs.attention_mask,
    max_length=50,
    pad_token_id=tokenizer.eos_token_id
)

# Decode the generated text
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
