from transformers import AutoTokenizer, AutoModelForCausalLM

api_key = "hugging-face-api"

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B", token=api_key)
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B", token=api_key)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

input_text = "Once upon a time"
inputs = tokenizer(input_text, return_tensors="pt", padding=True)

outputs = model.generate(
    inputs.input_ids,
    attention_mask=inputs.attention_mask,
    max_length=50,
    pad_token_id=tokenizer.eos_token_id
)

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
