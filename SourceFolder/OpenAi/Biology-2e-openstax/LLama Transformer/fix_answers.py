import json
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from tqdm import tqdm

model_id = "TheBloke/Llama-2-13B-GPTQ"  # Update if using a different one

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    max_memory={
        0: "16GiB",
        "cpu": "16GiB"
    }
)

generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

def fix_answer(question, answer):
    prompt = f"Question: {question.strip()}\nAnswer (fix and complete it properly):"
    result = generator(prompt, max_new_tokens=200, return_full_text=False)[0]["generated_text"]
    return result.strip()

input_path = "cleaned_for_finetuning_FIXED.jsonl"
output_path = "cleaned_fixed_llama.jsonl"

with open(input_path, "r", encoding="utf-8") as infile:
    entries = [json.loads(line) for line in infile]

fixed_entries = []
for entry in tqdm(entries, desc="Fixing answers with LLaMA"):
    fixed_answer = fix_answer(entry["question"], entry["answer"])
    fixed_entries.append({"question": entry["question"], "answer": fixed_answer})

with open(output_path, "w", encoding="utf-8") as outfile:
    for item in fixed_entries:
        outfile.write(json.dumps(item) + "\n")

print(f"✅ All answers cleaned. Output saved to {output_path}")
