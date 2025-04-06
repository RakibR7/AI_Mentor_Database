import json
import re
import time
import requests
import concurrent.futures

OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"
MODEL = "llama2"
INPUT_FILE = "cleaned_for_finetuning_FIXED.jsonl"
OUTPUT_FILE = "cleaned_for_finetuning_FIXED_OLLAMAV2.jsonl"

# Customize the prompt
PROMPT_TEMPLATE = (
    "Rewrite the following biology question so that it is standalone, clear, complete, and makes sense without any figures or missing context. "
    "Then answer it with a correct and concise biology answer."
    "\n\nQuestion: {question}\n\nOutput in JSON format with 'question' and 'answer' keys."
)

# Call Ollama for a single question
def call_ollama(entry):
    question = entry.get("question", "").strip()
    payload = {
        "model": MODEL,
        "prompt": PROMPT_TEMPLATE.format(question=question),
        "stream": False
    }

    try:
        response = requests.post(OLLAMA_ENDPOINT, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json().get("response", "{}")

        # Attempt to extract JSON block
        match = re.search(r'\{.*\}', result, re.DOTALL)
        if match:
            parsed = json.loads(match.group(0))
            return {"question": parsed["question"].strip(), "answer": parsed["answer"].strip()}
        else:
            print("⚠️ JSON parse failed. Got:", result)
            return None

    except Exception as e:
        print(f"❌ Error processing question: {question[:60]}...\n{e}")
        return None

# Read input
with open(INPUT_FILE, 'r', encoding='utf-8') as f:
    raw_data = [json.loads(line) for line in f]

print(f"📚 Loaded {len(raw_data)} entries from {INPUT_FILE}")

# Process in parallel
cleaned_data = []
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
    futures = [executor.submit(call_ollama, entry) for entry in raw_data]
    for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
        result = future.result()
        if result:
            cleaned_data.append(result)
            print(f"✅ Processed {i}/{len(raw_data)}")
        else:
            print(f"⚠️ Skipped {i}/{len(raw_data)}")

# Write output
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    for entry in cleaned_data:
        f.write(json.dumps(entry) + '\n')

print(f"\n🎉 Saved cleaned file to {OUTPUT_FILE} ({len(cleaned_data)} entries)")
