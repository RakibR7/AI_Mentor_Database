import json

input_file = 'cleaned_for_finetuning_FIXED.jsonl'
output_file = 'openai_finetune_ready.jsonl'

with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
    for line in infile:
        item = json.loads(line)
        prompt = item["question"].strip()
        completion = item["answer"].strip()

        # Make sure format is OpenAI-ready
        outfile.write(json.dumps({
            "prompt": prompt,
            "completion": f" {completion}###"
        }) + '\n')

print(f"✅ Rewritten {output_file} with OpenAI-ready format.")
