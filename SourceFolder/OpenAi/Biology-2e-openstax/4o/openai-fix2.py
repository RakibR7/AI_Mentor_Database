# fixed_for_openai.py

import json

input_file = 'cleaned_for_finetuning_FIXED.jsonl'
output_file = 'openai_finetune_ready.jsonl'

with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
    for line in infile:
        item = json.loads(line)
        prompt = item["question"].strip()
        completion = item["answer"].strip()

        formatted_prompt = prompt + "\n\n###\n\n"
        formatted_completion = " " + completion.strip() + "###"

        outfile.write(json.dumps({
            "prompt": formatted_prompt,
            "completion": formatted_completion
        }) + '\n')

print(f"✅ Rewritten with separator: {output_file}")
