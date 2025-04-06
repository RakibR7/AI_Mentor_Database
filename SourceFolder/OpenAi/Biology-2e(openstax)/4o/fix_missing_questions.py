# fix_missing_questions.py

import json

input_file = 'question_matches_MANUAL_HANDLING.jsonl'
output_file = 'cleaned_for_finetuning_FIXED.jsonl'

fixed_entries = []

with open(input_file, 'r', encoding='utf-8') as infile:
    for line in infile:
        obj = json.loads(line)
        question = obj.get("question", "").strip()

        # 🔧 Restore question from 'content' if missing
        if not question and "content" in obj:
            question = obj["content"].strip()

        # ✅ Add cleaned question/answer
        fixed_entries.append({
            "question": question,
            "answer": obj["answer"].strip()
        })

# Save fixed entries
with open(output_file, 'w', encoding='utf-8') as outfile:
    for entry in fixed_entries:
        json.dump(entry, outfile)
        outfile.write('\n')

print(f"✅ Fixed {len(fixed_entries)} entries -> {output_file}")
