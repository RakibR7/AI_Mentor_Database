import json
from ollama import Client

client = Client(host='http://localhost:11434')

def fix_answer(question, answer):
    prompt = f"""You are an expert biology tutor. Improve the following answer for accuracy and completeness.

Question: {question}
Answer: {answer}

Improved Answer:"""

    response = client.chat(
        model='llama2:13b',
        messages=[{"role": "user", "content": prompt}]
    )
    return response['message']['content'].strip()

input_file = "cleaned_for_finetuning_FIXED.jsonl"
output_file = "cleaned_for_finetuning_OLLAMA.jsonl"

with open(input_file, 'r', encoding='utf-8') as f:
    data = [json.loads(line) for line in f]

fixed_data = []
for i, entry in enumerate(data, 1):
    q, a = entry['question'], entry['answer']
    try:
        fixed_answer = fix_answer(q, a)
        fixed_data.append({"question": q, "answer": fixed_answer})
        print(f"✅ Fixed {i}/{len(data)}")
    except Exception as e:
        print(f"❌ Error on entry {i}: {e}")

with open(output_file, 'w', encoding='utf-8') as f:
    for item in fixed_data:
        f.write(json.dumps(item) + '\n')

print("🎉 All answers fixed and written to cleaned_for_finetuning_FIXED.jsonl")
