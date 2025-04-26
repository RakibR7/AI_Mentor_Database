import json

with open("openai_finetune_ready_prepared.jsonl", "r", encoding="utf-8") as f:
    for i, line in enumerate(f, start=1):
        obj = json.loads(line)
        print(f"\n--- Entry {i} ---")
        print("Prompt (question):", obj['prompt'].replace('\n\n###\n\n', '').strip())
        print("Completion (answer):", obj['completion'].replace('###', '').strip())
