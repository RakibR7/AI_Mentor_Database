import os
import json
import requests
import time

OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"
MODEL = "llama2"
CHAPTER_TEXT_DIR = "chapters_cleaned"
SOLUTIONS_FILE = "../Biology2e-Solution.txt"
OUTPUT_FILE = "generated_qna_llama.jsonl"

def load_solutions_by_chapter(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()

    solutions = {}
    current_chapter = None
    buffer = []

    for line in text.splitlines():
        line = line.strip()
        if line.startswith("Chapter") and any(char.isdigit() for char in line):
            if current_chapter and buffer:
                solutions[current_chapter] = "\n".join(buffer).strip()
                buffer = []
            current_chapter = int(line.split()[1])
        elif current_chapter:
            buffer.append(line)

    if current_chapter and buffer:
        solutions[current_chapter] = "\n".join(buffer).strip()

    return solutions

def generate_qa(context, solution):
    prompt = f"""
You are a helpful biology tutor. Based on the textbook content and the solution summary provided, generate 3 clear, standalone biology questions and answers.

Textbook Content:
{context}

Solution Hints:
{solution}

Only return valid JSONL entries in the format:
{{"question": "...", "answer": "..."}}
Return only JSONL lines, no explanation or extra formatting.
"""

    try:
        response = requests.post(OLLAMA_ENDPOINT, json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False
        })
        return response.json().get("response", "")
    except Exception as e:
        print("❌ Error calling LLaMA:", e)
        return ""

def main():
    solutions = load_solutions_by_chapter(SOLUTIONS_FILE)

    output = []
    for filename in sorted(os.listdir(CHAPTER_TEXT_DIR)):
        if not filename.endswith(".txt"):
            continue

        chapter_num = int(filename.split("_")[1].split(".")[0])
        with open(os.path.join(CHAPTER_TEXT_DIR, filename), 'r', encoding='utf-8') as f:
            context = f.read()

        solution_text = solutions.get(chapter_num, "")
        if not solution_text.strip():
            print(f"⚠️ Skipping Chapter {chapter_num}, no solution found.")
            continue

        print(f"🧠 Generating Q&A for Chapter {chapter_num}...")
        response = generate_qa(context, solution_text)

        for line in response.strip().splitlines():
            try:
                obj = json.loads(line)
                output.append(obj)
            except json.JSONDecodeError:
                continue

        time.sleep(1)

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for item in output:
            f.write(json.dumps(item) + '\n')

    print(f"\n✅ Generated {len(output)} Q&A pairs. Saved to: {OUTPUT_FILE}")

if __name__ == '__main__':
    main()