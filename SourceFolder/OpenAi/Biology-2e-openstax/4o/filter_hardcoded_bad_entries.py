import json
import re

REMOVE_PAIRS = {
    (3, 33), (4, 29), (4, 43), (5, 4), (9, 33), (11, 29), (12, 24),
    (14, 4), (16, 29), (19, 1), (19, 2), (21, 1), (22, 32), (22, 42),
    (23, 31), (25, 26), (25, 30), (25, 32), (33, 33), (35, 26),
    (35, 32), (35, 36), (39, 22), (39, 100), (41, 24), (41, 26),
    (43, 35), (46, 1), (47, 19)
}

def extract_chapter_qnum(question_text):
    """Extract (chapter, question_num) from question like 'CH3 Q33: ...'"""
    match = re.match(r'CH(\d+)\s+Q(\d+):', question_text)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None

def is_incomplete(text):
    return (
        not text or
        len(text.strip()) < 10 or
        text.strip().lower() in {'the', 'it', 'yes', 'no', 'maybe'} or
        text.strip().endswith(':')
    )

def clean_and_filter(jsonl_file):
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    cleaned = []
    removed = []

    for item in data:
        qtext = item.get('question', '').strip()
        answer = item.get('answer', '').strip()

        key = extract_chapter_qnum(qtext)

        if key and key in REMOVE_PAIRS:
            removed.append((key, 'Hardcoded removal'))
            continue

        if is_incomplete(answer):
            removed.append((key or ('?', '?'), f"Incomplete answer: {answer[:40]}..."))
            continue

        cleaned.append({
            "question": qtext,
            "answer": answer
        })

    with open("cleaned_for_finetuning.jsonl", "w", encoding="utf-8") as f:
        for item in cleaned:
            f.write(json.dumps(item) + '\n')

    with open("removed_entries_log.txt", "w", encoding="utf-8") as f:
        for (ch, qn), reason in removed:
            f.write(f"[CH{ch} Q{qn}] {reason}\n")

    print(f"✅ Final cleaned entries: {len(cleaned)}")
    print(f"🗑️ Removed entries: {len(removed)} (see removed_entries_log.txt)")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python postprocess_and_clean_final.py <question_matches_MANUAL_HANDLING.jsonl>")
        sys.exit(1)

    clean_and_filter(sys.argv[1])
