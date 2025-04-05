import re
import json

def is_incomplete(text):
    """Check if answer is too short or incomplete"""
    return (
        not text or
        len(text.strip()) < 10 or
        text.strip().lower() in {'the', 'it', 'yes', 'no', 'maybe'} or
        text.strip().endswith(':')
    )

def extract_mcq_choices(chapter_text, question_num):
    """
    Extract multiple choice options from chapter text for a given question number.
    Returns string like "A. ..., B. ..., C. ..., D. ..."
    """
    pattern = re.compile(rf"{question_num}\.\s+.*?\n((?:[A-D]\.\s+.*?\n)+)", re.DOTALL)
    match = pattern.search(chapter_text)
    if not match:
        return ""

    options_block = match.group(1)
    # Clean formatting
    options = re.findall(r'([A-D])\.\s+(.*?)\n', options_block)
    return '\n'.join([f"{opt}. {desc.strip()}" for opt, desc in options])

def postprocess_matches(matches_path, cleaned_book_path):
    with open(matches_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    with open(cleaned_book_path, 'r', encoding='utf-8') as f:
        book_text = f.read()

    cleaned = []
    flagged = []
    removed_log = []

    for line in lines:
        try:
            entry = json.loads(line)
            chapter = entry['chapter']
            qnum = entry['question_num']
            question = entry['content'].strip()
            answer = entry['answer'].strip()

            # Remove very short or broken answers
            if is_incomplete(answer):
                flagged.append(entry)
                removed_log.append(f"[{chapter}-{qnum}] Incomplete answer: {answer}")
                continue

            # Handle multi-question answers packed into one
            multi_answers = re.findall(r'(\d+):?\s+([A-D]|\w.+?)(?:[;\n]|$)', answer)
            if len(multi_answers) > 1:
                for q_str, ans_part in multi_answers:
                    qid = int(q_str)
                    cleaned.append({
                        "question": f"CH{chapter} Q{qid}: {question}",
                        "answer": ans_part.strip()
                    })
                continue

            # Inject MCQ choices if answer is just a letter
            if re.fullmatch(r'[A-D]', answer):
                chapter_text_match = re.search(rf'CHAPTER\s+{chapter}.*?(?=CHAPTER\s+\d+|$)', book_text, re.DOTALL | re.IGNORECASE)
                if chapter_text_match:
                    mcq_block = extract_mcq_choices(chapter_text_match.group(0), qnum)
                    if mcq_block:
                        question = f"{question}\n{mcq_block}"

            cleaned.append({
                "question": question,
                "answer": answer
            })

        except Exception as e:
            removed_log.append(f"[ERROR] {e} on line: {line[:100]}")

    # Write final output
    with open('cleaned_for_finetuning.jsonl', 'w', encoding='utf-8') as f:
        for obj in cleaned:
            f.write(json.dumps(obj) + '\n')

    with open('flagged_bad_entries.jsonl', 'w', encoding='utf-8') as f:
        for obj in flagged:
            f.write(json.dumps(obj) + '\n')

    with open('removed_entries_log.txt', 'w', encoding='utf-8') as f:
        for reason in removed_log:
            f.write(reason + '\n')

    print(f"✅ Cleaned: {len(cleaned)} entries")
    print(f"⚠️ Flagged: {len(flagged)} entries")
    print(f"🗑️ Removed logs written to removed_entries_log.txt")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python postprocess_for_finetuning.py question_matches.jsonl Biology2e_cleaned.txt")
        sys.exit(1)
    postprocess_matches(sys.argv[1], sys.argv[2])
