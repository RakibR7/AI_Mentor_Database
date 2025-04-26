import re
import json
import os
import sys


def split_by_chapter(input_text):
    """
    Splits the entire text by 'CHAPTER X' headings (and optionally 'UNIT X').
    Returns a dict: { "chapter_1": "<chapter text>", "chapter_2": "...", ... }
    """
    # Regex to find lines that start with 'CHAPTER' or 'UNIT' followed by a space and a digit.
    # We'll treat each match as a new chunk.
    pattern = r'(?=^CHAPTER\s+\d|^UNIT\s+\d)'

    # Split the text on these chapter/unit headings.
    chunks = re.split(pattern, input_text, flags=re.MULTILINE)

    # The first chunk (chunks[0]) may be front matter, so we’ll keep it as “chapter_0” if it has content.
    chapters = {}
    for i, chunk in enumerate(chunks):
        chunk = chunk.strip()
        if not chunk:
            continue  # skip empty
        # Attempt to find a heading line for naming
        first_line = chunk.split('\n', 1)[0]
        # example: "CHAPTER 1\nThe Study of Life ..."
        # We'll form a label "chapter_1" or "unit_1" from that line.

        # Let’s do a quick match:
        heading_match = re.match(r'(CHAPTER|UNIT)\s+(\d+)', first_line, flags=re.IGNORECASE)
        if heading_match:
            label = f"{heading_match.group(1).lower()}_{heading_match.group(2)}"
        else:
            # If no heading found, label as 'chapter_X' with i
            label = f"chapter_{i}"

        chapters[label] = chunk
    return chapters


def extract_questions_from_chunk(chunk_text):
    """
    Within one chapter text, find 'Review Questions' or 'Critical Thinking Questions'.
    Gather lines after each heading as Q&As, until we reach another heading like 'Key Terms',
    'Chapter Summary', 'Visual Connection Questions', or blank line, or end of chunk.

    Returns a list of (question, answer) pairs.
    """
    lines = chunk_text.split('\n')

    # We'll search for lines that match these patterns:
    question_headers = ["review questions", "critical thinking questions"]
    stop_headers = [
        "key terms",
        "chapter summary",
        "visual connection questions",
        # add others if needed
    ]

    qna_pairs = []
    collecting = False
    current_header = None

    for i, line in enumerate(lines):
        stripped_line = line.strip().lower()
        if not stripped_line:
            continue  # skip empty lines

        # Check if this line is one of the question headers
        if any(h == stripped_line for h in question_headers):
            collecting = True
            current_header = line.strip()  # e.g. "Review Questions"
            continue

        # If we see a "stop header", or "CHAPTER" or "UNIT" or something, we stop collecting
        if any(h == stripped_line for h in stop_headers):
            collecting = False
            current_header = None
            continue

        # Also stop collecting if we encounter a new "CHAPTER" or "UNIT" line
        if re.match(r'^(CHAPTER|UNIT)\s+\d', line, flags=re.IGNORECASE):
            collecting = False
            current_header = None
            continue

        if collecting:
            # Each non-empty line is presumably a question (since there's no built-in answer text).
            question_text = line.strip()
            # You could guess an answer from chapter summary or leave blank
            # We'll do placeholders for now:
            answer_text = "No direct answer provided in text."

            # Add the pair
            qna_pairs.append((question_text, answer_text))

    return qna_pairs


def create_jsonl_from_qna(qna_pairs, output_path):
    """
    Takes a list of (question, answer) pairs. Writes them to a .jsonl file for fine-tuning.
    Format:
      { "prompt": "Q: <question>\n\n###\n\n", "completion": " A: <answer>" }
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        for (q, a) in qna_pairs:
            record = {
                "prompt": f"Q: {q}\n\n###\n\n",
                "completion": f" A: {a}"
            }
            f.write(json.dumps(record) + "\n")


def main(input_file, output_jsonl):
    # 1) Load the entire text
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()

    # 2) Split by chapters
    chapters = split_by_chapter(text)

    # 3) For each chapter chunk, extract Q&As
    all_qna_pairs = []
    for chapter_label, chunk_text in chapters.items():
        qna_pairs = extract_questions_from_chunk(chunk_text)
        all_qna_pairs.extend(qna_pairs)

    # 4) Write all Q&As to a single .jsonl
    create_jsonl_from_qna(all_qna_pairs, output_jsonl)
    print(f"Done! Extracted {len(all_qna_pairs)} Q&A pairs into {output_jsonl}.")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python split_and_create_qna.py <cleaned_input.txt> <output.jsonl>")
        sys.exit(1)

    input_txt = sys.argv[1]  # e.g. "biology_textbook.txt"
    output_jsonl = sys.argv[2]  # e.g. "biology_qna.jsonl"

    main(input_txt, output_jsonl)
