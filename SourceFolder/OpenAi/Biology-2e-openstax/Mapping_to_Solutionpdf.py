import re
import json

def parse_cleaned_textbook(cleaned_file):
    """
    Parse the cleaned textbook file, extracting (chapter_number, question_number, question_text).

    Example lines in the cleaned textbook might look like:
      "CHAPTER 1"
      "Q: 4. The first forms of life on Earth were ________."
      (possibly more lines until the next Q: or CHAPTER line)
    We'll store these in a dictionary keyed by (chapter_num, question_num).
    """

    results = {}  # {(chapter_num, question_num): question_text}

    current_chapter = None
    current_q_number = None
    current_q_text = []

    chapter_pattern = re.compile(r'^CHAPTER\s+(\d+)', re.IGNORECASE)
    # For question lines, we assume something like "Q: 4." or "Q: 6"
    question_pattern = re.compile(r'^Q:\s*(\d+)\.?', re.IGNORECASE)

    with open(cleaned_file, 'r', encoding='utf-8') as f:
        for line in f:
            line_stripped = line.strip()
            if not line_stripped:
                continue

            # Check for "CHAPTER X"
            ch_match = chapter_pattern.match(line_stripped)
            if ch_match:
                # If we were building a question, finalize it before resetting
                if current_chapter and current_q_number and current_q_text:
                    key = (current_chapter, current_q_number)
                    results[key] = "\n".join(current_q_text)

                # Reset state for the new chapter
                current_chapter = int(ch_match.group(1))
                current_q_number = None
                current_q_text = []
                continue

            # Check for question line "Q: X..."
            q_match = question_pattern.match(line_stripped)
            if q_match:
                # If we had a prior question, store it
                if current_q_number and current_q_text:
                    key = (current_chapter, current_q_number)
                    results[key] = "\n".join(current_q_text)

                # Start a new question
                current_q_number = int(q_match.group(1))
                current_q_text = [line_stripped]
            else:
                # If we are currently inside a question, accumulate the lines
                if current_q_number:
                    current_q_text.append(line_stripped)

    # Edge case: store the last question at EOF
    if current_chapter and current_q_number and current_q_text:
        key = (current_chapter, current_q_number)
        results[key] = "\n".join(current_q_text)

    # Debug prints
    print(f"\n[DEBUG] parse_cleaned_textbook: Found {len(results)} questions in {cleaned_file}")
    for k, v in results.items():
        print("[TEXTBOOK]", k, "->", repr(v[:60]) + ("..." if len(v) > 60 else ""))
    return results


def parse_solution_manual(solution_file):
    """
    Parse the solutions text, extracting (chapter_number, question_number, answer_text).
    This version handles multiple question references on one line, e.g.:
      "4 B 6 D 8 C 10 C 12 B 14 D 16 Answers will vary..."
    We'll repeatedly match patterns like (\\d+)\\s+([A-Za-z])?\\s*([^0-9]+)? so each question
    number is recorded separately.
    """

    solutions = {}  # {(chapter_num, question_num): answer_text}

    chapter_pattern = re.compile(r'^Chapter\s+(\d+)', re.IGNORECASE)
    multi_pattern = re.compile(r'(\d+)\s+([A-Za-z])?\s*([^0-9]+)?')

    current_chapter = None

    with open(solution_file, 'r', encoding='utf-8') as f:
        for line in f:
            line_stripped = line.strip()
            if not line_stripped:
                continue

            # 1) Check for "Chapter X"
            ch_match = chapter_pattern.match(line_stripped)
            if ch_match:
                current_chapter = int(ch_match.group(1))
                continue

            # 2) Repeatedly extract question references
            cursor = 0
            text_len = len(line_stripped)

            while cursor < text_len:
                match = multi_pattern.search(line_stripped, cursor)
                if not match:
                    break

                q_num_str = match.group(1)        # e.g. "4"
                letter_part = match.group(2) or ""# e.g. "B"
                remainder = match.group(3) or ""  # e.g. "Answers will vary..."

                # Convert question number
                question_num = int(q_num_str)

                # Build an answer string
                answer_text = (letter_part + " " + remainder).strip()

                if current_chapter:
                    key = (current_chapter, question_num)
                    if key not in solutions:
                        solutions[key] = answer_text
                    else:
                        solutions[key] += " " + answer_text

                cursor = match.end()

    # Debug prints
    print(f"\n[DEBUG] parse_solution_manual: Found {len(solutions)} solutions in {solution_file}")
    for k, v in solutions.items():
        print("[SOLUTIONS]", k, "->", repr(v[:60]) + ("..." if len(v) > 60 else ""))
    return solutions


def create_qna_jsonl(textbook_map, solution_map, output_file):
    """
    Combines question texts from 'textbook_map' with answers from 'solution_map'
    into a .jsonl file with the format:
      {"prompt": "Q: question_here\n\n###\n\n", "completion": " A: answer_here"}

    Only includes pairs that have both a question and a solution.
    """

    count = 0
    with open(output_file, 'w', encoding='utf-8') as out:
        for (ch, qnum), question_text in textbook_map.items():
            if (ch, qnum) in solution_map:
                answer_text = solution_map[(ch, qnum)]
                record = {
                    "prompt": f"Q: {question_text}\n\n###\n\n",
                    "completion": f" A: {answer_text}"
                }
                out.write(json.dumps(record) + "\n")
                count += 1

    print(f"\n[DEBUG] create_qna_jsonl: Created {count} Q&A pairs in {output_file}")


def main():
    cleaned_txt = "biology_textbook.txt"   # Your cleaned textbook file
    solutions_txt = "Biology2e-Solution.txt" # Your solutions file extracted from the PDF
    output_jsonl = "final_bio_qa.jsonl"

    # 1) Parse the textbook
    textbook_map = parse_cleaned_textbook(cleaned_txt)
    # 2) Parse the solutions
    solution_map = parse_solution_manual(solutions_txt)
    # 3) Build .jsonl
    create_qna_jsonl(textbook_map, solution_map, output_jsonl)


if __name__ == "__main__":
    main()
