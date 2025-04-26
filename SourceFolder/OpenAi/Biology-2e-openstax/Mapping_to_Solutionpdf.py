import re
import json

def parse_cleaned_textbook(cleaned_file):
    results = {}

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

            ch_match = chapter_pattern.match(line_stripped)
            if ch_match:

                if current_chapter and current_q_number and current_q_text:
                    key = (current_chapter, current_q_number)
                    results[key] = "\n".join(current_q_text)

                current_chapter = int(ch_match.group(1))
                current_q_number = None
                current_q_text = []
                continue

            q_match = question_pattern.match(line_stripped)
            if q_match:
                if current_q_number and current_q_text:
                    key = (current_chapter, current_q_number)
                    results[key] = "\n".join(current_q_text)

                current_q_number = int(q_match.group(1))
                current_q_text = [line_stripped]
            else:
                if current_q_number:
                    current_q_text.append(line_stripped)


    if current_chapter and current_q_number and current_q_text:
        key = (current_chapter, current_q_number)
        results[key] = "\n".join(current_q_text)

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

    solutions = {}

    chapter_pattern = re.compile(r'^Chapter\s+(\d+)', re.IGNORECASE)
    multi_pattern = re.compile(r'(\d+)\s+([A-Za-z])?\s*([^0-9]+)?')

    current_chapter = None

    with open(solution_file, 'r', encoding='utf-8') as f:
        for line in f:
            line_stripped = line.strip()
            if not line_stripped:
                continue

            ch_match = chapter_pattern.match(line_stripped)
            if ch_match:
                current_chapter = int(ch_match.group(1))
                continue

            cursor = 0
            text_len = len(line_stripped)

            while cursor < text_len:
                match = multi_pattern.search(line_stripped, cursor)
                if not match:
                    break

                q_num_str = match.group(1)
                letter_part = match.group(2) or ""
                remainder = match.group(3) or ""
                question_num = int(q_num_str)
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
    cleaned_txt = "biology_textbook.txt"
    solutions_txt = "Biology2e-Solution.txt"
    output_jsonl = "final_bio_qa.jsonl"


    textbook_map = parse_cleaned_textbook(cleaned_txt)
    solution_map = parse_solution_manual(solutions_txt)
    create_qna_jsonl(textbook_map, solution_map, output_jsonl)

if __name__ == "__main__":
    main()
