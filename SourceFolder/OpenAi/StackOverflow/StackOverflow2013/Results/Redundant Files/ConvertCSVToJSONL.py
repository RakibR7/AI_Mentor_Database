import csv
import json
from bs4 import BeautifulSoup

def clean_html(html_str):
    if not html_str:
        return ""
    soup = BeautifulSoup(html_str, "html.parser")
    return soup.get_text()

input_csv = "python_QA.csv"
output_jsonl = "fine_tune_data.jsonl"

with open(input_csv, "r", encoding="utf-8") as csv_file, \
     open(output_jsonl, "w", encoding="utf-8") as out_f:

    reader = csv.DictReader(csv_file)
    for row in reader:
        # Extract columns
        q_title = clean_html(row["QuestionTitle"])
        q_body  = clean_html(row["QuestionBody"])
        tags    = row["Tags"] or ""
        a_body  = clean_html(row["AnswerBody"])

        # Construct the prompt
        prompt_text = f"Q: {q_title}\n{q_body}\nTags: {tags}\n\n"

        # Construct the completion
        completion_text = f"A: {a_body}"

        # Build final dict
        record = {
            "prompt": prompt_text,
            "completion": completion_text
        }
        # Write as one JSON object per line
        out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

print("Created fine_tune_data.jsonl from CSV!")

