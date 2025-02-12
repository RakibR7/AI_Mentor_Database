import csv
import json
from bs4 import BeautifulSoup

def clean_html(html_str: str) -> str:
    """Remove HTML tags from a string using BeautifulSoup."""
    if not html_str:
        return ""
    soup = BeautifulSoup(html_str, "html.parser")
    return soup.get_text()

# Input CSV and output JSONL filenames
input_csv = "csv_v3.csv"
output_jsonl = "csv_v3_TUNED.jsonl"

with open(input_csv, "r", encoding="utf-8") as f_in, \
     open(output_jsonl, "w", encoding="utf-8") as f_out:

    reader = csv.DictReader(f_in)
    for row in reader:
        # Extract fields (adjust these to match your actual CSV headers)
        q_title = clean_html(row.get("QuestionTitle", ""))
        q_body  = clean_html(row.get("QuestionBody", ""))
        tags    = row.get("Tags", "")
        a_body  = clean_html(row.get("AnswerBody", ""))

        # Build your prompt text
        # Typically, you show the question, some context, maybe tags
        prompt_text = f"Q: {q_title}\n{q_body}\nTags: {tags}\n\n"

        # Build your completion text
        # The answer or solution
        completion_text = f"A: {a_body}"

        # Create the final record
        record = {
            "prompt": prompt_text,
            "completion": completion_text
        }

        # Write one record (as JSON) per line
        f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"Done! Created {output_jsonl} from {input_csv}")
