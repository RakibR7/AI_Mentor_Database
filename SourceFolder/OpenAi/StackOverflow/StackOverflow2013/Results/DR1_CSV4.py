import csv
import json
from bs4 import BeautifulSoup

def clean_html(html_str):
    if not html_str or not html_str.strip():
        return ""
    try:
        soup = BeautifulSoup(html_str, "html.parser")
        return soup.get_text()
    except Exception as e:
        print(f"Warning: Failed to parse HTML — {e}")
        return html_str  # Return raw text as fallback

input_csv = "DR1_CSV4.csv"
output_jsonl = "DR1_CSV4.jsonl"

with open(input_csv, "r", encoding="utf-8") as f_in, open(output_jsonl, "w", encoding="utf-8") as f_out:
    reader = csv.reader(f_in)
    next(reader)  # Skip header if already added
    for row in reader:
        # Access fields by index (adjust indices based on your CSV structure)
        q_title = clean_html(row[1])
        q_body  = clean_html(row[2])
        tags    = row[3]
        a_body  = clean_html(row[5])

        prompt_text = f"Q: {q_title}\n{q_body}\nTags: {tags}\n\n"
        completion_text = f"A: {a_body}"

        record = {
            "prompt": prompt_text,
            "completion": completion_text
        }
        f_out.write(json.dumps(record, ensure_ascii=False) + "\n")