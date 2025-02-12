import csv
import json
from bs4 import BeautifulSoup


def clean_html(html_str):
    if not html_str or not html_str.strip():
        return ""
    try:
        soup = BeautifulSoup(html_str, "html.parser")
        return soup.get_text().strip()
    except Exception as e:
        print(f"Warning: Failed to parse HTML — {e}")
        return html_str.strip()


input_csv = "DR1_CSV5_Shortened_20.csv"
output_jsonl = "DR1_CSV5_Shortened_20.csv.jsonl"

with open(input_csv, "r", encoding="utf-8") as f_in, open(output_jsonl, "w", encoding="utf-8") as f_out:
    reader = csv.reader(f_in)
    next(reader)  # Skip header

    for row in reader:
        # Clean and format fields (adjust indices as needed)
        q_title = clean_html(row[1])
        q_body = clean_html(row[2])
        tags = row[3].replace("<", "").replace(">", ", ")  # Clean tags
        a_body = clean_html(row[5])

        # Construct chat messages
        messages = [
            {
                "role": "system",
                "content": "You are an expert Python programming mentor trained on Stack Overflow data. Provide clear, concise answers to technical questions."
            },
            {
                "role": "user",
                "content": f"{q_title}\n\n{q_body}\n\nTags: {tags}"
            },
            {
                "role": "assistant",
                "content": a_body
            }
        ]

        record = {"messages": messages}
        f_out.write(json.dumps(record, ensure_ascii=False) + "\n")