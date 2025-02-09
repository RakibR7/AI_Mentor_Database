
#doesnt work

import json
from bs4 import BeautifulSoup

def clean_html(html_str):
    if not html_str:
        return ""
    soup = BeautifulSoup(html_str, "html.parser")
    return soup.get_text()

input_json = "JsonFormat_Python_V1.json"
output_jsonl = "fine_tune_data.jsonl"

with open(input_json, "r", encoding="utf-8") as f:
    data = json.load(f)  # data is a list of dicts

with open(output_jsonl, "w", encoding="utf-8") as out_f:
    for item in data:
        # Example: item["QuestionId"], item["QuestionTitle"], etc.
        q_title = clean_html(item.get("QuestionTitle", ""))
        q_body  = clean_html(item.get("QuestionBody", ""))
        tags    = item.get("Tags", "")
        a_body  = clean_html(item.get("AnswerBody", ""))

        prompt_text = f"Q: {q_title}\n{q_body}\nTags: {tags}\n\n"
        completion_text = f"A: {a_body}"

        record = {
            "prompt": prompt_text,
            "completion": completion_text
        }
        out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

print("Created fine_tune_data.jsonl from JSON array!")
