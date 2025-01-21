import pandas as pd
import json

# Load the CSV file
df = pd.read_csv('CSV_V3.csv')

# Convert each row to a dictionary and save as JSONL
with open('stackoverflow_python_data.jsonl', 'w') as f:
    for index, row in df.iterrows():
        prompt = row['QuestionTitle'] + '\n' + row['QuestionBody']
        completion = row['AnswerBody']
        f.write(json.dumps({"prompt": prompt, "completion": completion}) + '\n')
