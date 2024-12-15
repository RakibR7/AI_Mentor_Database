import pandas as pd
import json

# Load the CSV file
csv_file = 'CSV_V4.csv'
df = pd.read_csv(csv_file, header=None)  # Load without assuming headers

# Debugging: Print the first few rows to understand the structure
print("Data Preview:")
print(df.head())

# Rename columns to match the structure if no headers are present
df.columns = [
    "QuestionId", "QuestionTitle", "QuestionBody", "Tags",
    "AnswerId", "AnswerBody", "AnswerScore"
]

# Define the output file
output_file = 'CSV_V4.jsonl'

# Convert rows to JSONL format
with open(output_file, 'w') as f:
    for _, row in df.iterrows():
        try:
            # Adjust fields to match the renamed column names
            prompt = row["QuestionTitle"] + "\n" + row["QuestionBody"]
            completion = row["AnswerBody"]
            f.write(json.dumps({"prompt": prompt, "completion": completion}) + "\n")
        except KeyError as e:
            print(f"Missing column: {e}")
            continue

print(f"JSONL file saved to {output_file}")