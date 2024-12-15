import pandas as pd
import json

# Load the CSV file
csv_file = 'CSV_V4.csv'  # Path to your CSV file
output_file = 'CSV_V4.jsonl'  # Output JSONL file

# Read the CSV file into a DataFrame
df = pd.read_csv(csv_file)

# Open the output file for writing in JSONL format
with open(output_file, 'w') as f:
    # Iterate through each row of the DataFrame
    for _, row in df.iterrows():
        # Construct the "prompt" and "completion" fields
        prompt = row['QuestionTitle'] + '\n' + row['QuestionBody']
        completion = row['AnswerBody']

        # Create a dictionary for the JSONL format
        jsonl_record = {"prompt": prompt, "completion": completion}

        # Write the record as a JSON string to the output file
        f.write(json.dumps(jsonl_record) + '\n')

print(f"JSONL file saved to {output_file}")

