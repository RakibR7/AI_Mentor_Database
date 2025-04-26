import os
import json
import re

# Set the base directory containing all the extracted Wikipedia folders
base_directory = 'C:/Users/Rakib/Documents/Ai Mentor/SourceFolder/OpenAi/Wikipedia_dumps/Wiki_extracted_data/AA'  # Adjust this path as needed
output_file = 'C:/Users/Rakib/Documents/Ai Mentor/SourceFolder/OpenAi/Wikipedia_dumps/WikiData_JSONL'

jsonl_data = []

for folder in os.listdir(base_directory):
    folder_path = os.path.join(base_directory, folder)
    if os.path.isdir(folder_path):

        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)

            if os.path.isfile(file_path):
                with open(file_path, 'r', encoding='utf-8') as file:

                    content = file.read()
                    articles = content.splitlines()

                    for article in articles:
                        try:
                            data = json.loads(article)

                            title = data.get('title', '')
                            text = data.get('text', '').strip()

                            if title and text and len(text) > 50:  # Ensure meaningful content
                                prompt = f"What can you tell me about {title}?"
                                completion = f" {text}"

                                jsonl_data.append({
                                    'prompt': prompt,
                                    'completion': completion
                                })

                        except json.JSONDecodeError:
                            continue

#Save all prompt-completion pairs to a single JSONL file
with open(output_file, 'w', encoding='utf-8') as outfile:
    for entry in jsonl_data:
        json.dump(entry, outfile)
        outfile.write('\n')

print(f"JSONL file saved to {output_file}")
