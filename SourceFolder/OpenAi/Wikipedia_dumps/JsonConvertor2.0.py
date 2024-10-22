import os
import json
import re

# Set the base directory containing all the extracted Wikipedia folders
base_directory = 'C:/Users/Rakib/Documents/Ai Mentor/SourceFolder/OpenAi/Wikipedia_dumps/Wiki_extracted_data/AA'  # Adjust this path as needed
output_file = 'C:/Users/Rakib/Documents/Ai Mentor/SourceFolder/OpenAi/Wikipedia_dumps/WikiData_JSONL'

jsonl_data = []

# Loop through all folders in the base directory
for folder in os.listdir(base_directory):
    folder_path = os.path.join(base_directory, folder)

    # Check if it is a directory
    if os.path.isdir(folder_path):

        # Loop through all files in the current folder
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)

            # Check if it is a file
            if os.path.isfile(file_path):
                with open(file_path, 'r', encoding='utf-8') as file:
                    # Read content of the file
                    content = file.read()

                    # Split the content into individual JSON objects (one per line)
                    articles = content.splitlines()

                    # Process each article
                    for article in articles:
                        try:
                            # Parse each line as a JSON object
                            data = json.loads(article)

                            # Extract title and text
                            title = data.get('title', '')
                            text = data.get('text', '').strip()

                            # Create a prompt-completion pair
                            if title and text and len(text) > 50:  # Ensure meaningful content
                                prompt = f"What can you tell me about {title}?"
                                completion = f" {text}"

                                # Add the data to the JSONL list
                                jsonl_data.append({
                                    'prompt': prompt,
                                    'completion': completion
                                })

                        except json.JSONDecodeError:
                            # Skip lines that are not valid JSON
                            continue

# Save all prompt-completion pairs to a single JSONL file
with open(output_file, 'w', encoding='utf-8') as outfile:
    for entry in jsonl_data:
        json.dump(entry, outfile)
        outfile.write('\n')

print(f"JSONL file saved to {output_file}")
