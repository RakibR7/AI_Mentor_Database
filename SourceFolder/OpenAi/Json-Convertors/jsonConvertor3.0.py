import os
import json
import re
import time
import logging
from multiprocessing import Pool

#Set up logging for progress tracking
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

#Set the base directory containing all the extracted Wikipedia folders
base_directory = 'C:/Users/Rakib/Documents/Ai Mentor/SourceFolder/OpenAi/Wikipedia_dumps/Wiki_extracted_data/AA'
output_file = 'C:/Users/Rakib/Documents/Ai Mentor/SourceFolder/OpenAi/Wikipedia_dumps/WikiData_JSONL'


#Function to process each folder
def process_folder(folder):
    folder_path = os.path.join(base_directory, folder)
    jsonl_entries = []
    if os.path.isdir(folder_path):
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    articles = content.splitlines()
                    for article in articles:
                        try:
                            #Parse each line as a JSON object
                            data = json.loads(article)

                            #Extract title and text
                            title = data.get('title', '')
                            text = data.get('text', '').strip()

                            #Create a prompt-completion pair
                            if title and text and len(text) > 50:  # Ensure meaningful content
                                prompt = f"What can you tell me about {title}?"
                                completion = f" {text}"

                                jsonl_entries.append({'prompt': prompt, 'completion': completion})

                        except json.JSONDecodeError:
                            #Skip lines that are not valid JSON
                            continue
    return jsonl_entries


if __name__ == "__main__":
    #Start processing all folders
    start_time = time.time()

    #Use multiprocessing to speed up folder processing
    all_entries = []
    with Pool(processes=4) as pool:  # Adjust number of processes based on your CPU
        results = pool.map(process_folder, os.listdir(base_directory))
        for result in results:
            all_entries.extend(result)

    #Write all entries to the JSONL output file
    try:
        with open(output_file, 'w', encoding='utf-8') as outfile:
            for entry in all_entries:
                json.dump(entry, outfile)
                outfile.write('\n')
    except PermissionError:
        logging.error(f"Permission denied: Unable to write to {output_file}. Please check the file permissions.")

    end_time = time.time()
    logging.info(f"Total time taken: {(end_time - start_time) / 3600:.2f} hours")
    logging.info(f"JSONL file saved to {output_file}")
