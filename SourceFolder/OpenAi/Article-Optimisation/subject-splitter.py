import json
import tiktoken
from tqdm import tqdm
import logging
from typing import List, Dict, Set
import random
import os
import re

#GOLD

#
#C:\Users\Rakib\AppData\Local\Programs\Python\Python310\python.exe "C:\Users\Rakib\Documents\Ai Mentor\SourceFolder\OpenAi\Wikipedia_dumps\subject-splitter.py"
#2024-10-27 17:49:26,173 - INFO - Processing C:\Users\Rakib\Documents\Ai Mentor\SourceFolder\OpenAi\Wikipedia_dumps\training_data/gpt4_training_data.jsonl...
#37850it [00:08, 4317.04it/s]
#2024-10-27 17:49:34,951 - INFO -
#=== Processing Statistics ===
#2024-10-27 17:49:34,951 - INFO - Total processed: 27422
#2024-10-27 17:49:34,951 - INFO - Skipped: 0
#2024-10-27 17:49:34,952 - INFO -
#Subject Distribution:
#2024-10-27 17:49:34,952 - INFO - - mathematics: 693 articles
#2024-10-27 17:49:34,952 - INFO - - physics: 2575 articles
#2024-10-27 17:49:34,952 - INFO - - computer_science: 657 articles
#2024-10-27 17:49:34,952 - INFO - - biology: 1984 articles
#2024-10-27 17:49:34,952 - INFO - - chemistry: 2517 articles
#2024-10-27 17:49:34,952 - INFO - - history: 10000 articles
#

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('subject_split.log')
    ]
)

class SubjectSplitter:
    def __init__(self):
        self.encoding = tiktoken.encoding_for_model("gpt-4")

        self.MAX_TOKENS = 4096
        self.MIN_TOKENS = 10

        self.subjects = {
            "mathematics": {
                "keywords": [
                    "mathematics", "algebra", "geometry", "calculus", "statistics",
                    "probability", "theorem", "mathematical", "equation", "function",
                    "number theory", "linear algebra", "arithmetic"
                ],
                "count": 0
            },
            "physics": {
                "keywords": [
                    "physics", "quantum", "mechanics", "relativity", "thermodynamics",
                    "particle physics", "electromagnetic", "gravity", "force", "energy",
                    "motion", "dynamics", "kinematics"
                ],
                "count": 0
            },
            "computer_science": {
                "keywords": [
                    "computer science", "programming", "algorithm", "data structure",
                    "software", "coding", "computation", "database", "network",
                    "artificial intelligence", "machine learning", "computer architecture"
                ],
                "count": 0
            },
            "biology": {
                "keywords": [
                    "biology", "cell", "genetics", "evolution", "organism",
                    "molecular biology", "ecology", "physiology", "anatomy",
                    "biochemistry", "species", "DNA", "protein"
                ],
                "count": 0
            },
            "chemistry": {
                "keywords": [
                    "chemistry", "chemical", "molecule", "atom", "reaction",
                    "organic chemistry", "inorganic", "biochemistry", "element",
                    "compound", "acid", "base", "solution"
                ],
                "count": 0
            },
            "history": {
                "keywords": [
                    "history", "historical", "civilization", "empire", "war",
                    "revolution", "ancient", "medieval", "modern history",
                    "dynasty", "kingdom", "century", "era"
                ],
                "count": 0
            }
        }

    def determine_subject(self, text: str, title: str) -> str:
        """Determine the subject of an article based on content and title."""
        combined_text = (title + " " + text).lower()

        subject_scores = {}
        for subject, data in self.subjects.items():
            score = sum(1 for keyword in data["keywords"] 
                       if keyword.lower() in combined_text)
            subject_scores[subject] = score

        max_score = max(subject_scores.values())
        if max_score > 0:
            return max(subject_scores.items(), key=lambda x: x[1])[0]
        
        return "other"

    def validate_conversation(self, conversation: Dict) -> bool:
        """Validate conversation format and token count."""
        try:
            if 'messages' not in conversation:
                return False
                
            messages = conversation['messages']
            if not all(isinstance(m, dict) and 'role' in m and 'content' in m 
                      for m in messages):
                return False

            total_tokens = sum(len(self.encoding.encode(m['content'])) 
                             for m in messages)
            
            return self.MIN_TOKENS <= total_tokens <= self.MAX_TOKENS
            
        except Exception:
            return False

    def process_file(self, input_file: str, output_dir: str, max_per_subject: int = None) -> None:
        """Process and split conversations by subject."""
        if not os.path.exists(input_file):
            logging.error(f"Input file not found: {input_file}")
            return

        os.makedirs(output_dir, exist_ok=True)

        subject_files = {
            subject: open(os.path.join(output_dir, f"{subject}_dataset.jsonl"), 'w', 
                        encoding='utf-8')
            for subject in self.subjects.keys()
        }
        subject_files["other"] = open(os.path.join(output_dir, "other_dataset.jsonl"), 
                                    'w', encoding='utf-8')
        
        processed = 0
        skipped = 0
        
        logging.info(f"Processing {input_file}...")
        
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                for line in tqdm(f):
                    try:
                        conversation = json.loads(line)
                        
                        if not self.validate_conversation(conversation):
                            skipped += 1
                            continue

                        user_message = next((m for m in conversation['messages'] 
                                           if m['role'] == 'user'), None)
                        if not user_message:
                            continue

                        title_match = re.search(r"What is (.*?)\?", user_message['content'])
                        title = title_match.group(1) if title_match else ""

                        assistant_message = next((m for m in conversation['messages'] 
                                                if m['role'] == 'assistant'), None)
                        content = assistant_message['content'] if assistant_message else ""

                        subject = self.determine_subject(content, title)

                        if subject != "other" and max_per_subject:
                            if self.subjects[subject]["count"] >= max_per_subject:
                                continue
                            self.subjects[subject]["count"] += 1

                        subject_files[subject].write(json.dumps(conversation) + '\n')
                        processed += 1
                        
                    except json.JSONDecodeError:
                        skipped += 1
                        continue
                        
        finally:
            for f in subject_files.values():
                f.close()

        logging.info("\n=== Processing Statistics ===")
        logging.info(f"Total processed: {processed}")
        logging.info(f"Skipped: {skipped}")
        logging.info("\nSubject Distribution:")
        for subject, data in self.subjects.items():
            logging.info(f"- {subject}: {data['count']} articles")

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(script_dir, "training_data/gpt4_training_data.jsonl")
    output_dir = os.path.join(script_dir, "training_data/subject_datasets")
    splitter = SubjectSplitter()

    splitter.process_file(
        input_file=input_file,
        output_dir=output_dir,
        max_per_subject=10000
    )

if __name__ == "__main__":
    main()
