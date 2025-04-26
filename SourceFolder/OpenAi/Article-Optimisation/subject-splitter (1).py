import json
import random
from collections import defaultdict
import os
from tqdm import tqdm
import logging
from typing import Dict, List, Set
import re

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
        self.subject_patterns = {
            'mathematics': [
                r'\bmath\w*\b', r'\balgebra\b', r'\bgeometry\b', r'\bcalculus\b',
                r'\bstatistics\b', r'\btrigonometry\b', r'\bnumber theory\b'
            ],
            'physics': [
                r'\bphysics\b', r'\bmechanics\b', r'\bquantum\b', r'\belectricity\b',
                r'\bmagnetism\b', r'\bthermodynamics\b', r'\boptics\b'
            ],
            'computer_science': [
                r'\bcomputer science\b', r'\bprogramming\b', r'\balgorithm\b',
                r'\bdata structure\b', r'\bsoftware\b', r'\bhardware\b', r'\bcoding\b'
            ],
            'biology': [
                r'\bbiology\b', r'\becology\b', r'\bgenetics\b', r'\bevolution\b',
                r'\bcell\b', r'\borganism\b', r'\banatomy\b'
            ],
            'chemistry': [
                r'\bchemistry\b', r'\bchemical\b', r'\bmolecule\b', r'\belement\b',
                r'\breaction\b', r'\bcompound\b', r'\batomic\b'
            ],
            'history': [
                r'\bhistory\b', r'\bhistorical\b', r'\bcentury\b', r'\bempire\b',
                r'\bcivilization\b', r'\bwar\b', r'\bmonarchy\b'
            ]
        }
        

        self.target_examples = 2000
        
    def determine_subject(self, text: str) -> str:
        """Determine the subject of an article based on keyword patterns."""
        text = text.lower()
        scores = defaultdict(int)
        
        for subject, patterns in self.subject_patterns.items():
            for pattern in patterns:
                matches = len(re.findall(pattern, text))
                scores[subject] += matches
        
        if not scores:
            return 'other'
        
        return max(scores.items(), key=lambda x: x[1])[0]

    def split_and_balance_data(self, input_file: str, output_dir: str):
        """Split and balance data by subject."""
        subject_data = defaultdict(list)

        logging.info(f"Reading and categorizing data from {input_file}")
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f):
                conversation = json.loads(line)
                text = ' '.join(msg['content'] for msg in conversation['messages'])
                subject = self.determine_subject(text)
                subject_data[subject].append(conversation)

        logging.info("\nInitial distribution:")
        for subject, data in subject_data.items():
            logging.info(f"{subject}: {len(data)} examples")

        balanced_data = {}
        for subject, data in subject_data.items():
            if subject == 'other':
                continue

            if len(data) > self.target_examples:
                balanced_data[subject] = random.sample(data, self.target_examples)
            else:
                balanced_data[subject] = data

        os.makedirs(output_dir, exist_ok=True)
        
        for subject, data in balanced_data.items():
            output_file = os.path.join(output_dir, f"{subject}_training.jsonl")
            with open(output_file, 'w', encoding='utf-8') as f:
                for conversation in data:
                    f.write(json.dumps(conversation) + '\n')
            
            logging.info(f"\nCreated {subject} dataset:")
            logging.info(f"- File: {output_file}")
            logging.info(f"- Examples: {len(data)}")

            sample_size = min(3, len(data))
            samples = random.sample(data, sample_size)
            logging.info(f"\nSample {subject} conversations:")
            for i, sample in enumerate(samples, 1):
                logging.info(f"\nSample {i}:")
                for msg in sample['messages']:
                    logging.info(f"{msg['role']}: {msg['content'][:100]}...")

def main():
    input_file = "training_data/Filtered/gpt4_training_data.jsonl"
    output_dir = "training_data/subjects"

    splitter = SubjectSplitter()
    splitter.split_and_balance_data(input_file, output_dir)
    
    logging.info("\nSubject splitting complete!")
    logging.info("Upload these files to OpenAI dashboard for fine-tuning specialized models")
    logging.info("\nRecommended fine-tuning settings for each subject:")
    logging.info("- Model: GPT-4 Mini")
    logging.info("- Learning rate: 1e-5")
    logging.info("- Batch size: 4")
    logging.info("- Epochs: 3")
    logging.info("- Max tokens: 2048")

if __name__ == "__main__":
    main()
