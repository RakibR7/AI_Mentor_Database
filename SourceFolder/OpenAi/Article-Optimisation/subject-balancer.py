import json
import random
from collections import defaultdict
import os
from tqdm import tqdm
import logging
from typing import Dict, List, Set
import re
import numpy as np
from itertools import combinations

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('subject_balance.log')
    ]
)

class SubjectBalancer:
    def __init__(self):
        self.target_per_subject = 2000
        self.min_examples = 400
        
    def augment_examples(self, examples: List[Dict]) -> List[Dict]:
        """
        Augment examples using various techniques to increase dataset size.
        """
        augmented = []
        base_examples = examples.copy()
        
        # 1. Combine related QA pairs
        if len(examples) >= 2:
            for ex1, ex2 in combinations(examples, 2):
                combined = {
                    "messages": [
                        {"role": "system", "content": "You are a knowledgeable AI mentor helping users learn about various topics."},
                        {"role": "user", "content": ex1['messages'][1]['content']},
                        {"role": "assistant", "content": ex1['messages'][2]['content']},
                        {"role": "user", "content": ex2['messages'][1]['content']},
                        {"role": "assistant", "content": ex2['messages'][2]['content']}
                    ]
                }
                augmented.append(combined)

        for example in examples:
            content = example['messages'][2]['content']
            if len(content.split('.')) > 3:
                sentences = content.split('.')
                mid = len(sentences) // 2
                
                variation1 = {
                    "messages": [
                        {"role": "system", "content": "You are a knowledgeable AI mentor helping users learn about various topics."},
                        {"role": "user", "content": f"Can you explain the first part of: {example['messages'][1]['content']}"},
                        {"role": "assistant", "content": '.'.join(sentences[:mid]) + '.'}
                    ]
                }
                
                variation2 = {
                    "messages": [
                        {"role": "system", "content": "You are a knowledgeable AI mentor helping users learn about various topics."},
                        {"role": "user", "content": f"Can you continue explaining: {example['messages'][1]['content']}"},
                        {"role": "assistant", "content": '.'.join(sentences[mid:]) + '.'}
                    ]
                }
                
                augmented.extend([variation1, variation2])

        for example in examples:
            summary = {
                "messages": [
                    {"role": "system", "content": "You are a knowledgeable AI mentor helping users learn about various topics."},
                    {"role": "user", "content": f"Can you summarize: {example['messages'][1]['content']}"},
                    {"role": "assistant", "content": example['messages'][2]['content'][:500] + "..."}
                ]
            }
            augmented.append(summary)

        all_examples = base_examples + augmented
        while len(all_examples) < self.target_per_subject:
            if not augmented:
                break
            
            new_variations = []
            for example in augmented[:100]:
                variation = {
                    "messages": [
                        {"role": "system", "content": "You are a knowledgeable AI mentor helping users learn about various topics."},
                        {"role": "user", "content": f"Please explain differently: {example['messages'][1]['content']}"},
                        {"role": "assistant", "content": example['messages'][2]['content']}
                    ]
                }
                new_variations.append(variation)
            
            all_examples.extend(new_variations)
            if len(new_variations) == 0:
                break
        
        return all_examples

    def balance_dataset(self, subject_data: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
        """
        Balance the dataset across all subjects.
        """
        balanced_data = {}

        logging.info("\nInitial distribution:")
        for subject, examples in subject_data.items():
            logging.info(f"{subject}: {len(examples)} examples")
        
        # Handle each subject
        for subject, examples in subject_data.items():
            if len(examples) > self.target_per_subject:

                balanced_data[subject] = random.sample(examples, self.target_per_subject)
                logging.info(f"\n{subject}: Downsampled from {len(examples)} to {self.target_per_subject}")
            
            elif len(examples) < self.target_per_subject and len(examples) >= self.min_examples:

                augmented = self.augment_examples(examples)

                if len(augmented) > self.target_per_subject:
                    balanced_data[subject] = random.sample(augmented, self.target_per_subject)
                else:
                    balanced_data[subject] = augmented
                logging.info(f"\n{subject}: Augmented from {len(examples)} to {len(balanced_data[subject])}")
            
            else:

                balanced_data[subject] = examples
                logging.info(f"\n{subject}: Insufficient examples ({len(examples)}), keeping all without augmentation")
        
        return balanced_data

    def save_datasets(self, balanced_data: Dict[str, List[Dict]], output_dir: str):
        """Save balanced datasets to files."""
        os.makedirs(output_dir, exist_ok=True)
        

        for subject, examples in balanced_data.items():
            output_file = os.path.join(output_dir, f"{subject}_balanced.jsonl")
            with open(output_file, 'w', encoding='utf-8') as f:
                for example in examples:
                    f.write(json.dumps(example) + '\n')
            

            logging.info(f"\nSaved {subject} dataset:")
            logging.info(f"- File: {output_file}")
            logging.info(f"- Examples: {len(examples)}")

            if examples:
                sample = random.choice(examples)
                logging.info("\nSample conversation:")
                for msg in sample['messages']:
                    logging.info(f"{msg['role']}: {msg['content'][:100]}...")

def main():
    input_file = "training_data/subject_split_data.jsonl"
    output_dir = "training_data/balanced_subjects"

    subject_data = defaultdict(list)
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading data"):
            example = json.loads(line)
            subject = example.get('subject', 'other')
            subject_data[subject].append(example)

    balancer = SubjectBalancer()
    balanced_data = balancer.balance_dataset(subject_data)
    balancer.save_datasets(balanced_data, output_dir)

    logging.info("\nFinal dataset statistics:")
    for subject, examples in balanced_data.items():
        logging.info(f"{subject}: {len(examples)} examples")
    
    logging.info("\nRecommendations for training:")
    logging.info("1. Start with mathematics and history models first")
    logging.info("2. For subjects with < 1000 examples, consider:")
    logging.info("   - Using transfer learning from math/history models")
    logging.info("   - Reducing model complexity")
    logging.info("   - Increasing epochs for better learning from limited data")

if __name__ == "__main__":
    main()
