import json
import random
from collections import defaultdict
import os
from tqdm import tqdm
import logging
from typing import Dict, List, Set, Tuple
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('advanced_balancing.log')
    ]
)

class SimpleSubjectBalancer:
    def __init__(self):
        self.target_size = 3500
        self.min_similarity = 0.7
        self.subject_patterns = {
            'mathematics': {
                'patterns': [
                    r'\bmath\w*\b', r'\balgebra\b', r'\bgeometry\b', r'\bcalculus\b',
                    r'\bstatistics\b', r'\bprobability\b', r'\btrigonometry\b',
                    r'\bnumber theory\b', r'\btopology\b', r'\blinear algebra\b',
                    r'\boptimization\b', r'\banalysis\b'
                ],
                'keywords': {'theorem', 'proof', 'equation', 'formula', 'calculation'}
            },
            'physics': {
                'patterns': [
                    r'\bphysics\b', r'\bmechanics\b', r'\bquantum\b', r'\belectricity\b',
                    r'\bmagnetism\b', r'\bthermodynamics\b', r'\boptics\b', r'\brelativity\b',
                    r'\bparticle\b', r'\bwave\b', r'\bforce\b', r'\benergy\b'
                ],
                'keywords': {'experiment', 'observation', 'theory', 'measurement', 'force'}
            },
            'computer_science': {
                'patterns': [
                    r'\bcomputer science\b', r'\bprogramming\b', r'\balgorithm\b',
                    r'\bdata structure\b', r'\bsoftware\b', r'\bhardware\b', r'\bcoding\b',
                    r'\bdatabase\b', r'\bnetwork\b', r'\bai\b', r'\bmachine learning\b'
                ],
                'keywords': {'program', 'code', 'system', 'data', 'implementation'}
            },
            'biology': {
                'patterns': [
                    r'\bbiology\b', r'\becology\b', r'\bgenetics\b', r'\bevolution\b',
                    r'\bcell\b', r'\borganism\b', r'\banatomy\b', r'\bphysiology\b',
                    r'\bdna\b', r'\bprotein\b', r'\bspecies\b'
                ],
                'keywords': {'species', 'organism', 'cell', 'tissue', 'evolution'}
            },
            'chemistry': {
                'patterns': [
                    r'\bchemistry\b', r'\bchemical\b', r'\bmolecule\b', r'\belement\b',
                    r'\breaction\b', r'\bcompound\b', r'\batomic\b', r'\borganic\b',
                    r'\binorganic\b', r'\bpolymer\b'
                ],
                'keywords': {'reaction', 'compound', 'molecule', 'element', 'bond'}
            },
            'history': {
                'patterns': [
                    r'\bhistory\b', r'\bhistorical\b', r'\bcentury\b', r'\bempire\b',
                    r'\bcivilization\b', r'\bwar\b', r'\bmonarchy\b', r'\bancient\b',
                    r'\bmedieval\b', r'\bmodern\b'
                ],
                'keywords': {'event', 'period', 'era', 'dynasty', 'revolution'}
            }
        }
        
        self.vectorizer = TfidfVectorizer(max_features=5000)

    def analyze_text_complexity(self, text: str) -> float:
        """Simple text complexity analysis."""
        words = text.split()
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        avg_word_length = np.mean([len(w) for w in words]) if words else 0
        avg_sentence_length = len(words) / len(sentences) if sentences else 0
        unique_words = len(set(words))
        
        return (avg_word_length * 0.3 + avg_sentence_length * 0.4 + unique_words * 0.3) / 100

    def create_hybrid_example(self, examples: List[Dict]) -> Dict:
        """Create a new example by combining elements from existing ones."""
        base = random.choice(examples)
        donor = random.choice(examples)
        
        new_messages = []
        for msg in base['messages']:
            if msg['role'] == 'assistant':

                base_content = msg['content']
                donor_content = random.choice([m['content'] for m in donor['messages'] if m['role'] == 'assistant'])
                
                base_sentences = [s.strip() for s in base_content.split('.') if s.strip()]
                donor_sentences = [s.strip() for s in donor_content.split('.') if s.strip()]
                

                selected_sentences = (base_sentences[:2] + 
                                   random.sample(donor_sentences, min(2, len(donor_sentences))))
                new_content = '. '.join(selected_sentences) + '.'
                
                new_messages.append({"role": "assistant", "content": new_content})
            else:
                new_messages.append(msg)
        
        return {"messages": new_messages}

    def determine_subject(self, text: str) -> str:
        """Determine the primary subject of a text."""
        text = text.lower()
        scores = defaultdict(int)
        
        for subject, data in self.subject_patterns.items():

            for pattern in data['patterns']:
                matches = len(re.findall(pattern, text))
                scores[subject] += matches * 2
            

            for keyword in data['keywords']:
                if keyword in text:
                    scores[subject] += 1
        
        return max(scores.items(), key=lambda x: x[1])[0] if scores else 'other'

    def balance_dataset(self, input_file: str, output_dir: str):
        """Main method to balance the dataset."""
        subject_data = defaultdict(list)
        
        logging.info("Reading and categorizing data...")
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f):
                example = json.loads(line)
                text = ' '.join([msg['content'] for msg in example['messages']])
                subject = self.determine_subject(text)
                if subject != 'other':
                    subject_data[subject].append(example)

        logging.info("\nInitial distribution:")
        for subject, examples in subject_data.items():
            logging.info(f"{subject}: {len(examples)} examples")

        balanced_data = defaultdict(list)
        
        for subject, examples in subject_data.items():
            logging.info(f"\nProcessing {subject}...")
            current_size = len(examples)
            balanced_data[subject].extend(examples)
            
            if current_size < self.target_size:
                needed = self.target_size - current_size
                logging.info(f"Creating {needed} new examples for {subject}")
                
                for _ in tqdm(range(needed)):
                    hybrid = self.create_hybrid_example(examples)
                    balanced_data[subject].append(hybrid)

        os.makedirs(output_dir, exist_ok=True)
        
        for subject, examples in balanced_data.items():
            output_file = os.path.join(output_dir, f"{subject}_balanced.jsonl")

            random.shuffle(examples)

            final_examples = examples[:self.target_size]
            
            with open(output_file, 'w', encoding='utf-8') as f:
                for example in final_examples:
                    f.write(json.dumps(example) + '\n')
            
            logging.info(f"\nCreated balanced dataset for {subject}:")
            logging.info(f"- File: {output_file}")
            logging.info(f"- Examples: {len(final_examples)}")

def main():
    input_file = "../Wikipedia_dumps/training_data/gpt4_training_data.jsonl"
    output_dir = "../backup/Old project/Backup_old/balanced_subjects"
    
    balancer = SimpleSubjectBalancer()
    balancer.balance_dataset(input_file, output_dir)
    
    logging.info("\nBalancing complete!")

if __name__ == "__main__":
    main()
