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

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('biology_extraction.log')
    ]
)

class BiologyContentExtractor:
    def __init__(self):
        self.target_size = 3500

        self.biology_patterns = {
            'molecular_biology': [
                r'\bdna\b', r'\brna\b', r'\bprotein\b', r'\bgene\b', r'\bgenome\b',
                r'\btranscription\b', r'\btranslation\b', r'\bnucleic acid\b',
                r'\bchromosome\b', r'\bmutation\b', r'\bexpression\b', r'\bplasmid\b',
                r'\benzyme\b', r'\bpolymerase\b', r'\breplication\b'
            ],
            'cell_biology': [
                r'\bcell\b', r'\bmembrane\b', r'\borganelle\b', r'\bmitochondria\b',
                r'\bvacuole\b', r'\blysosome\b', r'\bendoplasmic\b', r'\bgolgi\b',
                r'\bcytoplasm\b', r'\bnucleus\b', r'\bribosome\b', r'\bvesicle\b',
                r'\bcytoskeleton\b', r'\bchloroplast\b'
            ],
            'anatomy_physiology': [
                r'\btissue\b', r'\borgan\b', r'\bmuscle\b', r'\bbone\b', r'\bblood\b',
                r'\bnerve\b', r'\bhormone\b', r'\bbrain\b', r'\bheart\b', r'\blung\b',
                r'\bdigestive\b', r'\brespiratory\b', r'\bcirculatory\b', r'\bimmune\b',
                r'\bendocrine\b', r'\breproductive\b'
            ],
            'genetics': [
                r'\ballele\b', r'\bgenotype\b', r'\bphenotype\b', r'\bheredity\b',
                r'\binheritance\b', r'\bhaploid\b', r'\bdiploid\b', r'\bdominant\b',
                r'\brecessive\b', r'\bpedigree\b', r'\blinkage\b', r'\bcrossing over\b',
                r'\bmeiosis\b', r'\bmitosis\b'
            ],
            'ecology': [
                r'\becosystem\b', r'\bhabitat\b', r'\bpopulation\b', r'\bcommunity\b',
                r'\bfood web\b', r'\bfood chain\b', r'\bbiome\b', r'\badaptation\b',
                r'\bsymbiosis\b', r'\bparasitism\b', r'\bmutualism\b', r'\bcompetition\b',
                r'\bpredation\b', r'\bbiodiversity\b'
            ],
            'evolution': [
                r'\bevolution\b', r'\bnatural selection\b', r'\bspeciation\b',
                r'\bphylogeny\b', r'\btaxonomy\b', r'\bspecies\b', r'\bfossil\b',
                r'\bhomolog\b', r'\banalog\b', r'\bconvergent\b', r'\bdivergent\b',
                r'\bcommon ancestor\b', r'\bextinction\b'
            ],
            'microbiology': [
                r'\bbacteria\b', r'\bvirus\b', r'\bfungi\b', r'\bprotozoa\b',
                r'\bpathogen\b', r'\bfermentation\b', r'\bantibiotic\b', r'\bculture\b',
                r'\bcolony\b', r'\binfection\b', r'\bimmunity\b', r'\bvaccine\b'
            ]
        }
        

        self.required_keywords = {
            'biology', 'biological', 'organism', 'cell', 'gene', 'species',
            'evolution', 'tissue', 'protein', 'dna', 'ecosystem'
        }
        

        self.exclusion_patterns = [
            r'\bchemical reaction\b(?!.*biological)',
            r'\bphysics\b(?!.*biological)',
            r'\bmathematical\b(?!.*biological)',
            r'\bcomputer\b(?!.*biological)',
            r'\bgeological\b(?!.*biological)'
        ]
        
        self.vectorizer = TfidfVectorizer(max_features=5000)

    def is_biology_content(self, text: str) -> Tuple[bool, float, Set[str]]:

        text = text.lower()

        if not any(keyword in text for keyword in self.required_keywords):
            return False, 0.0, set()

        for pattern in self.exclusion_patterns:
            if re.search(pattern, text):
                return False, 0.0, set()

        subfield_scores = {}
        matched_subfields = set()
        total_matches = 0
        
        for subfield, patterns in self.biology_patterns.items():
            matches = 0
            for pattern in patterns:
                count = len(re.findall(pattern, text))
                matches += count
                total_matches += count
            
            if matches > 0:
                matched_subfields.add(subfield)
                subfield_scores[subfield] = matches

        if not matched_subfields:
            return False, 0.0, set()

        subfield_coverage = len(matched_subfields) / len(self.biology_patterns)
        pattern_density = min(1.0, total_matches / 20)  # Cap at 20 matches
        
        confidence_score = (subfield_coverage * 0.6 + pattern_density * 0.4)
        
        return confidence_score > 0.3, confidence_score, matched_subfields

    def create_hybrid_example(self, examples: List[Dict]) -> Dict:
        """Create a new example by combining related biology content."""
        base = random.choice(examples)
        donor = random.choice(examples)
        
        new_messages = []
        for msg in base['messages']:
            if msg['role'] == 'assistant':
                base_content = msg['content']
                donor_content = random.choice([m['content'] for m in donor['messages'] if m['role'] == 'assistant'])
                
                base_sentences = [s.strip() + '.' for s in base_content.split('.') if s.strip()]
                donor_sentences = [s.strip() + '.' for s in donor_content.split('.') if s.strip()]

                if len(base_sentences) >= 2 and len(donor_sentences) >= 2:
                    new_content = (
                        ' '.join(base_sentences[:2]) +
                        ' ' + ' '.join(random.sample(donor_sentences, min(2, len(donor_sentences)))) +  # Supporting content
                        ' ' + base_sentences[-1]
                    )
                else:
                    new_content = base_content
                
                new_messages.append({"role": "assistant", "content": new_content})
            else:
                new_messages.append(msg)
        
        return {"messages": new_messages}

    def extract_and_balance_biology(self, input_file: str, output_file: str):
        biology_examples = []
        subfield_stats = defaultdict(int)
        confidence_scores = []
        
        logging.info("Extracting biology content...")
        
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f):
                example = json.loads(line)
                text = ' '.join([msg['content'] for msg in example['messages']])
                
                is_biology, confidence, subfields = self.is_biology_content(text)
                
                if is_biology:
                    biology_examples.append(example)
                    confidence_scores.append(confidence)
                    for subfield in subfields:
                        subfield_stats[subfield] += 1

        #Log statistics
        logging.info("\nInitial Biology Content Statistics:")
        logging.info(f"Total examples found: {len(biology_examples)}")
        logging.info(f"Average confidence score: {np.mean(confidence_scores):.2f}")
        logging.info("\nSubfield distribution:")
        for subfield, count in subfield_stats.items():
            logging.info(f"- {subfield}: {count}")
        
        # Balance if needed
        if len(biology_examples) < self.target_size:
            needed = self.target_size - len(biology_examples)
            logging.info(f"\nGenerating {needed} additional examples...")
            
            high_confidence_examples = [
                ex for ex, score in zip(biology_examples, confidence_scores)
                if score > 0.5
            ]
            
            for _ in tqdm(range(needed)):
                hybrid = self.create_hybrid_example(high_confidence_examples)
                biology_examples.append(hybrid)
        
        # Save balanced dataset
        random.shuffle(biology_examples)
        final_examples = biology_examples[:self.target_size]
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            for example in final_examples:
                f.write(json.dumps(example) + '\n')
        
        logging.info(f"\nFinal dataset created with {len(final_examples)} examples")
        logging.info(f"Output saved to: {output_file}")

def main():
    input_file = "../Wikipedia_dumps/training_data/gpt4_training_data.jsonl"
    output_file = "../Wikipedia_dumps/training_data/biology_training_data.jsonl"
    
    extractor = BiologyContentExtractor()
    extractor.extract_and_balance_biology(input_file, output_file)
    
    logging.info("\nBiology content extraction and balancing complete!")

if __name__ == "__main__":
    main()
