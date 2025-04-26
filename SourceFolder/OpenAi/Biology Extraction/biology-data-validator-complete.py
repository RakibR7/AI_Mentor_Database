
import json
import random
from collections import defaultdict
import os
from tqdm import tqdm
import logging
import re
from typing import Dict, List, Tuple, Set
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

#Validation Summary:
#Total examples processed: 3500
#Valid biology examples: 40
#Invalid examples: 3460
#Validation rate: 1.14%


class BiologyDataValidator:
    def __init__(self):
        self.biology_keywords = {
            'molecular_biology': {
                'primary': {'dna', 'rna', 'protein', 'gene', 'genome', 'chromosome'},
                'secondary': {'transcription', 'translation', 'enzyme', 'mutation'}
            },
            'cell_biology': {
                'primary': {'cell', 'membrane', 'nucleus', 'mitochondria', 'organelle'},
                'secondary': {'cytoplasm', 'ribosome', 'golgi', 'endoplasmic'}
            },
            'genetics': {
                'primary': {'allele', 'inheritance', 'genotype', 'phenotype'},
                'secondary': {'dominant', 'recessive', 'heredity', 'trait'}
            },
            'ecology': {
                'primary': {'ecosystem', 'species', 'population', 'habitat'},
                'secondary': {'adaptation', 'niche', 'biodiversity', 'community'}
            },
            'physiology': {
                'primary': {'organ', 'tissue', 'system', 'hormone'},
                'secondary': {'muscle', 'nerve', 'blood', 'brain'}
            }
        }
        
        self.non_biology_indicators = {
            'chemistry': {'chemical reaction', 'molecular weight', 'atomic number'},
            'physics': {'quantum', 'velocity', 'momentum', 'electromagnetic'},
            'math': {'equation', 'calculation', 'arithmetic', 'geometric'},
            'computer': {'programming', 'algorithm', 'database', 'software'}
        }

    def validate_file(self, input_file: str) -> Dict:
        if not os.path.exists(input_file):
            logging.error(f"Input file not found: {input_file}")
            return None

        logging.info(f"Starting validation of {input_file}")
        
        validation_results = {
            'total_examples': 0,
            'valid_examples': 0,
            'invalid_examples': 0,
            'subfield_distribution': defaultdict(int),
            'confidence_scores': [],
            'content_length_stats': [],
            'keyword_coverage': defaultdict(int),
            'potential_errors': [],
            'sample_examples': []
        }

        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(tqdm(f), 1):
                    try:
                        example = json.loads(line)
                        validation_results['total_examples'] += 1

                        is_valid, confidence, stats = self._validate_example(example, line_num)
                        
                        if is_valid:
                            validation_results['valid_examples'] += 1
                            validation_results['confidence_scores'].append(confidence)

                            for subfield in stats['subfields']:
                                validation_results['subfield_distribution'][subfield] += 1
                            
                            for keyword in stats['keywords']:
                                validation_results['keyword_coverage'][keyword] += 1

                            content = ' '.join([msg['content'] for msg in example['messages']])
                            validation_results['content_length_stats'].append(len(content))

                            if len(validation_results['sample_examples']) < 5 and random.random() < 0.1:
                                validation_results['sample_examples'].append(example)
                        else:
                            validation_results['invalid_examples'] += 1
                            if stats.get('error'):
                                validation_results['potential_errors'].append(stats['error'])
                            
                    except json.JSONDecodeError:
                        validation_results['invalid_examples'] += 1
                        validation_results['potential_errors'].append(f"JSON decode error at line {line_num}")
                        continue
                        
        except Exception as e:
            logging.error(f"Error processing file: {e}")
            return None

        return validation_results

    def _validate_example(self, example: Dict, line_num: int) -> Tuple[bool, float, Dict]:
        stats = {
            'subfields': set(),
            'keywords': set(),
            'error': None
        }
        
        try:
            text = ' '.join([msg['content'] for msg in example['messages']]).lower()
            
            #Check for non-biology content
            for field, indicators in self.non_biology_indicators.items():
                if any(indicator in text for indicator in indicators):
                    stats['error'] = f"Contains {field} content at line {line_num}"
                    return False, 0.0, stats
            
            #Calculate biology relevance
            confidence = 0
            total_keywords = 0
            
            for subfield, keywords in self.biology_keywords.items():
                primary_matches = sum(1 for keyword in keywords['primary'] if keyword in text)
                secondary_matches = sum(1 for keyword in keywords['secondary'] if keyword in text)
                
                if primary_matches + secondary_matches > 0:
                    stats['subfields'].add(subfield)
                    confidence += (primary_matches * 2 + secondary_matches) / (len(keywords['primary']) * 2 + len(keywords['secondary']))
                    
                total_keywords += primary_matches + secondary_matches
                stats['keywords'].update([k for k in keywords['primary'] if k in text])
                stats['keywords'].update([k for k in keywords['secondary'] if k in text])
            
            confidence = confidence / len(self.biology_keywords) if stats['subfields'] else 0
            
            #Validate example
            is_valid = (
                confidence > 0.2 and
                len(stats['subfields']) >= 1 and
                total_keywords >= 3
            )
            
            return is_valid, confidence, stats
            
        except Exception as e:
            stats['error'] = f"Error processing example at line {line_num}: {str(e)}"
            return False, 0.0, stats

    def _generate_visualizations(self, results: Dict, output_dir: str, timestamp: str):
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Subfield Distribution
        plt.figure(figsize=(12, 6))
        subfields = list(results['subfield_distribution'].keys())
        counts = list(results['subfield_distribution'].values())
        
        bars = plt.bar(range(len(subfields)), counts)
        plt.xticks(range(len(subfields)), subfields, rotation=45, ha='right')
        
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom')
        
        plt.title('Distribution of Biology Subfields')
        plt.xlabel('Subfields')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'subfield_distribution_{timestamp}.png'))
        plt.close()
        
        #Confidence Score Distribution
        if results['confidence_scores']:
            plt.figure(figsize=(10, 6))
            plt.hist(results['confidence_scores'], bins=30, edgecolor='black')
            plt.title('Distribution of Confidence Scores')
            plt.xlabel('Confidence Score')
            plt.ylabel('Count')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'confidence_distribution_{timestamp}.png'))
            plt.close()
        
        #Content Length Distribution
        if results['content_length_stats']:
            plt.figure(figsize=(10, 6))
            plt.hist(results['content_length_stats'], bins=30, edgecolor='black')
            plt.title('Distribution of Content Lengths')
            plt.xlabel('Content Length (characters)')
            plt.ylabel('Count')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'content_length_distribution_{timestamp}.png'))
            plt.close()

    def generate_report(self, validation_results: Dict, output_dir: str) -> str:
        if not validation_results:
            logging.error("No validation results to report")
            return None

        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        #Generate text report
        report_file = os.path.join(output_dir, f"validation_report_{timestamp}.txt")
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=== Biology Dataset Validation Report ===\n\n")
            
            #Overall statistics
            f.write("Overall Statistics:\n")
            f.write(f"Total examples: {validation_results['total_examples']}\n")
            f.write(f"Valid examples: {validation_results['valid_examples']}\n")
            f.write(f"Invalid examples: {validation_results['invalid_examples']}\n")
            f.write(f"Validity rate: {(validation_results['valid_examples']/validation_results['total_examples']*100):.2f}%\n\n")
            
            #Confidence scores
            if validation_results['confidence_scores']:
                f.write("Confidence Score Statistics:\n")
                f.write(f"Mean confidence: {np.mean(validation_results['confidence_scores']):.2f}\n")
                f.write(f"Median confidence: {np.median(validation_results['confidence_scores']):.2f}\n")
                f.write(f"Min confidence: {min(validation_results['confidence_scores']):.2f}\n")
                f.write(f"Max confidence: {max(validation_results['confidence_scores']):.2f}\n\n")
            
            #Subfield distribution
            f.write("Subfield Distribution:\n")
            for subfield, count in validation_results['subfield_distribution'].items():
                percentage = (count / validation_results['valid_examples'] * 100)
                f.write(f"{subfield}: {count} ({percentage:.2f}%)\n")
            f.write("\n")
            
            #Generate visualizations
            self._generate_visualizations(validation_results, output_dir, timestamp)
        
        return report_file

def main():
    #Configuration
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(script_dir, "training_data", "biology_training_data.jsonl")
    output_dir = os.path.join(script_dir, "validation_reports")
    
    #Initialize validator
    validator = BiologyDataValidator()
    
    #Run validation
    logging.info("Starting validation...")
    validation_results = validator.validate_file(input_file)
    
    if validation_results:
        #Generate report
        report_file = validator.generate_report(validation_results, output_dir)
        logging.info(f"Validation complete. Report saved to {report_file}")
        
        #Print summary
        print("\nValidation Summary:")
        print(f"Total examples processed: {validation_results['total_examples']}")
        print(f"Valid biology examples: {validation_results['valid_examples']}")
        print(f"Invalid examples: {validation_results['invalid_examples']}")
        print(f"Validation rate: {(validation_results['valid_examples']/validation_results['total_examples']*100):.2f}%")
    else:
        logging.error("Validation failed!")

if __name__ == "__main__":
    main()

