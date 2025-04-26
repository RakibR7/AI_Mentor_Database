
import json
import os
from collections import defaultdict
import logging
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from typing import Dict, List, Tuple
from datetime import datetime
import tiktoken
import shutil

class UnifiedDatasetValidator:
    def __init__(self):
        self.encoding = tiktoken.encoding_for_model("gpt-4")
        
        # Dataset paths
        self.dataset_paths = {
            'topic_training': 'topic_training_data',
            'subject_specific': 'training_data/Articles/subject_datasets',
            'balanced_subjects': 'training_data/balanced_subjects',
            'filtered': 'training_data/Filtered',
            'cost_optimized': 'training_data/Cost Redundacy'
        }
        
        # Validation config
        self.config = {
            'max_tokens': 2048,
            'min_tokens': 10,
            'required_fields': {'messages'},
            'allowed_roles': {'system', 'user', 'assistant'},
            'max_messages': 10,
            'min_messages': 2
        }

    def analyze_all_datasets(self, base_dir: str) -> Dict:
        """Analyze all datasets in the project."""
        results = defaultdict(dict)
        
        for dataset_type, relative_path in self.dataset_paths.items():
            full_path = os.path.join(base_dir, relative_path)
            if os.path.exists(full_path):
                logging.info(f"\nAnalyzing {dataset_type} datasets...")
                
                # Get all JSONL files in directory
                jsonl_files = []
                for root, _, files in os.walk(full_path):
                    jsonl_files.extend([
                        os.path.join(root, f) for f in files 
                        if f.endswith('.jsonl')
                    ])
                
                # Analyze each file
                for file_path in jsonl_files:
                    subject_name = os.path.splitext(os.path.basename(file_path))[0]
                    logging.info(f"Processing {subject_name}...")
                    
                    file_results = self.validate_file(file_path)
                    if file_results:
                        results[dataset_type][subject_name] = file_results
        
        return results

    def validate_file(self, file_path: str) -> Dict:
        """Validate and analyze a single JSONL file."""
        stats = {
            'total_examples': 0,
            'valid_examples': 0,
            'invalid_examples': 0,
            'token_stats': defaultdict(list),
            'message_stats': defaultdict(list),
            'role_distribution': defaultdict(int),
            'content_lengths': [],
            'errors': defaultdict(int)
        }
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(tqdm(f), 1):
                    try:
                        example = json.loads(line)
                        stats['total_examples'] += 1
                        
                        # Validate format and content
                        is_valid, error_type = self._validate_example(example)
                        
                        if is_valid:
                            stats['valid_examples'] += 1
                            self._update_stats(stats, example)
                        else:
                            stats['invalid_examples'] += 1
                            stats['errors'][error_type] += 1
                            
                    except json.JSONDecodeError:
                        stats['invalid_examples'] += 1
                        stats['errors']['json_decode_error'] += 1
                        
        except Exception as e:
            logging.error(f"Error processing {file_path}: {e}")
            return None
            
        # Calculate aggregate statistics
        stats['token_stats'] = {
            'mean': np.mean(stats['token_stats']['counts']),
            'median': np.median(stats['token_stats']['counts']),
            'min': min(stats['token_stats']['counts']),
            'max': max(stats['token_stats']['counts'])
        }
        
        stats['message_stats'] = {
            'mean': np.mean(stats['message_stats']['counts']),
            'median': np.median(stats['message_stats']['counts']),
            'min': min(stats['message_stats']['counts']),
            'max': max(stats['message_stats']['counts'])
        }
        
        return stats

    def _validate_example(self, example: Dict) -> Tuple[bool, str]:
        """Validate a single example."""
        if not all(field in example for field in self.config['required_fields']):
            return False, "missing_required_fields"
            
        if not isinstance(example['messages'], list):
            return False, "messages_not_array"
            
        if not (self.config['min_messages'] <= len(example['messages']) <= self.config['max_messages']):
            return False, "invalid_message_count"
            
        token_count = 0
        for msg in example['messages']:
            if not isinstance(msg, dict) or 'role' not in msg or 'content' not in msg:
                return False, "invalid_message_format"
                
            if msg['role'] not in self.config['allowed_roles']:
                return False, "invalid_role"
                
            token_count += len(self.encoding.encode(msg['content']))
            
        if not (self.config['min_tokens'] <= token_count <= self.config['max_tokens']):
            return False, "invalid_token_count"
            
        return True, ""

    def _update_stats(self, stats: Dict, example: Dict):
        """Update statistics for a valid example."""
        # Token counts
        total_tokens = sum(len(self.encoding.encode(msg['content'])) 
                         for msg in example['messages'])
        stats['token_stats']['counts'].append(total_tokens)
        
        # Message counts
        stats['message_stats']['counts'].append(len(example['messages']))
        
        # Role distribution
        for msg in example['messages']:
            stats['role_distribution'][msg['role']] += 1
            stats['content_lengths'].append(len(msg['content']))

    def generate_comparative_report(self, results: Dict, output_dir: str):
        """Generate comparative analysis report across all datasets."""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create report file
        report_file = os.path.join(output_dir, f"comparative_analysis_{timestamp}.txt")
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=== Comparative Dataset Analysis Report ===\n\n")
            
            for dataset_type, subjects in results.items():
                f.write(f"\n{dataset_type.upper()} DATASETS\n")
                f.write("=" * 50 + "\n")
                
                for subject, stats in subjects.items():
                    f.write(f"\n{subject}:\n")
                    f.write(f"Total examples: {stats['total_examples']}\n")
                    f.write(f"Valid examples: {stats['valid_examples']}\n")
                    f.write(f"Validity rate: {(stats['valid_examples']/stats['total_examples']*100):.2f}%\n")
                    
                    f.write("\nToken Statistics:\n")
                    for stat, value in stats['token_stats'].items():
                        f.write(f"{stat}: {value:.2f}\n")
                    
                    f.write("\nRole Distribution:\n")
                    total_roles = sum(stats['role_distribution'].values())
                    for role, count in stats['role_distribution'].items():
                        percentage = (count / total_roles * 100)
                        f.write(f"{role}: {count} ({percentage:.2f}%)\n")
                    
                    if stats['errors']:
                        f.write("\nErrors:\n")
                        for error, count in stats['errors'].items():
                            f.write(f"{error}: {count}\n")
                    f.write("\n" + "-"*40 + "\n")
        
        # Generate visualizations
        self._generate_comparative_visualizations(results, output_dir, timestamp)
        
        return report_file

    def _generate_comparative_visualizations(self, results: Dict, output_dir: str, timestamp: str):
        """Generate comparative visualizations across datasets."""
        # Prepare data for plotting
        plot_data = {
            'validity_rates': defaultdict(list),
            'token_stats': defaultdict(list),
            'role_distributions': defaultdict(list)
        }
        
        for dataset_type, subjects in results.items():
            for subject, stats in subjects.items():
                # Validity rates
                validity_rate = (stats['valid_examples']/stats['total_examples']*100)
                plot_data['validity_rates']['dataset'].append(dataset_type)
                plot_data['validity_rates']['subject'].append(subject)
                plot_data['validity_rates']['rate'].append(validity_rate)
                
                # Token statistics
                for stat, value in stats['token_stats'].items():
                    plot_data['token_stats']['dataset'].append(dataset_type)
                    plot_data['token_stats']['subject'].append(subject)
                    plot_data['token_stats']['metric'].append(stat)
                    plot_data['token_stats']['value'].append(value)
        
        # 1. Validity Rates Comparison
        plt.figure(figsize=(15, 8))
        df = pd.DataFrame(plot_data['validity_rates'])
        sns.barplot(data=df, x='subject', y='rate', hue='dataset')
        plt.title('Validity Rates Across Datasets')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'validity_comparison_{timestamp}.png'))
        plt.close()
        
        # 2. Token Statistics Comparison
        plt.figure(figsize=(15, 8))
        df = pd.DataFrame(plot_data['token_stats'])
        sns.boxplot(data=df, x='subject', y='value', hue='dataset')
        plt.title('Token Statistics Across Datasets')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'token_stats_comparison_{timestamp}.png'))
        plt.close()

def main():
    # Get the project root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)  # Assuming script is in project root
    
    # Initialize validator
    validator = UnifiedDatasetValidator()
    
    # Run analysis
    logging.info("Starting comprehensive dataset analysis...")
    results = validator.analyze_all_datasets(base_dir)
    
    # Generate comparative report
    output_dir = os.path.join(base_dir, "analysis_reports")
    report_file = validator.generate_comparative_report(results, output_dir)
    
    logging.info(f"Analysis complete. Report saved to {report_file}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()

