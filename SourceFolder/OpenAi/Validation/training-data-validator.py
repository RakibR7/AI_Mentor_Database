
import json
import os
from collections import defaultdict
import logging
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from datetime import datetime
import tiktoken

class TrainingDataValidator:
    def __init__(self):
        self.encoding = tiktoken.encoding_for_model("gpt-4")
        
        # Configuration settings
        self.config = {
            'max_tokens': 2048,  # Maximum tokens per example
            'min_tokens': 10,    # Minimum tokens per example
            'required_fields': {'messages'},  # Required fields in each example
            'allowed_roles': {'system', 'user', 'assistant'},  # Allowed message roles
            'max_messages': 10,  # Maximum messages per conversation
            'min_messages': 2    # Minimum messages per conversation
        }
        
        # Initialize statistics
        self.stats = defaultdict(int)
        self.token_counts = []
        self.message_counts = []
        self.role_distributions = defaultdict(int)
        self.format_errors = defaultdict(list)
        self.content_distributions = {
            'lengths': [],
            'roles_per_example': []
        }

    def validate_format(self, example: Dict) -> Tuple[bool, str]:
        """Validate the format of a single example."""
        # Check for required fields
        if not all(field in example for field in self.config['required_fields']):
            return False, "missing_required_fields"

        # Validate messages array
        if not isinstance(example['messages'], list):
            return False, "messages_not_array"

        # Check message count
        if not (self.config['min_messages'] <= len(example['messages']) <= self.config['max_messages']):
            return False, f"invalid_message_count_{len(example['messages'])}"

        # Validate each message
        for msg in example['messages']:
            if not isinstance(msg, dict):
                return False, "message_not_object"
                
            if 'role' not in msg or 'content' not in msg:
                return False, "message_missing_fields"
                
            if msg['role'] not in self.config['allowed_roles']:
                return False, f"invalid_role_{msg['role']}"
                
            if not isinstance(msg['content'], str):
                return False, "content_not_string"
                
            if not msg['content'].strip():
                return False, "empty_content"

        return True, "valid"

    def count_tokens(self, example: Dict) -> int:
        """Count the total tokens in an example."""
        total_tokens = 0
        for message in example['messages']:
            total_tokens += len(self.encoding.encode(message['content']))
            total_tokens += len(self.encoding.encode(message['role']))
        return total_tokens

    def validate_file(self, input_file: str) -> Dict:
        """Validate entire training data file."""
        logging.info(f"Starting validation of {input_file}")
        
        validation_results = {
            'total_examples': 0,
            'valid_examples': 0,
            'invalid_examples': 0,
            'token_stats': {},
            'message_stats': {},
            'role_distribution': {},
            'format_errors': {},
            'sample_examples': []
        }

        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(tqdm(f, desc="Validating examples"), 1):
                    try:
                        example = json.loads(line)
                        validation_results['total_examples'] += 1
                        
                        # Format validation
                        is_valid_format, error_type = self.validate_format(example)
                        
                        if is_valid_format:
                            # Token count validation
                            token_count = self.count_tokens(example)
                            self.token_counts.append(token_count)
                            
                            if not (self.config['min_tokens'] <= token_count <= self.config['max_tokens']):
                                is_valid_format = False
                                error_type = f"token_count_{token_count}"
                        
                        # Update statistics
                        if is_valid_format:
                            validation_results['valid_examples'] += 1
                            
                            # Message statistics
                            msg_count = len(example['messages'])
                            self.message_counts.append(msg_count)
                            
                            # Role distribution
                            roles = [msg['role'] for msg in example['messages']]
                            for role in roles:
                                self.role_distributions[role] += 1
                            
                            # Content length distribution
                            for msg in example['messages']:
                                self.content_distributions['lengths'].append(len(msg['content']))
                            
                            self.content_distributions['roles_per_example'].append(len(set(roles)))
                            
                            # Store sample examples (up to 5)
                            if len(validation_results['sample_examples']) < 5:
                                validation_results['sample_examples'].append(example)
                        else:
                            validation_results['invalid_examples'] += 1
                            self.format_errors[error_type].append(line_num)
                            
                    except json.JSONDecodeError:
                        validation_results['invalid_examples'] += 1
                        self.format_errors['json_decode_error'].append(line_num)
                        continue

            # Calculate statistics
            validation_results['token_stats'] = {
                'mean': np.mean(self.token_counts) if self.token_counts else 0,
                'median': np.median(self.token_counts) if self.token_counts else 0,
                'min': min(self.token_counts) if self.token_counts else 0,
                'max': max(self.token_counts) if self.token_counts else 0
            }
            
            validation_results['message_stats'] = {
                'mean': np.mean(self.message_counts) if self.message_counts else 0,
                'median': np.median(self.message_counts) if self.message_counts else 0,
                'min': min(self.message_counts) if self.message_counts else 0,
                'max': max(self.message_counts) if self.message_counts else 0
            }
            
            validation_results['role_distribution'] = dict(self.role_distributions)
            validation_results['format_errors'] = dict(self.format_errors)
            
        except Exception as e:
            logging.error(f"Error processing file: {e}")
            return None

        return validation_results

    def generate_report(self, validation_results: Dict, output_dir: str) -> str:
        """Generate validation report with visualizations."""
        if not validation_results:
            return None

        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(output_dir, f"validation_report_{timestamp}.txt")

        # Generate text report
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=== Training Data Validation Report ===\n\n")
            
            # Overall statistics
            f.write("Overall Statistics:\n")
            f.write(f"Total examples: {validation_results['total_examples']}\n")
            f.write(f"Valid examples: {validation_results['valid_examples']}\n")
            f.write(f"Invalid examples: {validation_results['invalid_examples']}\n")
            f.write(f"Validity rate: {(validation_results['valid_examples']/validation_results['total_examples']*100):.2f}%\n\n")
            
            # Token statistics
            f.write("Token Statistics:\n")
            for stat, value in validation_results['token_stats'].items():
                f.write(f"{stat}: {value:.2f}\n")
            f.write("\n")
            
            # Message statistics
            f.write("Message Statistics:\n")
            for stat, value in validation_results['message_stats'].items():
                f.write(f"{stat}: {value:.2f}\n")
            f.write("\n")
            
            # Role distribution
            f.write("Role Distribution:\n")
            total_roles = sum(validation_results['role_distribution'].values())
            for role, count in validation_results['role_distribution'].items():
                percentage = (count / total_roles * 100)
                f.write(f"{role}: {count} ({percentage:.2f}%)\n")
            f.write("\n")
            
            # Format errors
            f.write("Format Errors:\n")
            for error_type, line_numbers in validation_results['format_errors'].items():
                f.write(f"{error_type}: {len(line_numbers)} occurrences\n")
            f.write("\n")
            
            # Sample examples
            f.write("Sample Valid Examples:\n")
            for i, example in enumerate(validation_results['sample_examples'], 1):
                f.write(f"\nExample {i}:\n")
                f.write(json.dumps(example, indent=2))
                f.write("\n")

        # Generate visualizations
        self._generate_visualizations(validation_results, output_dir, timestamp)
        
        return report_file

    def _generate_visualizations(self, results: Dict, output_dir: str, timestamp: str):
        """Generate visualization plots."""
        # 1. Token Distribution
        plt.figure(figsize=(10, 6))
        plt.hist(self.token_counts, bins=50, edgecolor='black')
        plt.title('Distribution of Token Counts')
        plt.xlabel('Token Count')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, f'token_distribution_{timestamp}.png'))
        plt.close()
        
        # 2. Message Count Distribution
        plt.figure(figsize=(10, 6))
        plt.hist(self.message_counts, bins=range(min(self.message_counts), max(self.message_counts) + 2),
                edgecolor='black')
        plt.title('Distribution of Messages per Example')
        plt.xlabel('Number of Messages')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, f'message_distribution_{timestamp}.png'))
        plt.close()
        
        # 3. Role Distribution
        plt.figure(figsize=(8, 6))
        roles = list(results['role_distribution'].keys())
        counts = list(results['role_distribution'].values())
        plt.bar(roles, counts)
        plt.title('Distribution of Message Roles')
        plt.xlabel('Role')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'role_distribution_{timestamp}.png'))
        plt.close()

def main():
    # Configuration
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(script_dir, "training_data", "training_data/Filtered/gpt4_training_data.jsonl")
    output_dir = os.path.join(script_dir, "validation_reports")
    
    # Initialize validator
    validator = TrainingDataValidator()
    
    # Run validation
    logging.info("Starting validation...")
    validation_results = validator.validate_file(input_file)
    
    if validation_results:
        # Generate report
        report_file = validator.generate_report(validation_results, output_dir)
        logging.info(f"Validation complete. Report saved to {report_file}")
        
        # Print summary
        print("\nValidation Summary:")
        print(f"Total examples processed: {validation_results['total_examples']}")
        print(f"Valid examples: {validation_results['valid_examples']}")
        print(f"Invalid examples: {validation_results['invalid_examples']}")
        print(f"Validity rate: {(validation_results['valid_examples']/validation_results['total_examples']*100):.2f}%")
        
        print("\nToken Statistics:")
        for stat, value in validation_results['token_stats'].items():
            print(f"{stat}: {value:.2f}")
            
        print("\nRole Distribution:")
        total_roles = sum(validation_results['role_distribution'].values())
        for role, count in validation_results['role_distribution'].items():
            percentage = (count / total_roles * 100)
            print(f"{role}: {count} ({percentage:.2f}%)")
    else:
        logging.error("Validation failed!")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()

