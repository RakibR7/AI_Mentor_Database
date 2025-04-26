
import json
import os
import re
import logging
import numpy as np
from typing import Dict, List, Tuple, Set
from collections import defaultdict
from tqdm import tqdm
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt

class SubjectProcessor:
    """Base class for processing subject-specific content."""
    
    def __init__(self, subject_name: str):
        self.subject_name = subject_name
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging configuration."""
        log_file = f"{self.subject_name.lower()}_processing.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_file)
            ]
        )

class ContentFilter:
    """Filter and validate content for specific subjects."""
    
    def __init__(self, subject_patterns: Dict):
        self.subject_patterns = subject_patterns
        self.vectorizer = TfidfVectorizer(max_features=5000)
        
    def is_valid_content(self, text: str, min_confidence: float = 0.2) -> Tuple[bool, float]:
        """
        Validate if content matches subject patterns.
        
        Args:
            text: Content to validate
            min_confidence: Minimum confidence threshold
            
        Returns:
            Tuple of (is_valid, confidence_score)
        """
        text = text.lower()
        confidence = self._calculate_confidence(text)
        return confidence >= min_confidence, confidence
    
    def _calculate_confidence(self, text: str) -> float:
        """Calculate confidence score based on pattern matches."""
        total_score = 0
        total_patterns = 0
        
        for category, patterns in self.subject_patterns.items():
            category_score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, text))
                category_score += matches
            
            if category_score > 0:
                total_score += category_score
            total_patterns += len(patterns)
        
        return total_score / total_patterns if total_patterns > 0 else 0

class DataProcessor:
    """Process and prepare training data."""
    
    def __init__(self, input_dir: str, output_dir: str):
        self.input_dir = input_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def process_file(self, file_path: str) -> List[Dict]:
        """Process a single file and extract content."""
        processed_data = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        processed_data.append(self._process_entry(data))
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logging.error(f"Error processing file {file_path}: {e}")
            
        return processed_data
    
    def _process_entry(self, entry: Dict) -> Dict:
        """Process individual data entry."""
        raise NotImplementedError("Implement in subject-specific class")

class TrainingDataCreator:
    """Create and format training data."""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def create_conversation(self, title: str, content: str) -> Dict:
        """Create conversation format from content."""
        return {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a knowledgeable AI mentor helping users learn about various topics."
                },
                {
                    "role": "user",
                    "content": f"What is {title}?"
                },
                {
                    "role": "assistant",
                    "content": content
                }
            ]
        }
    
    def save_to_jsonl(self, data: List[Dict], filename: str):
        """Save data in JSONL format."""
        output_path = os.path.join(self.output_dir, filename)
        with open(output_path, 'w', encoding='utf-8') as f:
            for entry in data:
                f.write(json.dumps(entry) + '\n')

class DataValidator:
    """Validate and analyze processed data."""
    
    def __init__(self):
        self.validation_results = defaultdict(list)
        
    def validate_dataset(self, data_path: str) -> Dict:
        """Validate entire dataset and generate statistics."""
        stats = {
            'total_examples': 0,
            'valid_examples': 0,
            'invalid_examples': 0,
            'confidence_scores': [],
            'content_lengths': [],
            'subject_distribution': defaultdict(int)
        }
        
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    stats['total_examples'] += 1
                    example = json.loads(line)
                    
                    is_valid, confidence = self._validate_example(example)
                    
                    if is_valid:
                        stats['valid_examples'] += 1
                        stats['confidence_scores'].append(confidence)
                        
                        content = self._get_content(example)
                        stats['content_lengths'].append(len(content))
                        
                        subject = self._determine_subject(content)
                        stats['subject_distribution'][subject] += 1
                    else:
                        stats['invalid_examples'] += 1
                        
        except Exception as e:
            logging.error(f"Error validating dataset: {e}")
            
        return stats
    
    def _validate_example(self, example: Dict) -> Tuple[bool, float]:
        """Validate individual example."""
        raise NotImplementedError("Implement in subject-specific validator")
    
    def _get_content(self, example: Dict) -> str:
        """Extract content from example."""
        return ' '.join([msg['content'] for msg in example['messages']])
    
    def _determine_subject(self, content: str) -> str:
        """Determine subject of content."""
        raise NotImplementedError("Implement in subject-specific validator")

class DataBalancer:
    """Balance dataset across subjects and categories."""
    
    def __init__(self, target_size: int = 3000):
        self.target_size = target_size
        
    def balance_dataset(self, data: List[Dict]) -> List[Dict]:
        """Balance dataset to target size."""
        if len(data) >= self.target_size:
            return self._reduce_dataset(data)
        else:
            return self._augment_dataset(data)
    
    def _reduce_dataset(self, data: List[Dict]) -> List[Dict]:
        """Reduce dataset while maintaining distribution."""
        return np.random.choice(data, self.target_size, replace=False).tolist()
    
    def _augment_dataset(self, data: List[Dict]) -> List[Dict]:
        """Augment dataset through example generation."""
        augmented_data = data.copy()
        
        while len(augmented_data) < self.target_size:
            base_example = np.random.choice(data)
            new_example = self._create_variant(base_example)
            augmented_data.append(new_example)
            
        return augmented_data
    
    def _create_variant(self, example: Dict) -> Dict:
        """Create variation of existing example."""
        raise NotImplementedError("Implement based on subject requirements")

class ReportGenerator:
    """Generate validation and analysis reports."""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def generate_report(self, stats: Dict, timestamp: str = None):
        """Generate comprehensive report with visualizations."""
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
        self._generate_text_report(stats, timestamp)
        self._generate_visualizations(stats, timestamp)
    
    def _generate_text_report(self, stats: Dict, timestamp: str):
        """Generate text report with statistics."""
        report_path = os.path.join(self.output_dir, f"report_{timestamp}.txt")
        
        with open(report_path, 'w') as f:
            f.write("=== Dataset Analysis Report ===\n\n")
            
            f.write("General Statistics:\n")
            f.write(f"Total examples: {stats['total_examples']}\n")
            f.write(f"Valid examples: {stats['valid_examples']}\n")
            f.write(f"Invalid examples: {stats['invalid_examples']}\n")
            f.write(f"Validation rate: {(stats['valid_examples']/stats['total_examples']*100):.2f}%\n\n")
            
            if stats['confidence_scores']:
                f.write("Confidence Statistics:\n")
                f.write(f"Mean confidence: {np.mean(stats['confidence_scores']):.3f}\n")
                f.write(f"Median confidence: {np.median(stats['confidence_scores']):.3f}\n")
                f.write(f"Min confidence: {min(stats['confidence_scores']):.3f}\n")
                f.write(f"Max confidence: {max(stats['confidence_scores']):.3f}\n\n")
    
    def _generate_visualizations(self, stats: Dict, timestamp: str):
        """Generate visualization plots."""
        # Distribution plot
        plt.figure(figsize=(12, 6))
        plt.hist(stats['confidence_scores'], bins=50)
        plt.title('Confidence Score Distribution')
        plt.xlabel('Confidence Score')
        plt.ylabel('Count')
        plt.savefig(os.path.join(self.output_dir, f"confidence_dist_{timestamp}.png"))
        plt.close()

def main():
    """Main execution function."""
    # Configuration
    input_dir = "wikipedia_dumps"
    output_dir = "processed_data"
    subject = "biology"  # Change for different subjects
    
    # Initialize processors
    processor = DataProcessor(input_dir, output_dir)
    validator = DataValidator()
    balancer = DataBalancer()
    report_gen = ReportGenerator("reports")
    
    # Process data
    logging.info(f"Processing {subject} data...")
    processed_data = []
    
    for root, _, files in os.walk(input_dir):
        for file in tqdm(files):
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                processed_data.extend(processor.process_file(file_path))
    
    # Validate and balance
    logging.info("Validating and balancing dataset...")
    stats = validator.validate_dataset(processed_data)
    balanced_data = balancer.balance_dataset(processed_data)
    
    # Generate report
    logging.info("Generating report...")
    report_gen.generate_report(stats)
    
    logging.info("Processing complete!")

if __name__ == "__main__":
    main()

