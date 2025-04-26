import json
import tiktoken
from tqdm import tqdm
import logging
from typing import List, Dict, Set
import random
import os
import re
from collections import defaultdict
from pathlib import Path

#Set up logging with timestamp
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training_data/Filtered/balanced_split.log')
    ]
)

class BalancedSubjectSplitter:
    def __init__(self):
        self.encoding = tiktoken.encoding_for_model("gpt-4")
        self.MAX_TOKENS = 4096
        self.MIN_TOKENS = 10

        self.target_distribution = {
            "mathematics": 0.20,
            "physics": 0.15,
            "computer_science": 0.20,
            "biology": 0.15,
            "chemistry": 0.15,
            "history": 0.15
        }

        self.subjects = self._initialize_subjects()
        
    def _initialize_subjects(self) -> Dict:
        """Initialize subject definitions with comprehensive keyword lists."""
        return {
            "mathematics": {
                "primary_keywords": [
                    "mathematics", "algebra", "geometry", "calculus",
                    "mathematical theory", "mathematical concept"
                ],
                "secondary_keywords": [
                    "theorem", "proof", "equation", "mathematical",
                    "function", "number theory", "linear algebra", "arithmetic",
                    "topology", "analysis", "discrete mathematics",
                    "differential", "integral", "matrix", "vector",
                    "probability", "statistics", "set theory", "graph theory"
                ],
                "count": 0,
                "output_dir": "mathematics_training"
            },
            "computer_science": {
                "primary_keywords": [
                    "computer science", "programming", "algorithm", "data structure",
                    "software engineering", "computer system"
                ],
                "secondary_keywords": [
                    "software", "coding", "computation", "database", "network",
                    "artificial intelligence", "machine learning", 
                    "operating system", "compiler", "cybersecurity",
                    "web development", "cloud computing", "distributed systems",
                    "computer architecture", "information technology",
                    "python", "java", "javascript", "c++", "programming language"
                ],
                "count": 0,
                "output_dir": "cs_training"
            },
            "physics": {
                "primary_keywords": [
                    "physics", "quantum", "mechanics", "relativity",
                    "physical law", "physical theory"
                ],
                "secondary_keywords": [
                    "thermodynamics", "electromagnetic", "gravity", "force",
                    "energy", "motion", "dynamics", "kinematics", "optics",
                    "nuclear", "particle physics", "wave", "field theory",
                    "condensed matter", "astrophysics", "quantum mechanics",
                    "classical mechanics", "electromagnetism", "statistical physics"
                ],
                "count": 0,
                "output_dir": "physics_training"
            },
        }

    def setup_directory_structure(self, base_dir: str) -> Dict[str, str]:
        """Create necessary directory structure for output files."""
        paths = {}
        
        # Create main directories
        topic_training = os.path.join(base_dir, "topic_training_data")
        filtered_dir = os.path.join(base_dir, "Filtered")
        os.makedirs(topic_training, exist_ok=True)
        os.makedirs(filtered_dir, exist_ok=True)
        
        # Create subject-specific directories
        for subject, data in self.subjects.items():
            subject_dir = os.path.join(topic_training, data["output_dir"])
            os.makedirs(subject_dir, exist_ok=True)
            paths[subject] = subject_dir
            
        return paths

    def process_data(self, input_dir: str, output_base_dir: str, desired_total: int = 10000):
        """Process data files and create balanced datasets."""
        # Setup directory structure
        output_paths = self.setup_directory_structure(output_base_dir)
        
        # Locate input files
        input_files = []
        filtered_dir = os.path.join(input_dir, "Filtered")
        
        if os.path.exists(os.path.join(filtered_dir, "gpt4_training_data.jsonl")):
            input_files.append(os.path.join(filtered_dir, "gpt4_training_data.jsonl"))
        if os.path.exists(os.path.join(filtered_dir, "validated_training_data.jsonl")):
            input_files.append(os.path.join(filtered_dir, "validated_training_data.jsonl"))
        
        if not input_files:
            logging.error("No input files found in the specified directory")
            return

        articles_by_subject = defaultdict(list)
        
        for input_file in input_files:
            logging.info(f"Processing file: {input_file}")
            self._process_input_file(input_file, articles_by_subject)

        self._create_balanced_datasets(articles_by_subject, output_paths, desired_total)
        self._create_summary_report(output_base_dir)

    def _process_input_file(self, input_file: str, articles_by_subject: defaultdict):
        """Process a single input file and categorize articles."""
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc=f"Processing {os.path.basename(input_file)}"):
                try:
                    conversation = json.loads(line)
                    if not self._validate_conversation(conversation):
                        continue
                    
                    subject = self._categorize_conversation(conversation)
                    if subject != "other":
                        articles_by_subject[subject].append(conversation)
                        
                except json.JSONDecodeError:
                    continue

    def _validate_conversation(self, conversation: Dict) -> bool:
        """Validate conversation format and content."""
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

    def _categorize_conversation(self, conversation: Dict) -> str:
        """Categorize a conversation into a subject."""
        user_message = next((m for m in conversation['messages'] 
                           if m['role'] == 'user'), None)
        assistant_message = next((m for m in conversation['messages'] 
                                if m['role'] == 'assistant'), None)
        
        if not user_message or not assistant_message:
            return "other"
        
        title_match = re.search(r"What is (.*?)\?", user_message['content'])
        title = title_match.group(1) if title_match else ""
        content = assistant_message['content']
        
        return self.determine_subject(content, title)

    def _create_balanced_datasets(self, articles_by_subject: Dict, 
                                output_paths: Dict, desired_total: int):
        target_counts = {
            subject: int(desired_total * self.target_distribution[subject])
            for subject in self.subjects.keys()
        }
        
        for subject, target in target_counts.items():
            available = articles_by_subject[subject]
            n_articles = min(target, len(available))
            selected = random.sample(available, n_articles)
            
            output_file = os.path.join(output_paths[subject], 
                                     f"{subject}_balanced_dataset.jsonl")
            
            with open(output_file, 'w', encoding='utf-8') as f:
                for conversation in selected:
                    f.write(json.dumps(conversation) + '\n')
            
            self.subjects[subject]["count"] = n_articles
            logging.info(f"{subject}: selected {n_articles} articles (target: {target})")

    def _create_summary_report(self, base_dir: str):
        report_path = os.path.join(base_dir, "balanced_distribution_report.txt")
        
        total_selected = sum(data["count"] for data in self.subjects.values())
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=== Balanced Dataset Distribution Report ===\n\n")
            
            for subject, data in self.subjects.items():
                percentage = (data["count"] / total_selected) * 100
                f.write(f"{subject}:\n")
                f.write(f"  Articles: {data['count']}\n")
                f.write(f"  Percentage: {percentage:.1f}%\n")
                f.write(f"  Target: {self.target_distribution[subject]*100:.1f}%\n")
                f.write("\n")

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    splitter = BalancedSubjectSplitter()
    splitter.process_data(
        input_dir=os.path.join(base_dir, "training_data"),
        output_base_dir=base_dir,
        desired_total=10000
    )

if __name__ == "__main__":
    main()
