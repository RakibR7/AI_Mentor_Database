import json
import tiktoken
from tqdm import tqdm
import logging
from typing import List, Dict
import os
from collections import defaultdict
import re

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('topic_split.log')
    ]
)

class TopicSplitter:
    def __init__(self):
        self.encoding = tiktoken.encoding_for_model("gpt-4")
        
        # Define topics and their keywords
        self.topics = {
            "mathematics": [
                "mathematics", "algebra", "geometry", "calculus", "statistics",
                "probability", "arithmetic", "mathematical", "theorem", "equation",
                "numerical", "computation", "logarithm", "multiplication"
            ],
            "physics": [
                "physics", "quantum", "mechanics", "relativity", "thermodynamics",
                "electromagnetic", "particle", "energy", "force", "motion",
                "gravity", "nuclear", "atom", "electron", "wavelength"
            ],
            "chemistry": [
                "chemistry", "chemical", "molecule", "atom", "reaction", "acid",
                "base", "organic", "inorganic", "compound", "element", "solution",
                "bond", "periodic table", "catalyst"
            ],
            "biology": [
                "biology", "cell", "organism", "gene", "evolution", "species",
                "ecosystem", "protein", "dna", "rna", "molecular", "tissue",
                "organ", "metabolism", "chromosome"
            ],
            "computer_science": [
                "computer science", "programming", "algorithm", "data structure",
                "database", "software", "hardware", "network", "operating system",
                "code", "computation", "artificial intelligence", "machine learning"
            ],
            "history": [
                "history", "civilization", "empire", "war", "revolution",
                "ancient", "medieval", "modern", "century", "dynasty",
                "historical", "archaeology", "artifact", "cultural"
            ],
            "literature": [
                "literature", "novel", "poetry", "author", "writing", "literary",
                "fiction", "narrative", "genre", "character", "plot", "story",
                "drama", "prose", "manuscript"
            ]
        }

    def detect_topic(self, text: str) -> str:
        """Detect the primary topic of a text based on keyword frequency."""
        text = text.lower()
        topic_scores = defaultdict(int)
        
        for topic, keywords in self.topics.items():
            for keyword in keywords:
                count = len(re.findall(r'\b' + re.escape(keyword) + r'\b', text))
                topic_scores[topic] += count
        
        if not topic_scores:
            return "general"
        
        return max(topic_scores.items(), key=lambda x: x[1])[0]

    def split_by_topics(self, input_file: str, output_dir: str, max_tokens_per_file: int = 1900000):
        """Split conversations into topic-specific files."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize token counters and conversation lists for each topic
        topic_data = {topic: {'conversations': [], 'tokens': 0} for topic in self.topics}
        topic_data['general'] = {'conversations': [], 'tokens': 0}
        
        file_counters = defaultdict(int)
        
        logging.info(f"Processing {input_file}...")
        
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f):
                try:
                    conversation = json.loads(line)
                    
                    # Combine all text for topic detection
                    full_text = ' '.join(msg['content'] for msg in conversation['messages'])
                    topic = self.detect_topic(full_text)
                    
                    # Count tokens
                    conv_tokens = sum(len(self.encoding.encode(msg['content'])) 
                                   for msg in conversation['messages'])
                    
                    # Check if adding this conversation would exceed token limit
                    if topic_data[topic]['tokens'] + conv_tokens > max_tokens_per_file:
                        # Save current file and start a new one
                        self._save_topic_file(
                            topic_data[topic]['conversations'],
                            topic,
                            file_counters[topic],
                            output_dir
                        )
                        file_counters[topic] += 1
                        topic_data[topic] = {'conversations': [], 'tokens': 0}
                    
                    # Add conversation to topic data
                    topic_data[topic]['conversations'].append(conversation)
                    topic_data[topic]['tokens'] += conv_tokens
                    
                except Exception as e:
                    logging.error(f"Error processing conversation: {e}")
                    continue
        
        # Save remaining conversations for each topic
        for topic, data in topic_data.items():
            if data['conversations']:
                self._save_topic_file(
                    data['conversations'],
                    topic,
                    file_counters[topic],
                    output_dir
                )
        
        # Log statistics
        self._log_statistics(topic_data, file_counters)

    def _save_topic_file(self, conversations: List[Dict], topic: str, 
                        file_number: int, output_dir: str) -> None:
        """Save conversations to a topic-specific JSONL file."""
        filename = f"{topic}_{file_number}.jsonl"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            for conv in conversations:
                f.write(json.dumps(conv) + '\n')
        
        logging.info(f"Saved {len(conversations)} conversations to {filepath}")

    def _log_statistics(self, topic_data: Dict, file_counters: Dict) -> None:
        """Log detailed statistics about the topic splitting process."""
        logging.info("\n=== Topic Split Statistics ===")
        for topic, data in topic_data.items():
            total_conversations = len(data['conversations'])
            total_tokens = data['tokens']
            num_files = file_counters[topic] + 1
            
            logging.info(f"\nTopic: {topic}")
            logging.info(f"Total conversations: {total_conversations}")
            logging.info(f"Total tokens: {total_tokens:,}")
            logging.info(f"Number of files: {num_files}")
            logging.info(f"Average tokens per conversation: {total_tokens/total_conversations:,.1f}")

def main():
    # Configuration
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(script_dir, "gpt4_training_data.jsonl")
    output_dir = os.path.join(script_dir, "topic_specific_data")
    
    # Initialize splitter
    splitter = TopicSplitter()
    
    # Process file with topic splitting
    splitter.split_by_topics(
        input_file=input_file,
        output_dir=output_dir,
        max_tokens_per_file=1900000  # Slightly under 2M limit to be safe
    )

if __name__ == "__main__":
    main()
