import json
import os
from collections import defaultdict
from typing import Dict, List
import logging
from tqdm import tqdm

class TopicSplitter:
    def __init__(self):
        self.topics = {
            'science': [
                'physics', 'chemistry', 'biology', 'astronomy',
                'mathematics', 'computer science', 'scientific'
            ],
            'history': [
                'history', 'historical', 'ancient', 'medieval',
                'civilization', 'empire', 'dynasty'
            ],
            'technology': [
                'technology', 'computer', 'software', 'hardware',
                'programming', 'artificial intelligence', 'digital'
            ],
            'arts': [
                'art', 'music', 'literature', 'painting',
                'sculpture', 'architecture', 'film', 'theater'
            ],
            'business': [
                'business', 'economics', 'finance', 'management',
                'marketing', 'entrepreneurship', 'commerce'
            ]
        }
        
        # Initialize metrics
        self.metrics = defaultdict(int)
        
    def detect_topic(self, text: str, title: str) -> str:
        """Detect the topic of an article based on content and title."""
        combined_text = (text + " " + title).lower()
        
        # Count keyword matches for each topic
        topic_scores = defaultdict(int)
        
        for topic, keywords in self.topics.items():
            for keyword in keywords:
                if keyword.lower() in combined_text:
                    topic_scores[topic] += 1
        
        # Return the topic with the most matches, or 'general' if no clear match
        if topic_scores:
            max_topic = max(topic_scores.items(), key=lambda x: x[1])
            if max_topic[1] > 0:  # If we have at least one match
                return max_topic[0]
        
        return 'general'

    def split_data(self, input_file: str, output_dir: str):
        """Split the training data into topic-specific files."""
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize file handlers for each topic
        topic_files = {}
        for topic in list(self.topics.keys()) + ['general']:
            file_path = os.path.join(output_dir, f'{topic}_training_data.jsonl')
            topic_files[topic] = open(file_path, 'w', encoding='utf-8')
        
        logging.info(f"Processing {input_file}...")
        
        # Process each conversation
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f):
                try:
                    conversation = json.loads(line)
                    
                    # Get the content from the conversation
                    messages = conversation.get('messages', [])
                    content = ""
                    title = ""
                    
                    for msg in messages:
                        if msg.get('role') == 'user':
                            title = msg.get('content', '')
                        if msg.get('role') == 'assistant':
                            content += msg.get('content', '') + " "
                    
                    # Detect topic
                    topic = self.detect_topic(content, title)
                    
                    # Write to appropriate file
                    topic_files[topic].write(line)
                    self.metrics[topic] += 1
                    
                except json.JSONDecodeError:
                    logging.error(f"Error decoding JSON line: {line[:100]}...")
                    continue
        
        # Close all files
        for file in topic_files.values():
            file.close()
        
        # Log statistics
        logging.info("\n=== Topic Distribution ===")
        total = sum(self.metrics.values())
        for topic, count in self.metrics.items():
            percentage = (count / total) * 100 if total > 0 else 0
            logging.info(f"{topic}: {count} ({percentage:.2f}%)")

def main():
    # Configuration
    input_file = "../Wikipedia_dumps/training_data/gpt4_training_data.jsonl"
    output_dir = "../backup/Old project/Backup_old/topics"
    
    # Initialize splitter
    splitter = TopicSplitter()
    
    # Process the data
    splitter.split_data(input_file, output_dir)

if __name__ == "__main__":
    main()
