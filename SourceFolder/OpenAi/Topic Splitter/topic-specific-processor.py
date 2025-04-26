import json
import os
from tqdm import tqdm
import logging
from typing import Dict, List, Set
import tiktoken

# Set up logging
logging.basicConfig(level=logging.INFO)

class TopicProcessor:
    def __init__(self):
        # Define main topics and their related keywords
        self.topics = {
            "computer_science": {
                "keywords": [
                    "programming", "algorithm", "computer", "software", "database",
                    "artificial intelligence", "machine learning", "coding", "data structure",
                    "web development", "python", "java", "javascript", "computing"
                ],
                "max_articles": 2000  # Adjust based on your needs
            },
            "mathematics": {
                "keywords": [
                    "mathematics", "algebra", "geometry", "calculus", "statistics",
                    "probability", "theorem", "mathematical", "equation", "number theory",
                    "linear algebra", "arithmetic", "mathematical concept"
                ],
                "max_articles": 2000
            },
            "physics": {
                "keywords": [
                    "physics", "quantum", "mechanics", "relativity", "particle",
                    "thermodynamics", "electromagnetic", "nuclear", "gravitational",
                    "physics concept", "physical law", "energy", "force"
                ],
                "max_articles": 2000
            },
            "biology": {
                "keywords": [
                    "biology", "cell", "genetics", "evolution", "organism",
                    "molecular", "ecology", "biological", "species", "anatomy",
                    "physiology", "biochemistry", "biological process"
                ],
                "max_articles": 2000
            },
            "history": {
                "keywords": [
                    "history", "historical", "civilization", "empire", "war",
                    "revolution", "ancient", "medieval", "modern history",
                    "historical event", "historical figure", "dynasty"
                ],
                "max_articles": 2000
            }
        }

    def classify_article(self, article: Dict) -> str:
        """Classify article into a topic based on keyword matching."""
        text = (article.get('title', '') + ' ' + article.get('text', '')).lower()
        
        # Count keyword matches for each topic
        topic_matches = {}
        for topic, info in self.topics.items():
            matches = sum(1 for keyword in info["keywords"] if keyword.lower() in text)
            if matches > 0:
                topic_matches[topic] = matches
        
        # Return the topic with the most matches, or None if no matches
        if topic_matches:
            return max(topic_matches.items(), key=lambda x: x[1])[0]
        return "other"

    def create_conversation(self, article: Dict, topic: str) -> Dict:
        """Create a conversation format optimized for GPT-4."""
        title = article.get('title', 'Unknown')
        text = article.get('text', '').strip()
        
        # Create a more focused system message based on the topic
        system_messages = {
            "computer_science": "You are an expert computer science mentor helping users understand computing concepts.",
            "mathematics": "You are an expert mathematics mentor helping users understand mathematical concepts.",
            "physics": "You are an expert physics mentor helping users understand physics concepts.",
            "biology": "You are an expert biology mentor helping users understand biological concepts.",
            "history": "You are an expert history mentor helping users understand historical events and concepts."
        }
        
        return {
            "messages": [
                {"role": "system", "content": system_messages.get(topic, "You are a knowledgeable mentor helping users learn about various topics.")},
                {"role": "user", "content": f"What is {title}? Please explain it clearly."},
                {"role": "assistant", "content": text}
            ]
        }

def process_wiki_data(input_directory: str, output_directory: str):
    """Process Wikipedia data into topic-specific files."""
    processor = TopicProcessor()
    topic_counters = {topic: 0 for topic in processor.topics.keys()}
    topic_files = {}
    
    # Initialize token counter
    encoding = tiktoken.encoding_for_model("gpt-4")
    topic_tokens = {topic: 0 for topic in processor.topics.keys()}
    
    # Create output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)
    
    # Initialize file handlers
    for topic in processor.topics.keys():
        file_path = os.path.join(output_directory, f"{topic}_training.jsonl")
        topic_files[topic] = open(file_path, 'w', encoding='utf-8')
    
    # Process each file in the input directory
    for root, _, files in tqdm(os.walk(input_directory)):
        for file in files:
            if file.startswith('wiki_') and file.endswith('.txt'):
                file_path = os.path.join(root, file)
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            article = json.loads(line)
                            topic = processor.classify_article(article)
                            
                            if topic in processor.topics:
                                # Check if we've reached the max articles for this topic
                                if topic_counters[topic] >= processor.topics[topic]["max_articles"]:
                                    continue
                                
                                conversation = processor.create_conversation(article, topic)
                                
                                # Count tokens
                                tokens = sum(len(encoding.encode(msg["content"])) 
                                           for msg in conversation["messages"])
                                
                                # Check if adding this would exceed token limit
                                if topic_tokens[topic] + tokens <= 2000000:  # 2M token limit
                                    topic_files[topic].write(json.dumps(conversation) + '\n')
                                    topic_tokens[topic] += tokens
                                    topic_counters[topic] += 1
                                
                        except json.JSONDecodeError:
                            continue
    
    # Close all files
    for file in topic_files.values():
        file.close()
    
    # Print statistics
    for topic in processor.topics.keys():
        logging.info(f"{topic}: {topic_counters[topic]} articles, {topic_tokens[topic]} tokens")

def main():
    input_directory = "training_data/validated_training_data.jsonl"  # Your Wikipedia data directory
    output_directory = "training_data/topic_training_data"  # Output directory for topic-specific files
    
    logging.info("Starting topic-specific data processing...")
    process_wiki_data(input_directory, output_directory)
    logging.info("Processing complete!")

if __name__ == "__main__":
    main()
