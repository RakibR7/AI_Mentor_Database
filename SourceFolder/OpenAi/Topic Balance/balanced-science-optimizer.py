import json
import tiktoken
from tqdm import tqdm
import logging
from collections import defaultdict
import random
from typing import Dict, List, Set

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class BalancedScienceOptimizer:
    def __init__(self):
        self.encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        self.token_cost = 0.008

        self.topics = {
            'physics': {
                'keywords': {'physics', 'quantum', 'mechanics', 'relativity', 'particle', 'energy', 
                           'force', 'motion', 'wave', 'electromagnetic', 'nucleus', 'atom'},
                'conversations': []
            },
            'chemistry': {
                'keywords': {'chemistry', 'chemical', 'molecule', 'reaction', 'compound', 'element', 
                           'acid', 'base', 'organic', 'inorganic', 'bond', 'solution'},
                'conversations': []
            },
            'biology': {
                'keywords': {'biology', 'cell', 'genetics', 'evolution', 'organism', 'protein', 
                           'dna', 'species', 'ecosystem', 'metabolism', 'enzyme'},
                'conversations': []
            },
            'astronomy': {
                'keywords': {'astronomy', 'planet', 'star', 'galaxy', 'cosmos', 'solar system', 
                           'nebula', 'black hole', 'universe', 'space', 'celestial'},
                'conversations': []
            },
            'mathematics': {
                'keywords': {'mathematics', 'theorem', 'equation', 'algebra', 'geometry', 'calculus', 
                           'function', 'probability', 'statistics', 'mathematical'},
                'conversations': []
            }
        }

    def calculate_quality_score(self, content: str, topic_keywords: Set[str]) -> float:
        """Calculate quality score for content based on multiple factors."""
        score = 0.0
        content_lower = content.lower()
        

        keyword_count = sum(1 for keyword in topic_keywords if keyword in content_lower)
        score += min(keyword_count / 5, 1.0)

        educational_patterns = [
            r'is defined as',
            r'refers to',
            r'example of',
            r'explanation',
            r'understand',
            r'concept',
            r'principle'
        ]
        pattern_matches = sum(1 for pattern in educational_patterns 
                            if pattern in content_lower)
        score += pattern_matches * 0.2

        token_count = len(self.encoding.encode(content))
        if 200 <= token_count <= 800:
            score += 1.0
        elif 100 <= token_count <= 1000:
            score += 0.5
        
        return score

    def classify_and_score_conversation(self, conversation: Dict) -> List[tuple]:
        content = conversation["messages"][-1]["content"]
        results = []
        
        for topic, data in self.topics.items():
            if any(keyword in content.lower() for keyword in data['keywords']):
                score = self.calculate_quality_score(content, data['keywords'])
                if score > 1.0:
                    results.append((topic, score, conversation))
        
        return results

    def optimize_dataset(self, input_file: str, output_file: str, 
                        examples_per_topic: int = 30) -> None:
        logging.info(f"Creating balanced dataset with {examples_per_topic} examples per topic...")

        for topic_data in self.topics.values():
            topic_data['conversations'] = []

        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                for line in tqdm(f, desc="Processing conversations"):
                    try:
                        conversation = json.loads(line)
                        classifications = self.classify_and_score_conversation(conversation)
                        
                        for topic, score, conv in classifications:
                            self.topics[topic]['conversations'].append((score, conv))
                    except json.JSONDecodeError:
                        continue

            selected_conversations = []
            total_tokens = 0
            
            for topic, data in self.topics.items():

                data['conversations'].sort(key=lambda x: x[0], reverse=True)
                top_conversations = data['conversations'][:examples_per_topic]
                
                for _, conv in top_conversations:
                    selected_conversations.append(conv)
                    total_tokens += sum(len(self.encoding.encode(m['content'])) 
                                     for m in conv['messages'])
            random.shuffle(selected_conversations)

            with open(output_file, 'w', encoding='utf-8') as f:
                for conv in selected_conversations:
                    f.write(json.dumps(conv) + '\n')

            actual_cost = (total_tokens / 1000) * self.token_cost
            
            logging.info("\n=== Dataset Statistics ===")
            for topic, data in self.topics.items():
                selected_count = len(data['conversations'][:examples_per_topic])
                logging.info(f"{topic.capitalize()}: {selected_count} examples")
            
            logging.info(f"\nTotal conversations: {len(selected_conversations)}")
            logging.info(f"Total tokens: {total_tokens:,}")
            logging.info(f"Estimated cost: ${actual_cost:.2f}")
            logging.info(f"Average tokens per conversation: {total_tokens/len(selected_conversations):.0f}")
            
        except Exception as e:
            logging.error(f"Error optimizing dataset: {e}")

def main():
    input_file = "../Wikipedia_dumps/training_data/gpt4_training_data.jsonl"
    output_file = "../Wikipedia_dumps/training_data/Cost Redundacy/balanced_science_30each.jsonl"
    
    optimizer = BalancedScienceOptimizer()
    optimizer.optimize_dataset(input_file, output_file, examples_per_topic=30)

if __name__ == "__main__":
    main()
