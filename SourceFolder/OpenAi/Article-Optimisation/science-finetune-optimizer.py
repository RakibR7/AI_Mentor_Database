import json
import tiktoken
from tqdm import tqdm
import logging
import re
from typing import List, Dict, Set
from collections import defaultdict

logging.basicConfig(level=logging.INFO)

class ScienceDataOptimizer:
    def __init__(self):
        self.encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        self.token_cost = 0.008

        self.science_categories = {
            'physics': {'physics', 'quantum', 'mechanics', 'relativity', 'particle', 'energy'},
            'chemistry': {'chemistry', 'chemical', 'molecule', 'reaction', 'compound'},
            'biology': {'biology', 'cell', 'genetics', 'evolution', 'organism'},
            'astronomy': {'astronomy', 'planet', 'star', 'galaxy', 'cosmos'},
            'math': {'mathematics', 'theorem', 'equation', 'algebra', 'geometry'}
        }

        self.quality_indicators = {
            'definition': r'(?:is|are|refers to|defined as)',
            'explanation': r'(?:explain|understand|concept|principle)',
            'example': r'(?:example|instance|case|such as)',
            'application': r'(?:used|applied|application|practical)'
        }

    def calculate_quality_score(self, content: str) -> float:
        """Calculate a quality score for the content."""
        score = 0.0

        for indicator_type, pattern in self.quality_indicators.items():
            if re.search(pattern, content.lower()):
                score += 0.25

        science_terms = 0
        for category_terms in self.science_categories.values():
            science_terms += sum(1 for term in category_terms if term in content.lower())
        score += min(science_terms / 10, 1)

        token_count = len(self.encoding.encode(content))
        if 100 <= token_count <= 500:
            score += 0.5
        
        return min(score, 2.0)

    def optimize_dataset(self, input_file: str, output_file: str, 
                        max_cost: float = 5.0) -> None:
        """Create an optimized science-focused dataset within budget."""
        max_tokens = int((max_cost / self.token_cost) * 1000)
        logging.info(f"Maximum tokens allowed: {max_tokens}")

        science_conversations = []
        total_examined = 0
        
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                for line in tqdm(f, desc="Analyzing conversations"):
                    total_examined += 1
                    try:
                        conv = json.loads(line)
                        content = conv["messages"][-1]["content"].lower()

                        is_science = False
                        for category_terms in self.science_categories.values():
                            if any(term in content for term in category_terms):
                                is_science = True
                                break
                        
                        if is_science:
                            quality_score = self.calculate_quality_score(content)
                            token_count = sum(len(self.encoding.encode(m["content"])) 
                                            for m in conv["messages"])
                            
                            science_conversations.append({
                                "conversation": conv,
                                "quality_score": quality_score,
                                "tokens": token_count
                            })
                            
                    except json.JSONDecodeError:
                        continue

            science_conversations.sort(key=lambda x: x["quality_score"], reverse=True)

            selected_conversations = []
            total_tokens = 0
            selected_topics = set()
            
            for conv_data in science_conversations:
                if total_tokens + conv_data["tokens"] > max_tokens:
                    continue

                content = conv_data["conversation"]["messages"][-1]["content"].lower()
                current_topic = None
                for topic, terms in self.science_categories.items():
                    if any(term in content for term in terms):
                        current_topic = topic
                        break

                if len(selected_conversations) < 5 or current_topic not in selected_topics:
                    selected_conversations.append(conv_data["conversation"])
                    total_tokens += conv_data["tokens"]
                    if current_topic:
                        selected_topics.add(current_topic)

            with open(output_file, 'w', encoding='utf-8') as f:
                for conv in selected_conversations:
                    f.write(json.dumps(conv) + '\n')

            actual_cost = (total_tokens / 1000) * self.token_cost
            topics_covered = len(selected_topics)
            
            logging.info("\n=== Optimization Results ===")
            logging.info(f"Total conversations examined: {total_examined}")
            logging.info(f"Science conversations selected: {len(selected_conversations)}")
            logging.info(f"Total tokens: {total_tokens}")
            logging.info(f"Estimated cost: ${actual_cost:.2f}")
            logging.info(f"Topics covered: {topics_covered}")
            logging.info(f"Topics: {', '.join(selected_topics)}")
            logging.info(f"Average tokens per conversation: {total_tokens/len(selected_conversations):.0f}")
            
        except Exception as e:
            logging.error(f"Error optimizing dataset: {e}")

def main():
    input_file = "training_data/Filtered/gpt4_training_data.jsonl"
    output_file = "training_data/Cost Redundacy/science_optimized_5dollars.jsonl"
    
    optimizer = ScienceDataOptimizer()
    optimizer.optimize_dataset(input_file, output_file, max_cost=5.0)

if __name__ == "__main__":
    main()
