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
            'physics': {
                'physics', 'quantum', 'mechanics', 'relativity', 'particle', 'energy',
                'force', 'motion', 'gravity', 'electromagnetic', 'nuclear', 'waves'
            },
            'chemistry': {
                'chemistry', 'chemical', 'molecule', 'reaction', 'compound', 'element',
                'atomic', 'acid', 'base', 'organic', 'inorganic', 'bond'
            },
            'biology': {
                'biology', 'cell', 'genetics', 'evolution', 'organism', 'protein',
                'dna', 'species', 'enzyme', 'molecular', 'tissue', 'system'
            },
            'astronomy': {
                'astronomy', 'planet', 'star', 'galaxy', 'cosmos', 'solar',
                'space', 'universe', 'orbital', 'celestial', 'stellar'
            },
            'math': {
                'mathematics', 'theorem', 'equation', 'algebra', 'geometry',
                'calculus', 'statistical', 'function', 'probability', 'mathematical'
            }
        }

    def calculate_quality_score(self, content: str) -> float:
        """Calculate a quality score for the content."""
        score = 0.0
        content_lower = content.lower()

        education_patterns = {
            'definition': r'\b(is|are|refers to|defined as)\b',
            'explanation': r'\b(because|therefore|thus|hence|explains?)\b',
            'example': r'\b(for example|such as|instance|like)\b',
            'structure': r'\b(first|second|finally|moreover|however)\b'
        }
        
        for pattern in education_patterns.values():
            if re.search(pattern, content_lower):
                score += 0.2

        science_term_count = 0
        for category_terms in self.science_categories.values():
            terms_found = sum(1 for term in category_terms if term in content_lower)
            science_term_count += terms_found
        
        score += min(science_term_count / 5, 1.0)

        token_count = len(self.encoding.encode(content))
        if 300 <= token_count <= 800:
            score += 0.5

        if re.search(r'\d+(\.\d+)?', content):
            score += 0.2
        
        return score

    def optimize_dataset(self, input_file: str, output_file: str, max_cost: float = 5.0) -> None:
        max_tokens = int((max_cost / self.token_cost) * 1000)
        logging.info(f"Maximum tokens allowed: {max_tokens}")

        science_conversations = []
        category_counts = defaultdict(int)
        
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Analyzing conversations"):
                try:
                    conv = json.loads(line)
                    content = conv["messages"][-1]["content"]
                    content_lower = content.lower()

                    categories_present = set()
                    for category, terms in self.science_categories.items():
                        if any(term in content_lower for term in terms):
                            categories_present.add(category)
                    
                    if categories_present:
                        quality_score = self.calculate_quality_score(content)
                        token_count = sum(len(self.encoding.encode(m["content"])) 
                                        for m in conv["messages"])
                        
                        if 200 <= token_count <= 1000:
                            science_conversations.append({
                                "conversation": conv,
                                "categories": categories_present,
                                "quality_score": quality_score,
                                "tokens": token_count
                            })
                
                except json.JSONDecodeError:
                    continue

        selected_conversations = []
        total_tokens = 0
        target_per_category = 10

        science_conversations.sort(key=lambda x: x["quality_score"], reverse=True)

        for category in self.science_categories.keys():
            category_examples = [
                conv for conv in science_conversations 
                if category in conv["categories"] 
                and total_tokens + conv["tokens"] <= max_tokens
            ][:target_per_category]
            
            for conv_data in category_examples:
                if conv_data not in selected_conversations:
                    selected_conversations.append(conv_data)
                    total_tokens += conv_data["tokens"]
                    for cat in conv_data["categories"]:
                        category_counts[cat] += 1

        with open(output_file, 'w', encoding='utf-8') as f:
            for conv_data in selected_conversations:
                f.write(json.dumps(conv_data["conversation"]) + '\n')

        actual_cost = (total_tokens / 1000) * self.token_cost
        logging.info("\n=== Optimization Results ===")
        logging.info(f"Selected conversations: {len(selected_conversations)}")
        logging.info(f"Total tokens: {total_tokens}")
        logging.info(f"Estimated cost: ${actual_cost:.2f}")
        logging.info("\nCategory distribution:")
        for category, count in category_counts.items():
            logging.info(f"- {category}: {count} examples")
        logging.info(f"\nAverage tokens per conversation: {total_tokens/len(selected_conversations):.0f}")

def main():
    input_file = "training_data/Filtered/gpt4_training_data.jsonl"
    output_file = "training_data/Cost Redundacy/science_optimized_5dollars_v2.jsonl"
    
    optimizer = ScienceDataOptimizer()
    optimizer.optimize_dataset(input_file, output_file, max_cost=5.0)

if __name__ == "__main__":
    main()
