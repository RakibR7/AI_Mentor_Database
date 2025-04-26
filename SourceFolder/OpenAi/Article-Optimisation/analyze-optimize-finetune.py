import json
import random
import tiktoken
from tqdm import tqdm
from typing import List, Dict
import logging
from collections import defaultdict

logging.basicConfig(level=logging.INFO)

class FineTuneOptimizer:
    def __init__(self, model_name="gpt-4"):
        self.encoding = tiktoken.encoding_for_model(model_name)
        self.token_costs = {
            "gpt-4": 0.03,
            "gpt-3.5-turbo": 0.0080
        }
    
    def analyze_file(self, input_file: str) -> Dict:
        """Analyze the JSONL file for token usage and costs."""
        stats = {
            "total_examples": 0,
            "total_tokens": 0,
            "token_distribution": defaultdict(int),
            "topic_distribution": defaultdict(int),
            "length_distribution": defaultdict(int)
        }
        
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                for line in tqdm(f):
                    conversation = json.loads(line)
                    stats["total_examples"] += 1
                    
                    # Count tokens
                    tokens = 0
                    for message in conversation["messages"]:
                        tokens += len(self.encoding.encode(message["content"]))
                    
                    stats["total_tokens"] += tokens
                    stats["token_distribution"][tokens // 500 * 500] += 1
                    
                    # Analyze content
                    content = conversation["messages"][-1]["content"]
                    stats["length_distribution"][len(content) // 1000 * 1000] += 1
                    
                    # Simple topic detection
                    for topic, keywords in TOPIC_KEYWORDS.items():
                        if any(keyword in content.lower() for keyword in keywords):
                            stats["topic_distribution"][topic] += 1
                            break
            
            return stats
        except Exception as e:
            logging.error(f"Error analyzing file: {e}")
            return stats

    def optimize_dataset(self, input_file: str, output_file: str, 
                        max_cost: float = 100.0, model: str = "gpt-4"):
        """Create an optimized dataset within budget constraints."""
        try:
            conversations = []
            with open(input_file, 'r', encoding='utf-8') as f:
                for line in f:
                    conversations.append(json.loads(line))
            
            # Random sampling with token limit
            max_tokens = int((max_cost / self.token_costs[model]) * 1000)
            
            #  or Prioritize diverse topics
            topic_buckets = defaultdict(list)
            for conv in conversations:
                content = conv["messages"][-1]["content"].lower()
                for topic, keywords in TOPIC_KEYWORDS.items():
                    if any(keyword in content for keyword in keywords):
                        topic_buckets[topic].append(conv)
                        break
            
            # or Select balanced sample
            optimized_conversations = []
            total_tokens = 0
            
            # Take equal samples from each topic
            tokens_per_topic = max_tokens // len(TOPIC_KEYWORDS)
            for topic, convs in topic_buckets.items():
                topic_tokens = 0
                random.shuffle(convs)
                
                for conv in convs:
                    conv_tokens = sum(len(self.encoding.encode(m["content"])) 
                                   for m in conv["messages"])
                    
                    if topic_tokens + conv_tokens <= tokens_per_topic:
                        optimized_conversations.append(conv)
                        topic_tokens += conv_tokens
                        total_tokens += conv_tokens
            
            # Write optimized dataset
            with open(output_file, 'w', encoding='utf-8') as f:
                for conv in optimized_conversations:
                    f.write(json.dumps(conv) + '\n')
            
            # Calculate estimated cost
            estimated_cost = (total_tokens / 1000) * self.token_costs[model]
            
            logging.info(f"Optimized dataset created:")
            logging.info(f"Total conversations: {len(optimized_conversations)}")
            logging.info(f"Total tokens: {total_tokens}")
            logging.info(f"Estimated cost: ${estimated_cost:.2f}")
            
            return len(optimized_conversations), total_tokens, estimated_cost
            
        except Exception as e:
            logging.error(f"Error optimizing dataset: {e}")
            return 0, 0, 0

# Topic keywords for classification
TOPIC_KEYWORDS = {
    "science": ["physics", "chemistry", "biology", "scientific", "research"],
    "technology": ["computer", "software", "technology", "digital", "internet"],
    "history": ["history", "historical", "century", "ancient", "medieval"],
    "arts": ["art", "music", "literature", "cultural", "artistic"],
    "biography": ["born", "died", "career", "life", "works"]
}

def main():
    input_file = "../Wikipedia_dumps/training_data/gpt4_training_data.jsonl"
    optimizer = FineTuneOptimizer()
    logging.info("Analyzing current dataset...")
    stats = optimizer.analyze_file(input_file)

    #Create optimized versions for different budgets
    budgets = [50, 100, 200]
    for budget in budgets:
        output_file = f"training_data/optimized_data_{budget}.jsonl"
        logging.info(f"\nOptimizing dataset for ${budget} budget...")
        convs, tokens, cost = optimizer.optimize_dataset(
            input_file, output_file, max_cost=budget
        )

if __name__ == "__main__":
    main()
