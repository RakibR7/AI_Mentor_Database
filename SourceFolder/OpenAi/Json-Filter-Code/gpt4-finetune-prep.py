import json
import tiktoken
from tqdm import tqdm
import logging
from typing import List, Dict
import random

logging.basicConfig(level=logging.INFO)

class GPT4DataPrep:
    def __init__(self):
        # Initialize tokenizer for GPT-4
        self.encoding = tiktoken.encoding_for_model("gpt-4")
        
        # GPT-4 specific limits
        self.MAX_TOKENS_PER_EXAMPLE = 4096
        self.MIN_TOKENS_PER_EXAMPLE = 50
        
        # Quality thresholds
        self.MIN_CONTENT_LENGTH = 100  # characters
        self.MAX_CONTENT_LENGTH = 15000  # characters
        
    def create_system_message(self, article_type: str) -> str:
        """Create appropriate system message based on article type."""
        system_messages = {
            "default": "You are a knowledgeable AI mentor helping users learn about various topics. Provide clear, accurate, and detailed explanations.",
            "technical": "You are a technical expert AI mentor. Explain complex concepts clearly, using analogies when helpful.",
            "historical": "You are a history expert AI mentor. Provide detailed historical context and explain the significance of events.",
            "scientific": "You are a science expert AI mentor. Explain scientific concepts clearly, relating them to real-world applications."
        }
        return system_messages.get(article_type, system_messages["default"])

    def prepare_conversation(self, article: Dict) -> Dict:
        """Create an optimized conversation format for GPT-4 fine-tuning."""
        title = article.get('title', '')
        text = article.get('text', '').strip()
        
        # Determine article type based on content
        article_type = self.detect_article_type(text)
        
        # Create a structured conversation
        conversation = {
            "messages": [
                {
                    "role": "system",
                    "content": self.create_system_message(article_type)
                },
                {
                    "role": "user",
                    "content": f"Can you explain {title} in detail? I'd like to understand its key concepts and significance."
                },
                {
                    "role": "assistant",
                    "content": self.format_response(text)
                }
            ]
        }
        
        return conversation

    def detect_article_type(self, text: str) -> str:
        """Detect the type of article based on content analysis."""
        text_lower = text.lower()
        
        # Simple heuristic detection - can be made more sophisticated
        if any(word in text_lower for word in ['algorithm', 'computer', 'programming', 'software', 'hardware']):
            return "technical"
        elif any(word in text_lower for word in ['history', 'century', 'empire', 'war', 'revolution']):
            return "historical"
        elif any(word in text_lower for word in ['physics', 'chemistry', 'biology', 'scientist', 'experiment']):
            return "scientific"
        return "default"

    def format_response(self, text: str) -> str:
        """Format the response text in an engaging way."""
        # Extract first paragraph for overview
        paragraphs = text.split('\n\n')
        overview = paragraphs[0] if paragraphs else text[:500]
        
        # Structure the response
        formatted_text = f"{overview}\n\nKey points:\n"
        
        # Add key points from remaining content
        remaining_text = ' '.join(paragraphs[1:]) if len(paragraphs) > 1 else text[500:]
        sentences = [s.strip() for s in remaining_text.split('.') if len(s.strip()) > 20]
        
        # Select important sentences (avoiding redundancy)
        key_points = []
        for sentence in sentences[:5]:  # Limit to 5 key points
            if len(key_points) < 5 and not any(self.similar_content(sentence, kp) for kp in key_points):
                key_points.append(sentence)
        
        # Add key points to response
        for i, point in enumerate(key_points, 1):
            formatted_text += f"\n{i}. {point}."
        
        return formatted_text

    def similar_content(self, text1: str, text2: str, threshold: float = 0.5) -> bool:
        """Simple check for similar content to avoid redundancy."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        overlap = len(words1.intersection(words2)) / len(words1.union(words2))
        return overlap > threshold

    def validate_conversation(self, conversation: Dict) -> bool:
        """Validate conversation for GPT-4 fine-tuning requirements."""
        try:
            messages = conversation['messages']
            
            # Check message structure
            if not all(isinstance(m, dict) and 'role' in m and 'content' in m for m in messages):
                return False
            
            # Check content length
            for message in messages:
                content_length = len(message['content'])
                if not (self.MIN_CONTENT_LENGTH <= content_length <= self.MAX_CONTENT_LENGTH):
                    return False
            
            # Check token count
            total_tokens = sum(len(self.encoding.encode(m['content'])) for m in messages)
            if not (self.MIN_TOKENS_PER_EXAMPLE <= total_tokens <= self.MAX_TOKENS_PER_EXAMPLE):
                return False
            
            return True
            
        except Exception:
            return False

    def process_file(self, input_file: str, output_file: str, max_examples: int = None) -> None:
        """Process and validate Wikipedia data for GPT-4 fine-tuning."""
        valid_conversations = []
        processed = 0
        skipped = 0
        
        logging.info(f"Processing {input_file}...")
        
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                for line in tqdm(f):
                    try:
                        article = json.loads(line)
                        conversation = self.prepare_conversation(article)
                        
                        if self.validate_conversation(conversation):
                            valid_conversations.append(conversation)
                            processed += 1
                        else:
                            skipped += 1
                            
                        if max_examples and processed >= max_examples:
                            break
                            
                    except json.JSONDecodeError:
                        skipped += 1
                        continue
            
            # Shuffle conversations for better training
            random.shuffle(valid_conversations)
            
            # Write processed conversations
            with open(output_file, 'w', encoding='utf-8') as f:
                for conv in valid_conversations:
                    f.write(json.dumps(conv) + '\n')
            
            logging.info(f"Successfully processed conversations: {processed}")
            logging.info(f"Skipped conversations: {skipped}")
            logging.info(f"Output saved to: {output_file}")
            
        except Exception as e:
            logging.error(f"Error processing file: {e}")

def main():
    # Configuration
    input_file = "../Wikipedia_dumps/training_data/wiki_training_data.jsonl"  # Your processed Wikipedia data
    output_file = "../Wikipedia_dumps/training_data/gpt4_training_data.jsonl"  # Output for GPT-4 fine-tuning
    
    # Initialize processor
    processor = GPT4DataPrep()
    
    # Process with a limit of examples (adjust based on your needs)
    # OpenAI recommends 100-50000 examples for GPT-4
    processor.process_file(input_file, output_file, max_examples=10000)

if __name__ == "__main__":
    main()
