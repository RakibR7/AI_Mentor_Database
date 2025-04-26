import json
import tiktoken
from tqdm import tqdm
import logging
from typing import List, Dict
import random
import os

# Set up detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('finetune_prep.log')
    ]
)

class GPT4DataPrep:
    def __init__(self):
        self.encoding = tiktoken.encoding_for_model("gpt-4")
        
        # Adjusted limits
        self.MAX_TOKENS_PER_EXAMPLE = 4096
        self.MIN_TOKENS_PER_EXAMPLE = 10  # Reduced from 50
        
        # Adjusted quality thresholds
        self.MIN_CONTENT_LENGTH = 20    # Reduced from 100
        self.MAX_CONTENT_LENGTH = 15000
        
        # Tracking validation failures
        self.validation_failures = {
            "structure": 0,
            "content_length": 0,
            "token_count": 0,
            "other": 0
        }

    def validate_conversation(self, conversation: Dict) -> tuple[bool, str]:
        """
        Validate conversation for GPT-4 fine-tuning requirements.
        Returns (is_valid, reason_if_invalid)
        """
        try:
            if 'messages' not in conversation:
                return False, "missing_messages"
                
            messages = conversation['messages']
            
            # Check message structure
            if not all(isinstance(m, dict) and 'role' in m and 'content' in m for m in messages):
                self.validation_failures["structure"] += 1
                return False, "invalid_message_structure"
            
            # Check content length for each message
            for message in messages:
                content_length = len(message['content'])
                if not (self.MIN_CONTENT_LENGTH <= content_length <= self.MAX_CONTENT_LENGTH):
                    self.validation_failures["content_length"] += 1
                    return False, f"content_length_{content_length}"
            
            # Check token count
            total_tokens = sum(len(self.encoding.encode(m['content'])) for m in messages)
            if not (self.MIN_TOKENS_PER_EXAMPLE <= total_tokens <= self.MAX_TOKENS_PER_EXAMPLE):
                self.validation_failures["token_count"] += 1
                return False, f"token_count_{total_tokens}"
            
            return True, "valid"
            
        except Exception as e:
            self.validation_failures["other"] += 1
            return False, f"error_{str(e)}"

    def process_file(self, input_file: str, output_file: str, max_examples: int = None) -> None:
        """Process and validate conversations for GPT-4 fine-tuning."""
        if not os.path.exists(input_file):
            logging.error(f"Input file not found: {input_file}")
            return
            
        valid_conversations = []
        processed = 0
        skipped = 0
        validation_reasons = {}
        
        logging.info(f"Processing {input_file}...")
        logging.info(f"Current working directory: {os.getcwd()}")
        
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(tqdm(f), 1):
                    try:
                        conversation = json.loads(line)
                        
                        # Print first conversation for debugging
                        if line_num == 1:
                            logging.info("First conversation structure:")
                            logging.info(json.dumps(conversation, indent=2))
                        
                        is_valid, reason = self.validate_conversation(conversation)
                        
                        if is_valid:
                            valid_conversations.append(conversation)
                            processed += 1
                        else:
                            skipped += 1
                            validation_reasons[reason] = validation_reasons.get(reason, 0) + 1
                            
                            # Log detailed information for first few failures
                            if skipped <= 5:
                                logging.info(f"Skipped conversation {skipped} due to: {reason}")
                                logging.info(f"Conversation structure: {json.dumps(conversation, indent=2)}")
                        
                        if max_examples and processed >= max_examples:
                            break
                            
                    except json.JSONDecodeError as e:
                        skipped += 1
                        logging.error(f"JSON decode error at line {line_num}: {e}")
                        continue
            
            # Shuffle valid conversations
            random.shuffle(valid_conversations)
            
            # Write processed conversations
            with open(output_file, 'w', encoding='utf-8') as f:
                for conv in valid_conversations:
                    f.write(json.dumps(conv) + '\n')
            
            # Log detailed statistics
            logging.info("\n=== Processing Statistics ===")
            logging.info(f"Total conversations processed: {processed + skipped}")
            logging.info(f"Successfully processed: {processed}")
            logging.info(f"Skipped: {skipped}")
            logging.info("\nValidation Failure Reasons:")
            for reason, count in validation_reasons.items():
                logging.info(f"- {reason}: {count}")
            logging.info("\nValidation Failure Categories:")
            for category, count in self.validation_failures.items():
                logging.info(f"- {category}: {count}")
            logging.info(f"\nOutput saved to: {output_file}")
            
        except Exception as e:
            logging.error(f"Error processing file: {e}")

def main():
    # Configuration
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(script_dir, "training_data", "validated_training_data.jsonl")
    output_file = os.path.join(script_dir, "training_data", "gpt4_training_data.jsonl")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Initialize processor
    processor = GPT4DataPrep()
    
    # Process files with detailed logging
    processor.process_file(input_file, output_file, max_examples=None)

if __name__ == "__main__":
    main()
