import json
import os
import tiktoken
from tqdm import tqdm
import logging
from typing import Dict, List
import math

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class UploadPreparation:
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        self.encoding = tiktoken.encoding_for_model(model_name)
        self.max_file_size = 100 * 1024 * 1024  # 100MB limit for web upload
        self.max_tokens_per_example = 4096  # Token limit for GPT-3.5-turbo

    def validate_conversation(self, conversation: Dict) -> bool:
        """Validate a single conversation."""
        try:
            messages = conversation.get('messages', [])
            
            # Check message structure
            if not messages or not all(isinstance(m, dict) for m in messages):
                return False
                
            # Check required fields
            if not all('role' in m and 'content' in m for m in messages):
                return False
                
            # Check roles are valid
            valid_roles = {'system', 'user', 'assistant'}
            if not all(m['role'] in valid_roles for m in messages):
                return False
                
            # Count tokens
            total_tokens = 0
            for message in messages:
                total_tokens += len(self.encoding.encode(message['content']))
                total_tokens += len(self.encoding.encode(message['role']))
                
            # Check token limit
            if total_tokens > self.max_tokens_per_example:
                return False
                
            return True
            
        except Exception as e:
            logging.error(f"Validation error: {e}")
            return False

    def prepare_files(self, input_file: str, output_dir: str, max_examples_per_file: int = 10000):
        """Prepare and split data into uploadable files."""
        os.makedirs(output_dir, exist_ok=True)
        
        valid_conversations = []
        total_tokens = 0
        skipped = 0
        file_count = 0
        current_file_size = 0
        
        logging.info(f"Processing {input_file}...")
        
        # Read and validate all conversations
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Validating conversations"):
                try:
                    conversation = json.loads(line)
                    if self.validate_conversation(conversation):
                        valid_conversations.append(conversation)
                        
                        # Count tokens for cost estimation
                        conv_tokens = sum(len(self.encoding.encode(m['content'])) + 
                                       len(self.encoding.encode(m['role']))
                                       for m in conversation['messages'])
                        total_tokens += conv_tokens
                    else:
                        skipped += 1
                        
                except json.JSONDecodeError:
                    skipped += 1
                    continue
                    
                # Split into multiple files if needed
                if len(valid_conversations) >= max_examples_per_file:
                    self.write_conversations_to_file(
                        valid_conversations, 
                        output_dir, 
                        file_count
                    )
                    file_count += 1
                    valid_conversations = []
        
        # Write remaining conversations
        if valid_conversations:
            self.write_conversations_to_file(
                valid_conversations, 
                output_dir, 
                file_count
            )
            file_count += 1
        
        # Print summary
        logging.info(f"\nProcessing complete!")
        logging.info(f"Total valid conversations: {len(valid_conversations)}")
        logging.info(f"Skipped conversations: {skipped}")
        logging.info(f"Total tokens: {total_tokens:,}")
        logging.info(f"Estimated training cost (at $0.03/1K tokens): ${(total_tokens/1000 * 0.03):.2f}")
        logging.info(f"Files created: {file_count}")
        logging.info("\nNext steps:")
        logging.info("1. Go to https://platform.openai.com/finetune")
        logging.info("2. Click 'Create new fine-tune'")
        logging.info("3. Upload the prepared JSONL file(s)")
        logging.info("4. Select gpt-3.5-turbo as your model")
        logging.info("5. Configure your fine-tuning parameters")
        logging.info("6. Start training")

    def write_conversations_to_file(self, conversations: List[Dict], output_dir: str, file_count: int):
        """Write conversations to a JSONL file."""
        output_file = os.path.join(output_dir, f'upload_ready_{file_count}.jsonl')
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for conv in conversations:
                f.write(json.dumps(conv) + '\n')
        
        file_size = os.path.getsize(output_file)
        if file_size > self.max_file_size:
            logging.warning(f"File {output_file} is larger than 100MB and may need to be split further")

def main():
    # Configuration
    input_file = "../Wikipedia_dumps/training_data/wiki_training_data.jsonl"  # Your processed Wikipedia data
    output_dir = "upload_ready_data"
    
    # Initialize and run preparation
    prep = UploadPreparation()
    prep.prepare_files(input_file, output_dir)

if __name__ == "__main__":
    main()
