#Displays token cost of training



import os
import json
import tiktoken
from tqdm import tqdm
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO)

def validate_and_prepare_data(input_file: str, output_file: str, model_name: str = "gpt-3.5-turbo"):
    """
    Validate and prepare the JSONL data for OpenAI fine-tuning.
    
    Args:
        input_file: Path to input JSONL file
        output_file: Path to output validated JSONL file
        model_name: OpenAI model name for token counting
    """
    # Initialize tokenizer
    encoding = tiktoken.encoding_for_model(model_name)
    
    valid_conversations = []
    total_tokens = 0
    skipped = 0
    
    logging.info(f"Processing {input_file}...")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            try:
                conversation = json.loads(line)
                
                # Validate message structure
                if 'messages' not in conversation:
                    skipped += 1
                    continue
                    
                messages = conversation['messages']
                if not all(isinstance(m, dict) and 'role' in m and 'content' in m 
                          for m in messages):
                    skipped += 1
                    continue
                
                # Count tokens
                conv_tokens = 0
                for message in messages:
                    conv_tokens += len(encoding.encode(message['content']))
                    conv_tokens += len(encoding.encode(message['role']))
                
                # OpenAI's limits (adjust these based on your model)
                if conv_tokens > 4096:  # GPT-3.5 token limit
                    skipped += 1
                    continue
                
                total_tokens += conv_tokens
                valid_conversations.append(conversation)
                
            except json.JSONDecodeError:
                skipped += 1
                continue
    
    # Write validated conversations
    with open(output_file, 'w', encoding='utf-8') as f:
        for conv in valid_conversations:
            f.write(json.dumps(conv) + '\n')
    
    logging.info(f"Processed conversations: {len(valid_conversations)}")
    logging.info(f"Skipped conversations: {skipped}")
    logging.info(f"Total tokens: {total_tokens}")
    logging.info(f"Estimated cost (using $0.03/1K tokens): ${(total_tokens/1000) * 0.03:.2f}")
    
    return len(valid_conversations), total_tokens

def main():
    # Configuration
    input_file = "../Wikipedia_dumps/training_data/wiki_training_data.jsonl"  # Your processed Wikipedia data
    output_file = "../Wikipedia_dumps/training_data/validated_training_data.jsonl"
    
    # Process and validate the data
    num_conversations, total_tokens = validate_and_prepare_data(input_file, output_file)

if __name__ == "__main__":
    main()
