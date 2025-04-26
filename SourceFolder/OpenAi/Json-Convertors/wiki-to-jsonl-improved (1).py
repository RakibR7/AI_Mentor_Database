import json
import os
from tqdm import tqdm
import re
import logging
from typing import List, Dict, Optional

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def clean_text(text: str) -> str:
    """Clean and format Wikipedia text."""
    # Remove multiple newlines and spaces
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    # Remove references [1], [2], etc.
    text = re.sub(r'\[\d+]', '', text)
    # Remove remaining square brackets content
    text = re.sub(r'\[[^]]*]', '', text)
    return text.strip()

def create_conversation(title: str, content: str) -> List[Dict]:
    """Create a conversation format from Wikipedia article."""
    cleaned_content = clean_text(content)
    
    conversations = [
        {
            "messages": [
                {"role": "system", "content": "You are a knowledgeable AI mentor helping users learn about various topics."},
                {"role": "user", "content": f"What is {title}?"},
                {"role": "assistant", "content": cleaned_content[:500]}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are a knowledgeable AI mentor helping users learn about various topics."},
                {"role": "user", "content": f"Can you explain {title} in detail?"},
                {"role": "assistant", "content": cleaned_content}
            ]
        }
    ]
    
    return conversations

def write_to_jsonl(conversations: List[Dict], output_file: str, append: bool = False) -> None:
    """Write conversations to JSONL file."""
    mode = 'a' if append else 'w'
    with open(output_file, mode, encoding='utf-8') as f:
        for conv in conversations:
            f.write(json.dumps(conv) + '\n')

def verify_directory(directory: str) -> bool:
    """
    Verify if directory exists and contains .txt files.
    
    Args:
        directory (str): Path to check
    Returns:
        bool: True if directory is valid and contains .txt files
    """
    if not os.path.exists(directory):
        logging.error(f"Directory does not exist: {directory}")
        return False
    
    has_txt_files = False
    for root, _, files in os.walk(directory):
        if any(f.endswith('.txt') for f in files):
            has_txt_files = True
            logging.info(f"Found .txt files in: {root}")
            return True
    
    if not has_txt_files:
        logging.error(f"No .txt files found in directory: {directory}")
    return has_txt_files

def process_wiki_files(input_directory: str, output_file: str, max_files: Optional[int] = None) -> None:
    """Process Wikipedia extracted files and convert to JSONL format."""
    logging.info(f"Starting processing with input directory: {input_directory}")
    logging.info(f"Output will be saved to: {output_file}")
    
    if not verify_directory(input_directory):
        logging.error("Directory verification failed. Exiting.")
        return
    
    all_conversations = []
    processed_files = 0
    total_articles = 0
    
    # Walk through all files in the directory
    for root, _, files in tqdm(os.walk(input_directory)):
        txt_files = [f for f in files if f.endswith('.txt')]
        logging.info(f"Found {len(txt_files)} .txt files in {root}")
        
        for file in txt_files:
            file_path = os.path.join(root, file)
            logging.info(f"Processing file: {file_path}")
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Split content into articles
                articles = content.split('<doc id=')
                article_count = len(articles) - 1  # Subtract 1 for the initial empty split
                logging.info(f"Found {article_count} articles in {file_path}")
                
                for article in articles[1:]:  # Skip first empty split
                    title_match = re.search(r'title="([^"]+)"', article)
                    title = title_match.group(1) if title_match else "Unknown"
                    
                    # Extract main content
                    content = article.split('>\n', 1)[1].split('</doc>')[0].strip()
                    
                    # Create conversations for this article
                    article_conversations = create_conversation(title, content)
                    all_conversations.extend(article_conversations)
                    total_articles += 1
                    
                    # Write to file if buffer is large enough
                    if len(all_conversations) >= 1000:
                        write_to_jsonl(all_conversations, output_file, append=True)
                        logging.info(f"Wrote batch of {len(all_conversations)} conversations to file")
                        all_conversations = []
                        
            except Exception as e:
                logging.error(f"Error processing file {file_path}: {e}")
                continue
            
            processed_files += 1
            if max_files and processed_files >= max_files:
                logging.info(f"Reached max_files limit of {max_files}")
                break
    
    # Write remaining conversations
    if all_conversations:
        write_to_jsonl(all_conversations, output_file, append=True)
        logging.info(f"Wrote final batch of {len(all_conversations)} conversations to file")
    
    logging.info(f"Processing completed. Total files processed: {processed_files}")
    logging.info(f"Total articles processed: {total_articles}")

def main() -> None:
    """Main execution function."""
    # Configuration
    input_directory = os.path.join("SourceFolder", "OpenAI", "Wikipedia_dumps", "Wiki_extracted_data")
    output_file = "wiki_training_data.jsonl"
    max_files = None  # Set to a number if you want to limit processing
    
    logging.info("Starting Wikipedia to JSONL conversion")
    logging.info(f"Input directory: {os.path.abspath(input_directory)}")
    
    # Process files
    process_wiki_files(input_directory, output_file, max_files)
    
    if os.path.exists(output_file):
        file_size = os.path.getsize(output_file)
        logging.info(f"Conversion completed. Output saved to {output_file} (Size: {file_size/1024/1024:.2f} MB)")
    else:
        logging.error(f"Output file was not created: {output_file}")

if __name__ == "__main__":
    main()
