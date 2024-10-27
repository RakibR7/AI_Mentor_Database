import json
import os
from tqdm import tqdm
import re
import logging
from typing import List, Dict, Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def clean_text(text: str) -> str:
    """Clean and format Wikipedia text."""
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\[\d+]', '', text)
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
    try:
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        with open(output_file, mode, encoding='utf-8') as f:
            for conv in conversations:
                f.write(json.dumps(conv) + '\n')
    except Exception as e:
        logging.error(f"Error writing to file {output_file}: {e}")

def verify_directory(directory: str) -> bool:
    """Verify if directory exists and contains .txt files."""
    if not os.path.exists(directory):
        logging.error(f"Directory does not exist: {directory}")
        return False
    
    if not os.path.isdir(directory):
        logging.error(f"Path exists but is not a directory: {directory}")
        return False
    
    txt_files = []
    for root, _, files in os.walk(directory):
        txt_files.extend([f for f in files if f.endswith('.txt')])
        if txt_files:
            logging.info(f"Found {len(txt_files)} .txt files in: {root}")
            return True
    
    logging.error(f"No .txt files found in directory: {directory}")
    return False

def process_wiki_files(input_directory: str, output_file: str, max_files: Optional[int] = None) -> None:
    """Process Wikipedia extracted files and convert to JSONL format."""
    logging.info(f"Starting processing with input directory: {input_directory}")
    logging.info(f"Output will be saved to: {output_file}")
    
    # Convert to absolute path for better error reporting
    input_directory = os.path.abspath(input_directory)
    
    if not verify_directory(input_directory):
        logging.error("Directory verification failed. Exiting.")
        return
    
    all_conversations = []
    processed_files = 0
    total_articles = 0
    
    for root, _, files in tqdm(os.walk(input_directory)):
        txt_files = [f for f in files if f.endswith('.txt')]
        
        for file in txt_files:
            file_path = os.path.join(root, file)
            logging.info(f"Processing file: {file_path}")
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                articles = content.split('<doc id=')
                logging.info(f"Found {len(articles)-1} articles in {file_path}")
                
                for article in articles[1:]:
                    title_match = re.search(r'title="([^"]+)"', article)
                    title = title_match.group(1) if title_match else "Unknown"
                    
                    content = article.split('>\n', 1)[1].split('</doc>')[0].strip()
                    
                    article_conversations = create_conversation(title, content)
                    all_conversations.extend(article_conversations)
                    total_articles += 1
                    
                    if len(all_conversations) >= 1000:
                        write_to_jsonl(all_conversations, output_file, append=True)
                        all_conversations = []
                        
            except Exception as e:
                logging.error(f"Error processing file {file_path}: {e}")
                continue
            
            processed_files += 1
            if max_files and processed_files >= max_files:
                break
    
    if all_conversations:
        write_to_jsonl(all_conversations, output_file, append=True)

def main() -> None:
    """Main execution function."""
    # Get the current script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Navigate to Wiki_extracted_data directory
    input_directory = os.path.join(script_dir, "Wiki_extracted_data")
    output_file = os.path.join(script_dir, "wiki_training_data.jsonl")
    
    logging.info("Starting Wikipedia to JSONL conversion")
    logging.info(f"Input directory: {input_directory}")
    logging.info(f"Output file: {output_file}")
    
    process_wiki_files(input_directory, output_file)

if __name__ == "__main__":
    main()
