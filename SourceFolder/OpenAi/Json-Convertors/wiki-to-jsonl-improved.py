import json
import os
from tqdm import tqdm
import re

def clean_text(text: str) -> str:
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\[\d+]', '', text)
    text = re.sub(r'\[[^]]*]', '', text)
    return text.strip()

def create_conversation(title: str, content: str) -> list:
    """
    Create a conversation format from Wikipedia article.
    
    Args:
        title (str): Article title
        content (str): Article content
    
    Returns:
        list: List of conversation dictionaries
    """
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

def write_to_jsonl(conversations: list, output_file: str, append: bool = False) -> None:
    """
    Write conversations to JSONL file.
    
    Args:
        conversations (list): List of conversation dictionaries
        output_file (str): Path to output file
        append (bool): Whether to append to existing file
    """
    mode = 'a' if append else 'w'
    with open(output_file, mode, encoding='utf-8') as f:
        for conv in conversations:
            f.write(json.dumps(conv) + '\n')

def process_wiki_files(input_directory: str, output_file: str, max_files: int = None) -> None:
    """
    Process Wikipedia extracted files and convert to JSONL format.
    
    Args:
        input_directory (str): Path to directory containing Wiki-extractor output
        output_file (str): Path to output JSONL file
        max_files (int, optional): Maximum number of files to process
    """
    all_conversations = []
    processed_files = 0
    
    # Walk through all files in the directory
    for root, _, files in tqdm(os.walk(input_directory)):
        for file in files:
            if not file.endswith('.txt'):
                continue
                
            file_path = os.path.join(root, file)
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Split content into articles
                articles = content.split('<doc id=')
                
                for article in articles[1:]:  # Skip first empty split
                    title_match = re.search(r'title="([^"]+)"', article)
                    title = title_match.group(1) if title_match else "Unknown"
                    
                    # Extract main content
                    content = article.split('>\n', 1)[1].split('</doc>')[0].strip()
                    
                    # Create conversations for this article
                    article_conversations = create_conversation(title, content)
                    all_conversations.extend(article_conversations)
                    
                    # Write to file if buffer is large enough
                    if len(all_conversations) >= 1000:
                        write_to_jsonl(all_conversations, output_file, append=True)
                        all_conversations = []
                        
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
                continue
            
            processed_files += 1
            if max_files and processed_files >= max_files:
                break
    
    # Write remaining conversations
    if all_conversations:
        write_to_jsonl(all_conversations, output_file, append=True)

def main() -> None:
    """Main execution function."""
    # Configuration
    input_directory = "C:/Users/Rakib/Documents/Ai Mentor/SourceFolder/OpenAi/Wikipedia_dumps/Wiki_extracted_data/AA"  # Replace with your Wiki-extractor output path
    output_file = "wiki_training_data.jsonl"
    max_files = None  # Set to a number if you want to limit processing
    
    # Process files
    process_wiki_files(input_directory, output_file, max_files)
    
    print(f"Conversion completed. Output saved to {output_file}")

if __name__ == "__main__":
    main()
