import json
import os
from tqdm import tqdm
import logging
from typing import List, Dict, Set
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class ArticleFilter:
    """Filter configuration for Wikipedia articles."""
    
    def __init__(self):
        # Content length filters
        self.min_length: int = 200  # Minimum text length in characters
        self.max_length: int = 50000  # Maximum text length in characters
        
        # Title filters
        self.title_keywords_include: Set[str] = set()  # If empty, accept all titles
        self.title_keywords_exclude: Set[str] = {
            'disambiguation',
            'template:',
            'category:',
            'file:',
            'portal:'
        }
        
        # Text content filters
        self.content_keywords_include: Set[str] = set()  # If empty, accept all content
        self.content_keywords_exclude: Set[str] = {
            'this article needs additional citations',
            'this article is a stub',
            'this disambiguation page',
            'wikiproject'
        }

    def is_valid_article(self, article: Dict) -> bool:
        """
        Check if an article meets the filtering criteria.
        
        Args:
            article: Dictionary containing article data with 'title' and 'text' fields
        
        Returns:
            bool: True if article meets all criteria, False otherwise
        """
        text = article.get('text', '').lower()
        title = article.get('title', '').lower()
        
        # Check text length
        if not (self.min_length <= len(text) <= self.max_length):
            return False
        
        # Check title exclusions
        if any(excl in title for excl in self.title_keywords_exclude):
            return False
            
        # Check title inclusions if specified
        if self.title_keywords_include and not any(incl in title for incl in self.title_keywords_include):
            return False
        
        # Check content exclusions
        if any(excl in text for excl in self.content_keywords_exclude):
            return False
            
        # Check content inclusions if specified
        if self.content_keywords_include and not any(incl in text for incl in self.content_keywords_include):
            return False
        
        return True

def create_conversation(article: Dict) -> List[Dict]:
    """
    Create conversation format suitable for OpenAI fine-tuning.
    
    Args:
        article: Dictionary containing article data
    
    Returns:
        list: List of conversation dictionaries in OpenAI format
    """
    title = article.get('title', 'Unknown')
    text = article.get('text', '').strip()
    
    # Create multiple conversation patterns for the same article
    conversations = [
        {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a knowledgeable AI mentor helping users learn about various topics."
                },
                {
                    "role": "user",
                    "content": f"What is {title}?"
                },
                {
                    "role": "assistant",
                    "content": text[:1000]  # First 1000 characters for overview
                }
            ]
        },
        {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a knowledgeable AI mentor helping users learn about various topics."
                },
                {
                    "role": "user",
                    "content": f"Can you explain {title} in detail?"
                },
                {
                    "role": "assistant",
                    "content": text
                }
            ]
        }
    ]
    
    return conversations

def process_file(input_file: str, output_file: str, article_filter: ArticleFilter, append: bool = False) -> tuple:
    """
    Process a single JSONL file and filter articles.
    
    Args:
        input_file: Path to input JSONL file
        output_file: Path to output JSONL file
        article_filter: ArticleFilter instance
        append: Whether to append to output file
    
    Returns:
        tuple: (processed_count, filtered_count)
    """
    processed_count = 0
    filtered_count = 0
    
    mode = 'a' if append else 'w'
    
    try:
        with open(input_file, 'r', encoding='utf-8') as infile, \
             open(output_file, mode, encoding='utf-8') as outfile:
            
            for line in infile:
                try:
                    article = json.loads(line.strip())
                    
                    # Apply filters
                    if article_filter.is_valid_article(article):
                        # Create and write conversations
                        conversations = create_conversation(article)
                        for conv in conversations:
                            outfile.write(json.dumps(conv) + '\n')
                        processed_count += 1
                    else:
                        filtered_count += 1
                        
                except json.JSONDecodeError as e:
                    logging.error(f"Error parsing JSON line in {input_file}: {e}")
                    continue
                
    except Exception as e:
        logging.error(f"Error processing file {input_file}: {e}")
    
    return processed_count, filtered_count

def process_directory(input_dir: str, output_file: str, article_filter: ArticleFilter) -> None:
    """
    Process all wiki_XX files in a directory.
    
    Args:
        input_dir: Directory containing wiki_XX files
        output_file: Path to output JSONL file
        article_filter: ArticleFilter instance
    """
    total_processed = 0
    total_filtered = 0
    
    # Get all wiki files in the directory
    wiki_files = sorted([f for f in os.listdir(input_dir) if f.startswith('wiki_')])
    
    logging.info(f"Found {len(wiki_files)} wiki files to process")
    
    # Process each file
    for i, wiki_file in enumerate(tqdm(wiki_files, desc="Processing files")):
        input_file = os.path.join(input_dir, wiki_file)
        processed, filtered = process_file(input_file, output_file, article_filter, append=(i > 0))
        
        total_processed += processed
        total_filtered += filtered
        
        logging.info(f"File {wiki_file}: processed {processed}, filtered {filtered} articles")
    
    logging.info(f"Processing completed. Total processed: {total_processed}, filtered: {total_filtered}")

def main():
    """Main execution function."""
    # Configure paths
    input_directory = "training_data"  # Directory containing wiki_XX files
    output_file = "training_data/filtered_training_data.jsonl"

    # Configure filters
    article_filter = ArticleFilter()
    
    # Example: Set minimum length to 500 characters
    article_filter.min_length = 500
    
    # Example: Include only specific topics
    # article_filter.content_keywords_include = {'science', 'history', 'technology'}
    
    # Example: Exclude additional patterns
    # article_filter.content_keywords_exclude.add('citation needed')
    
    logging.info("Starting JSONL filtering and conversion")
    logging.info(f"Input directory: {input_directory}")
    logging.info(f"Output file: {output_file}")
    
    # Process the directory
    process_directory(input_directory, output_file, article_filter)

if __name__ == "__main__":
    main()
