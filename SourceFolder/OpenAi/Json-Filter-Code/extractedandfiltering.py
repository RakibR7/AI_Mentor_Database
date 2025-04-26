import json
import os
from tqdm import tqdm
import logging
from typing import List, Dict, Optional, Set
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from pathlib import Path
import re

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('../Json-Convertors/wiki_conversion.log')
    ]
)

class ArticleFilter:
    """Filter configuration and methods for Wikipedia articles."""
    
    def __init__(self):
        # Minimum article length (in characters)
        self.min_length: int = 500
        
        # Maximum article length (in characters)
        self.max_length: int = 100000
        
        # Categories to include (empty set means all categories)
        self.include_categories: Set[str] = set()
        
        # Categories to exclude
        self.exclude_categories: Set[str] = {
            'disambiguation',
            'stub',
            'template',
            'redirect'
        }
        
        # Required keywords (empty set means no required keywords)
        self.required_keywords: Set[str] = set()
        
        # Excluded keywords
        self.excluded_keywords: Set[str] = {
            'This article needs additional citations',
            'This article is a stub',
            'This disambiguation page'
        }
        
        # Topic filters (add your specific topics here)
        self.topic_filters: Dict[str, List[str]] = {
            'science': [
                'physics', 'chemistry', 'biology', 'astronomy',
                'mathematics', 'computer science', 'technology'
            ],
            'history': [
                'history', 'historical', 'ancient', 'medieval',
                'modern history', 'civilization'
            ],
            'arts': [
                'art', 'music', 'literature', 'painting',
                'sculpture', 'architecture', 'film'
            ],
            # Add more topics as needed
        }
        
        # Active topic filter (empty string means no topic filtering)
        self.active_topic: str = ""

    def is_valid_article(self, article: Dict) -> bool:
        """
        Check if an article meets all filtering criteria.
        
        Args:
            article: Dictionary containing article data
            
        Returns:
            bool: True if article meets all criteria, False otherwise
        """
        text = article.get('text', '')
        title = article.get('title', '')
        
        # Check article length
        if not (self.min_length <= len(text) <= self.max_length):
            return False
        
        # Check for excluded keywords in title or text
        for keyword in self.excluded_keywords:
            if keyword.lower() in title.lower() or keyword.lower() in text.lower():
                return False
        
        # Check for required keywords
        if self.required_keywords:
            if not any(keyword.lower() in text.lower() for keyword in self.required_keywords):
                return False
        
        # Check for excluded categories
        for category in self.exclude_categories:
            if category.lower() in title.lower() or category.lower() in text.lower():
                return False
        
        # Check for required categories
        if self.include_categories:
            if not any(category.lower() in text.lower() for category in self.include_categories):
                return False
        
        # Check topic filter
        if self.active_topic:
            topic_keywords = self.topic_filters.get(self.active_topic, [])
            if not any(keyword.lower() in text.lower() or keyword.lower() in title.lower() 
                      for keyword in topic_keywords):
                return False
        
        return True

def create_conversation(article: Dict) -> List[Dict]:
    """Create conversation format from Wikipedia article."""
    title = article.get('title', 'Unknown')
    text = article.get('text', '').strip()
    
    conversations = [
        {
            "messages": [
                {"role": "system", "content": "You are a knowledgeable AI mentor helping users learn about various topics."},
                {"role": "user", "content": f"What is {title}?"},
                {"role": "assistant", "content": text[:500]}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are a knowledgeable AI mentor helping users learn about various topics."},
                {"role": "user", "content": f"Can you explain {title} in detail?"},
                {"role": "assistant", "content": text}
            ]
        }
    ]
    
    return conversations

def process_wiki_file(file_path: str, output_dir: str, article_filter: ArticleFilter, 
                     batch_size: int = 1000) -> tuple:
    """
    Process a single Wikipedia JSON file with filtering.
    
    Args:
        file_path: Path to the wiki file
        output_dir: Directory to save batch files
        article_filter: ArticleFilter instance for filtering articles
        batch_size: Number of conversations per batch
    
    Returns:
        tuple: (articles_processed, articles_filtered, batches_written)
    """
    articles_processed = 0
    articles_filtered = 0
    current_batch = []
    batches_written = 0
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    article = json.loads(line.strip())
                    
                    # Apply filters
                    if not article_filter.is_valid_article(article):
                        articles_filtered += 1
                        continue
                    
                    conversations = create_conversation(article)
                    current_batch.extend(conversations)
                    articles_processed += 1
                    
                    # Write batch if we've reached batch_size
                    if len(current_batch) >= batch_size:
                        batch_file = os.path.join(
                            output_dir,
                            f'batch_{os.path.basename(file_path)}_{batches_written}.jsonl'
                        )
                        write_to_jsonl(current_batch, batch_file)
                        batches_written += 1
                        current_batch = []
                        
                except json.JSONDecodeError as e:
                    logging.error(f"Error parsing JSON in {file_path}: {e}")
                    continue
                
        # Write remaining conversations
        if current_batch:
            batch_file = os.path.join(
                output_dir,
                f'batch_{os.path.basename(file_path)}_{batches_written}.jsonl'
            )
            write_to_jsonl(current_batch, batch_file)
            batches_written += 1
            
    except Exception as e:
        logging.error(f"Error processing file {file_path}: {e}")
    
    return articles_processed, articles_filtered, batches_written

def write_to_jsonl(conversations: List[Dict], output_file: str) -> None:
    """Write conversations to JSONL file."""
    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            for conv in conversations:
                f.write(json.dumps(conv) + '\n')
    except Exception as e:
        logging.error(f"Error writing to file {output_file}: {e}")

def process_wiki_directory(input_directory: str, output_directory: str, article_filter: ArticleFilter) -> None:
    """Process all Wikipedia folders and files with filtering."""
    batch_dir = os.path.join(output_directory, 'batches')
    os.makedirs(batch_dir, exist_ok=True)
    
    total_articles_processed = 0
    total_articles_filtered = 0
    total_batches = 0
    
    subdirs = sorted([d for d in os.listdir(input_directory) 
                     if os.path.isdir(os.path.join(input_directory, d))])
    
    logging.info(f"Found {len(subdirs)} subdirectories to process")
    
    for subdir in tqdm(subdirs, desc="Processing folders"):
        subdir_path = os.path.join(input_directory, subdir)
        wiki_files = sorted([f for f in os.listdir(subdir_path) 
                           if f.startswith('wiki_')])
        
        logging.info(f"Processing {subdir}: {len(wiki_files)} files")
        
        with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            futures = []
            for wiki_file in wiki_files:
                file_path = os.path.join(subdir_path, wiki_file)
                future = executor.submit(process_wiki_file, file_path, batch_dir, article_filter)
                futures.append(future)
            
            for future in tqdm(futures, desc=f"Processing {subdir} files"):
                processed, filtered, batches = future.result()
                total_articles_processed += processed
                total_articles_filtered += filtered
                total_batches += batches
    
    logging.info(f"Articles processed: {total_articles_processed}")
    logging.info(f"Articles filtered out: {total_articles_filtered}")
    logging.info(f"Total batches created: {total_batches}")
    
    # Merge all batch files into final output
    final_output = os.path.join(output_directory, 'wiki_training_data.jsonl')
    logging.info("Merging batch files...")
    
    with open(final_output, 'w', encoding='utf-8') as outfile:
        batch_files = sorted(Path(batch_dir).glob('batch_*.jsonl'))
        for batch_file in tqdm(batch_files, desc="Merging batches"):
            with open(batch_file, 'r', encoding='utf-8') as infile:
                outfile.write(infile.read())
            os.remove(batch_file)
    
    os.rmdir(batch_dir)
    logging.info(f"Conversion completed. Final output saved to {final_output}")

def main() -> None:
    """Main execution function."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_directory = os.path.join(script_dir, "Wiki_extracted_data")
    output_directory = os.path.join(script_dir, "training_data")
    
    # Configure article filtering
    article_filter = ArticleFilter()
    
    # Example: Set up specific filters
    article_filter.min_length = 1000  # Minimum 1000 characters
    article_filter.max_length = 50000  # Maximum 50000 characters
    
    # Example: Filter for science topics only
    # article_filter.active_topic = "science"
    
    # Example: Add required categories
    # article_filter.include_categories = {"history", "biography"}
    
    # Example: Add additional excluded keywords
    # article_filter.excluded_keywords.add("needs update")
    
    logging.info("Starting Wikipedia to JSONL conversion with filtering")
    logging.info(f"Input directory: {input_directory}")
    logging.info(f"Output directory: {output_directory}")
    
    process_wiki_directory(input_directory, output_directory, article_filter)

if __name__ == "__main__":
    main()
