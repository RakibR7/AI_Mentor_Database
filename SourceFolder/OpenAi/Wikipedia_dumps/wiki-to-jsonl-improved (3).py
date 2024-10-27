import json
import os
from tqdm import tqdm
import logging
from typing import List, Dict, Optional
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('wiki_conversion.log')
    ]
)

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

def process_wiki_file(file_path: str, output_dir: str, batch_size: int = 1000) -> tuple:
    """
    Process a single Wikipedia JSON file.
    
    Args:
        file_path: Path to the wiki file
        output_dir: Directory to save batch files
        batch_size: Number of conversations per batch
    
    Returns:
        tuple: (articles_processed, batches_written)
    """
    articles_processed = 0
    current_batch = []
    batches_written = 0
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    article = json.loads(line.strip())
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
    
    return articles_processed, batches_written

def write_to_jsonl(conversations: List[Dict], output_file: str) -> None:
    """Write conversations to JSONL file."""
    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            for conv in conversations:
                f.write(json.dumps(conv) + '\n')
    except Exception as e:
        logging.error(f"Error writing to file {output_file}: {e}")

def merge_batch_files(batch_dir: str, final_output: str) -> None:
    """Merge all batch files into a single output file."""
    try:
        with open(final_output, 'w', encoding='utf-8') as outfile:
            batch_files = sorted(Path(batch_dir).glob('batch_*.jsonl'))
            for batch_file in tqdm(batch_files, desc="Merging batches"):
                with open(batch_file, 'r', encoding='utf-8') as infile:
                    outfile.write(infile.read())
                os.remove(batch_file)  # Remove batch file after merging
    except Exception as e:
        logging.error(f"Error merging batch files: {e}")

def process_wiki_directory(input_directory: str, output_directory: str) -> None:
    """Process all Wikipedia folders and files."""
    # Create output directories
    batch_dir = os.path.join(output_directory, 'batches')
    os.makedirs(batch_dir, exist_ok=True)
    
    total_articles = 0
    total_batches = 0
    
    # Get all subdirectories (AA, AB, AC, etc.)
    subdirs = sorted([d for d in os.listdir(input_directory) 
                     if os.path.isdir(os.path.join(input_directory, d))])
    
    logging.info(f"Found {len(subdirs)} subdirectories to process")
    
    # Process each subdirectory
    for subdir in tqdm(subdirs, desc="Processing folders"):
        subdir_path = os.path.join(input_directory, subdir)
        wiki_files = sorted([f for f in os.listdir(subdir_path) 
                           if f.startswith('wiki_')])
        
        logging.info(f"Processing {subdir}: {len(wiki_files)} files")
        
        # Process files in parallel
        with ProcessPoolExecutor(max_workers=6) as executor:
            futures = []
            for wiki_file in wiki_files:
                file_path = os.path.join(subdir_path, wiki_file)
                future = executor.submit(process_wiki_file, file_path, batch_dir)
                futures.append(future)
            
            # Collect results
            for future in tqdm(futures, desc=f"Processing {subdir} files"):
                articles, batches = future.result()
                total_articles += articles
                total_batches += batches
    
    logging.info(f"All folders processed. Total articles: {total_articles}")
    logging.info(f"Total batches created: {total_batches}")
    
    # Merge all batch files into final output
    final_output = os.path.join(output_directory, 'wiki_training_data.jsonl')
    logging.info("Merging batch files into final output...")
    merge_batch_files(batch_dir, final_output)
    
    # Clean up batch directory
    os.rmdir(batch_dir)
    
    logging.info(f"Conversion completed. Final output saved to {final_output}")

def main() -> None:
    """Main execution function."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Input directory should be the parent folder containing AA, AB, AC, etc.
    input_directory = os.path.join(script_dir, "Wiki_extracted_data")
    output_directory = os.path.join(script_dir, "training_data")
    
    logging.info("Starting large-scale Wikipedia to JSONL conversion")
    logging.info(f"Input directory: {input_directory}")
    logging.info(f"Output directory: {output_directory}")
    
    process_wiki_directory(input_directory, output_directory)

if __name__ == "__main__":
    main()
