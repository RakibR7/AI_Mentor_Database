import json
import os
import re
from pathlib import Path


def is_biology_related(title, text):
    """
    Check if article is biology related based on keywords and content.
    """
    biology_keywords = {
        'biology', 'species', 'cell', 'organism', 'gene', 'evolution',
        'protein', 'bacteria', 'virus', 'animal', 'plant', 'tissue',
        'chromosome', 'dna', 'rna', 'enzyme', 'molecular', 'ecology',
        'taxonomy', 'biodiversity', 'ecosystem', 'reproduction',
        'metabolism', 'microbe', 'fungi', 'algae', 'amphibian',
        'anatomy', 'physiology', 'heredity', 'genetics', 'botany',
        'zoology', 'microbiology', 'embryology', 'immune', 'hormone',
        'neuron', 'biosphere', 'prokaryote', 'eukaryote', 'photosynthesis',
        'respiration', 'mammal', 'bird', 'fish', 'insect', 'invertebrate',
        'vertebrate', 'population', 'symbiosis', 'parasite'
    }

    title_lower = title.lower()
    text_lower = text.lower()

    if any(keyword in title_lower for keyword in biology_keywords):
        return True

    first_section = text_lower[:1000]
    keyword_count = sum(1 for keyword in biology_keywords if keyword in first_section)

    return keyword_count >= 3


def is_high_quality(text):
    """
    Check if the article is of high quality based on length, clarity, and content.
    """
    # Check article length
    if len(text) < 500 or len(text) > 5000:
        return False

    # Check for clarity and readability
    sentences = re.split(r'(?<=[^A-Z].[.?]) +(?=[A-Z])', text)
    if len(sentences) < 5 or any(len(sentence) > 200 for sentence in sentences):
        return False

    # Check for content quality
    if 'disambiguation' in text.lower() or 'redirect' in text.lower():
        return False

    return True


def create_training_example(title, text):
    """
    Create a training example in the format needed for GPT fine-tuning.
    """
    system_message = ("You are a knowledgeable biology tutor. Provide accurate, educational, "
                      "and engaging information about biological concepts. Break down complex "
                      "topics into understandable explanations.")

    # Clean and format the text
    text = text.replace('\n\n', ' ').strip()
    if len(text) > 2000:
        text = text[:2000] + '...'

    messages = [
        {
            "role": "system",
            "content": system_message
        },
        {
            "role": "user",
            "content": f"Please explain the concept of {title} in biology. What are its key characteristics and significance?"
        },
        {
            "role": "assistant",
            "content": text
        }
    ]

    return {"messages": messages}


def process_wiki_files(base_directory, output_file):
    """
    Process all wiki files in the subdirectories of the base directory and create biology training data.
    """
    biology_articles = []
    wiki_dir = Path(base_directory)

    # Counter for logging
    total_processed = 0
    biology_found = 0
    high_quality_found = 0

    print(f"Looking for files in: {wiki_dir}")

    try:
        # Process each subdirectory in the base directory
        for subdir in wiki_dir.iterdir():
            if subdir.is_dir():
                print(f"\nProcessing subdirectory: {subdir.name}")

                # Process each wiki file in the subdirectory
                for wiki_file in sorted(subdir.glob("wiki_*")):
                    print(f"Processing {wiki_file.name}...")

                    try:
                        with open(wiki_file, 'r', encoding='utf-8') as file:
                            for line in file:
                                total_processed += 1

                                try:
                                    article = json.loads(line.strip())

                                    # Skip empty or very short articles
                                    if not article['text'] or len(article['text']) < 200:
                                        continue

                                    # Check if article is biology related
                                    if is_biology_related(article['title'], article['text']):
                                        biology_found += 1

                                        # Check if article is of high quality
                                        if is_high_quality(article['text']):
                                            high_quality_found += 1
                                            training_example = create_training_example(
                                                article['title'],
                                                article['text']
                                            )
                                            biology_articles.append(training_example)

                                        # Log progress periodically
                                        if biology_found % 10 == 0:
                                            print(f"Found {biology_found} biology articles out of {total_processed} processed")
                                            print(f"Found {high_quality_found} high-quality biology articles")

                                except json.JSONDecodeError:
                                    continue
                                except KeyError:
                                    continue

                    except Exception as e:
                        print(f"Error processing file {wiki_file.name}: {str(e)}")
                        continue

        # Write high-quality biology articles to JSONL file
        if biology_articles:
            print(f"\nWriting {len(biology_articles)} high-quality articles to {output_file}...")
            with open(output_file, 'w', encoding='utf-8') as f:
                for article in biology_articles:
                    f.write(json.dumps(article) + '\n')
            print("Writing complete!")
        else:
            print("\nNo high-quality biology articles found!")

        print(f"\nFinal Statistics:")
        print(f"Total articles processed: {total_processed}")
        print(f"Biology articles found: {biology_found}")
        print(f"High-quality biology articles found: {high_quality_found}")
        print(f"Output saved to: {output_file}")

    except Exception as e:
        print(f"Critical error encountered: {str(e)}")

    return high_quality_found


def validate_directory(directory):
    """
    Validate that the directory exists and contains wiki files.
    """
    path = Path(directory)
    if not path.exists():
        raise ValueError(f"Directory does not exist: {directory}")

    aa_dir = path / "AA"
    if not aa_dir.exists():
        raise ValueError(f"AA directory not found in: {directory}")

    wiki_files = list(aa_dir.glob("wiki_*"))
    if not wiki_files:
        raise ValueError(f"No wiki files found in: {aa_dir}")

    return True


if __name__ == "__main__":
    # Specify your base directory where the Wiki_extracted_data folder is located
    base_dir = r"C:\Users\Rakib\Documents\Ai Mentor\SourceFolder\OpenAi\Wikipedia_dumps\Wiki_extracted_data"
    output_file = "Wikipedia_dumps/training_data_2/biology_training_data_high_quality.jsonl"

    try:
        # Validate directory structure before processing
        if validate_directory(base_dir):
            print(f"Starting processing of wiki files...")
            num_articles = process_wiki_files(base_dir, output_file)

            if num_articles > 0:
                print(f"\nSuccessfully created training data with {num_articles} high-quality biology articles!")
                print(f"Output file: {output_file}")
            else:
                print("\nNo high-quality biology articles were found in the processed files.")

    except Exception as e:
        print(f"Error: {str(e)}")
        print("Please check your directory structure and try again.")