import json
import os
import re
from pathlib import Path


def is_biology_related(title, text):
    """
    Check if article is biology-related based on keywords in the title or text.
    """
    keywords = {
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

    # Quick check for keywords in the title
    if any(keyword in title_lower for keyword in keywords):
        return True

    # Check for keywords in the first 1000 characters of text
    return sum(1 for keyword in keywords if keyword in text_lower[:1000]) >= 3


def is_high_quality(text):
    """
    Check if the article is high-quality based on length, clarity, and content.
    """
    # Check for length constraints
    if not (500 <= len(text) <= 5000):
        return False

    # Basic sentence clarity check
    sentences = re.split(r'(?<=[^A-Z].[.?]) +(?=[A-Z])', text)
    if len(sentences) < 5 or any(len(sentence) > 200 for sentence in sentences):
        return False

    # Exclude disambiguation or redirect pages
    return 'disambiguation' not in text.lower() and 'redirect' not in text.lower()


def create_training_example(title, text):
    """
    Create a training example formatted for GPT fine-tuning.
    """
    system_message = (
        "You are a knowledgeable biology tutor. Provide accurate, educational, "
        "and engaging information about biological concepts. Break down complex "
        "topics into understandable explanations."
    )

    # Clean and trim text
    text = text.replace('\n\n', ' ').strip()
    text = text[:2000] + '...' if len(text) > 2000 else text

    return {
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user",
             "content": f"Please explain the concept of {title} in biology. What are its key characteristics and significance?"},
            {"role": "assistant", "content": text}
        ]
    }


def process_wiki_files(base_directory, output_file):
    """
    Process all wiki files in the base directory to extract high-quality biology articles.
    """
    wiki_dir = Path(base_directory)
    biology_articles = []
    total_processed = biology_found = high_quality_found = 0

    for subdir in wiki_dir.iterdir():
        if subdir.is_dir():
            print(f"\nProcessing directory: {subdir.name}")

            for wiki_file in sorted(subdir.glob("wiki_*")):
                print(f"Processing file: {wiki_file.name}")

                with open(wiki_file, 'r', encoding='utf-8') as file:
                    for line in file:
                        total_processed += 1
                        try:
                            article = json.loads(line.strip())
                            title, text = article.get('title', ''), article.get('text', '')

                            if len(text) < 200:
                                continue

                            if is_biology_related(title, text):
                                biology_found += 1

                                if is_high_quality(text):
                                    high_quality_found += 1
                                    biology_articles.append(create_training_example(title, text))

                                if biology_found % 10 == 0:
                                    print(f"Biology articles found: {biology_found} / Processed: {total_processed}")

                        except (json.JSONDecodeError, KeyError):
                            continue

    # Save results to file
    if biology_articles:
        with open(output_file, 'w', encoding='utf-8') as f:
            for article in biology_articles:
                f.write(json.dumps(article) + '\n')
        print(f"\nSaved {len(biology_articles)} high-quality articles to {output_file}")
    else:
        print("\nNo high-quality articles found.")

    print(f"\nTotal articles processed: {total_processed}")
    print(f"Biology articles found: {biology_found}")
    print(f"High-quality articles found: {high_quality_found}")

    return high_quality_found


def validate_directory(directory):
    """
    Validate the base directory structure.
    """
    path = Path(directory)
    if not path.exists():
        raise FileNotFoundError(f"Directory does not exist: {directory}")

    aa_dir = path / "AA"
    if not aa_dir.exists():
        raise FileNotFoundError(f"AA directory not found in: {directory}")

    if not list(aa_dir.glob("wiki_*")):
        raise FileNotFoundError(f"No wiki files found in: {aa_dir}")

    return True


if __name__ == "__main__":
    base_dir = r"C:\\Users\\Rakib\\Documents\\Ai Mentor\\SourceFolder\\OpenAi\\Wikipedia_dumps\\Wiki_extracted_data"
    output_file = "Wikipedia_dumps/training_data_2/biology_training_data_high_quality.jsonl"

    try:
        if validate_directory(base_dir):
            print("Starting processing...")
            num_articles = process_wiki_files(base_dir, output_file)

            if num_articles > 0:
                print(f"\nSuccessfully created training data with {num_articles} high-quality articles.")
            else:
                print("\nNo high-quality biology articles found.")

    except Exception as e:
        print(f"Error: {e}")
