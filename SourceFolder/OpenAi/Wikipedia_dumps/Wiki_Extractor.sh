# Step 1: Clone WikiExtractor
#git clone https://github.com/attardi/wikiextractor.git
#cd wikiextractor

# Step 2: Extract Wikipedia dump
python3 WikiExtractor.py C:\Users\Rakib\Documents\Ai Mentor\SourceFolder\OpenAi\Wikipedia_dumps\enwiki-20240901-pages-articles-multistream.xml.bz2 -o C:\Users\Rakib\Documents\Ai Mentor\SourceFolder\OpenAi\Wikipedia_dumps\Wiki_extracted_data


probably working version 
python -m wikiextractor.WikiExtractor /enwiki-20240901-pages-articles-multistream.xml.bz2 -o /output --json

python -m wikiextractor.WikiExtractor C:\Users\Rakib\Documents\Ai Mentor\SourceFolder\OpenAi\Wikipedia_dumps\enwiki-20240901-pages-articles-multistream.xml.bz2 -o C:\Users\Rakib\Documents\Ai Mentor\SourceFolder\OpenAi\Wikipedia_dumps\output --json

###
python -m wikiextractor.WikiExtractor "C:\Users\Rakib\Documents\Ai Mentor\SourceFolder\OpenAi\Wikipedia_dumps\enwiki-20240901-pages-articles-multistream.xml.bz2" -o "C:\Users\Rakib\Documents\Ai Mentor\SourceFolder\OpenAi\Wikipedia_dumps\Wiki_extracted_data" --json
###


# Step 3: Preprocess using Python (remove templates, normalize, split)
# Custom Python script using libraries like re and mwparserfromhell.

# Step 4: Tokenize
# Use Hugging Face's Tokenizers library.

# Step 5: Format to JSONL
# Custom Python script to convert into JSONL.

# Step 6: Train/Fine-tune OpenAI model
# Load JSONL and begin training.
