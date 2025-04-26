import json
import asyncio
from pathlib import Path
import re
from typing import Dict, Set, Generator, List, Tuple
import logging
from dataclasses import dataclass
import aiofiles
import nltk
from nltk.tokenize import sent_tokenize
import spacy

nltk.download('punkt')

@dataclass
class BiologyPatterns:
    SUBJECTS = {
        'molecular_biology': {
            'primary': {
                'dna', 'rna', 'protein', 'gene', 'genome', 'chromosome', 'nucleic acid',
                'transcription', 'translation', 'replication'
            },
            'secondary': {
                'enzyme', 'mutation', 'nucleotide', 'amino acid', 'helicase', 'polymerase',
                'ribosome', 'codon', 'genetic code'
            }
        },
        'cell_biology': {
            'primary': {
                'cell', 'membrane', 'nucleus', 'mitochondria', 'organelle', 'cytoplasm',
                'golgi', 'endoplasmic reticulum'
            },
            'secondary': {
                'vesicle', 'cytoskeleton', 'membrane transport', 'cell division',
                'cell cycle', 'apoptosis'
            }
        },
        'genetics': {
            'primary': {
                'inheritance', 'allele', 'phenotype', 'genotype', 'heredity',
                'genetic disorder', 'mutation'
            },
            'secondary': {
                'dominant', 'recessive', 'homozygous', 'heterozygous', 'pedigree',
                'genetic mapping'
            }
        },
        'ecology': {
            'primary': {
                'ecosystem', 'habitat', 'species', 'population', 'community',
                'biodiversity', 'food web'
            },
            'secondary': {
                'niche', 'symbiosis', 'competition', 'predation', 'mutualism',
                'adaptation', 'natural selection'
            }
        }
    }

    EXCLUSION_TERMS = {
        'fictional', 'movie', 'film', 'novel', 'book series', 'game', 'mythology',
        'folklore', 'legend', 'supernatural', 'fantasy'
    }

class BiologyContentExtractor:
    def __init__(self, patterns: BiologyPatterns):
        self.patterns = patterns
        self.nlp = spacy.load("en_core_web_sm")
        self.primary_terms = set()
        self.secondary_terms = set()
        for subject in patterns.SUBJECTS.values():
            self.primary_terms.update(subject['primary'])
            self.secondary_terms.update(subject['secondary'])
    
    def is_biology_content(self, text: str, title: str) -> Tuple[bool, float, str]:
        if len(text) < 100 or len(text) > 50000:
            return False, 0.0, ""

        text_lower = text.lower()
        title_lower = title.lower()

        if any(term in title_lower for term in self.patterns.EXCLUSION_TERMS):
            return False, 0.0, ""

        doc = self.nlp(text[:10000])

        subject_scores = {}
        for subject, terms in self.patterns.SUBJECTS.items():
            primary_matches = sum(1 for term in terms['primary'] 
                                if term in text_lower)
            secondary_matches = sum(1 for term in terms['secondary'] 
                                  if term in text_lower)
            
            #Weight primary matches more heavily
            score = (primary_matches * 2 + secondary_matches) / \
                   (len(terms['primary']) * 2 + len(terms['secondary']))
            subject_scores[subject] = score
        
        #Get the subject with highest score
        best_subject = max(subject_scores.items(), key=lambda x: x[1])
        
        #Calculate overall confidence based on best subject score
        confidence = best_subject[1]
        
        #Require minimum confidence threshold
        is_biology = confidence >= 0.15
        
        return is_biology, confidence, best_subject[0]
    
    def clean_content(self, text: str) -> str:
        #Remove references and citations
        text = re.sub(r'\[\d+\]', '', text)
        
        #Remove extra whitespace
        text = ' '.join(text.split())
        
        #Split into sentences and rejoin with proper spacing
        sentences = sent_tokenize(text)
        text = ' '.join(sentences)
        
        return text

class WikipediaExtractor:
    def __init__(self, input_path: str, output_path: str):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.biology_extractor = BiologyContentExtractor(BiologyPatterns())
        self.setup_logging()
    
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('biology_extraction.log'),
                logging.StreamHandler()
            ]
        )
    
    async def process_file(self, file_path: Path) -> List[dict]:
        biology_articles = []
        
        async with aiofiles.open(file_path, mode='r', encoding='utf-8') as f:
            line_count = 0
            async for line in f:
                line_count += 1
                if line_count % 1000 == 0:
                    logging.info(f"Processed {line_count} lines from {file_path.name}")
                
                try:
                    article = json.loads(line.strip())
                    
                    #Skip disambiguation pages and lists
                    if any(term in article.get('title', '').lower() 
                          for term in ['disambiguation', 'list of']):
                        continue
                    
                    is_biology, confidence, subject = self.biology_extractor.is_biology_content(
                        article.get('text', ''),
                        article.get('title', '')
                    )
                    
                    if is_biology:
                        clean_text = self.biology_extractor.clean_content(article['text'])
                        biology_articles.append({
                            'title': article['title'],
                            'text': clean_text,
                            'subject': subject,
                            'confidence': confidence
                        })
                
                except json.JSONDecodeError:
                    logging.error(f"Error decoding JSON at line {line_count}")
                except Exception as e:
                    logging.error(f"Error processing line {line_count}: {str(e)}")
        
        return biology_articles
    
    async def process_dumps(self):
        #Create output directory if it doesn't exist
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        #Process each dump file
        for file_path in self.input_path.glob('*.jsonl'):
            logging.info(f"Processing file: {file_path.name}")
            
            biology_articles = await self.process_file(file_path)
            
            #Save extracted biology articles
            output_file = self.output_path / f"biology_{file_path.stem}.jsonl"
            async with aiofiles.open(output_file, mode='w', encoding='utf-8') as f:
                for article in biology_articles:
                    await f.write(json.dumps(article) + '\n')
            
            logging.info(f"Extracted {len(biology_articles)} biology articles from {file_path.name}")

async def main():
    #Configure your input and output paths
    input_path = "../Wikipedia_dumps/training_data/gpt4_training_data.jsonl"
    output_path = "../Wikipedia_dumps/training_data"
    
    extractor = WikipediaExtractor(input_path, output_path)
    await extractor.process_dumps()

if __name__ == "__main__":
    asyncio.run(main())
