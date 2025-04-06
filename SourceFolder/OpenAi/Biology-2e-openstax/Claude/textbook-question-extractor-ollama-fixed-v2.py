import os
import json
import re
import requests
from pathlib import Path
from tqdm import tqdm

class TextbookQuestionExtractor:
    def __init__(self, textbook_path, output_dir="qa_results", model_name="llama2:13b"):
        """Initialize the extractor with path to textbook"""
        self.textbook_path = Path(textbook_path)
        self.output_dir = Path(output_dir)
        self.model_name = model_name
        self.ollama_api = "http://localhost:11434/api/generate"
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        print(f"Textbook: {self.textbook_path}")
        print(f"Output directory: {self.output_dir}")
        print(f"Using Ollama API with model: {self.model_name}")
    
    def load_textbook(self):
        """Load and return the textbook content"""
        print(f"Loading textbook from {self.textbook_path}")
        try:
            with open(self.textbook_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"Error loading textbook: {e}")
            return None
    
    def split_into_chapters(self, content):
        """Split the textbook content into chapters"""
        print("Splitting textbook into chapters...")
        
        # Try multiple patterns to identify chapter headings
        chapter_patterns = [
            r"Chapter\s+\d+[\.:]\s+.+",  # "Chapter 1: Introduction" or "Chapter 1. Introduction"
            r"Chapter\s+\d+",             # Just "Chapter 1"
            r"\d+\.\s+[A-Z][a-zA-Z\s]+",  # "1. Introduction to Biology"
            r"^[A-Z][A-Z\s]+$"            # "INTRODUCTION" (all caps)
        ]
        
        chapters = []
        for pattern in chapter_patterns:
            # Find all chapter headings with this pattern
            chapter_matches = list(re.finditer(pattern, content, re.MULTILINE))
            
            if chapter_matches:
                print(f"Found chapter pattern: {pattern}")
                
                # Extract each chapter
                for i in range(len(chapter_matches)):
                    start = chapter_matches[i].start()
                    end = chapter_matches[i+1].start() if i < len(chapter_matches) - 1 else len(content)
                    
                    chapter_content = content[start:end].strip()
                    chapter_title = chapter_matches[i].group(0).strip()
                    
                    # Try to extract chapter number
                    chapter_num_match = re.search(r"Chapter\s+(\d+)|^(\d+)\.", chapter_title)
                    if chapter_num_match:
                        # Get the first non-None group
                        chapter_num = next(g for g in chapter_num_match.groups() if g is not None)
                        chapter_num = int(chapter_num)
                    else:
                        chapter_num = i + 1
                    
                    chapters.append({
                        "number": chapter_num,
                        "title": chapter_title,
                        "content": chapter_content
                    })
                
                # If we found chapters with this pattern, stop trying other patterns
                if chapters:
                    break
        
        # If no chapters found with patterns, create artificial chapters
        if not chapters:
            print("No chapter patterns matched. Creating artificial chapters based on content size.")
            
            # Split into roughly equal chunks (50,000 characters each)
            chunk_size = 50000
            total_chunks = (len(content) + chunk_size - 1) // chunk_size
            
            for i in range(total_chunks):
                start = i * chunk_size
                end = min((i + 1) * chunk_size, len(content))
                
                chapter_content = content[start:end].strip()
                
                chapters.append({
                    "number": i + 1,
                    "title": f"Section {i + 1}",
                    "content": chapter_content
                })
        
        print(f"Found {len(chapters)} chapters/sections")
        return chapters
    
    def chunk_chapter(self, chapter_text, max_length=4000):
        """Split a chapter into manageable chunks for the model"""
        # Split by paragraphs
        paragraphs = re.split(r'\n\s*\n', chapter_text)
        
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            # If adding this paragraph exceeds max length, start a new chunk
            if len(current_chunk) + len(para) > max_length and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = para
            else:
                current_chunk += "\n\n" + para if current_chunk else para
        
        # Add the last chunk if there's anything left
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def generate_questions_with_ollama(self, chunk, chapter_info):
        """Generate questions from text chunk using Ollama API"""
        prompt = f"""You are a biology tutor creating study questions from a textbook.
        
Below is a section from Chapter {chapter_info['number']} of a biology textbook:

{chunk}

Based on this text, generate 3 important conceptual questions and their answers that would appear on an exam.
Format your response as a JSON array with "question" and "answer" fields, and nothing else.
Example format:
[
  {{
    "question": "What is photosynthesis?",
    "answer": "Photosynthesis is the process by which plants convert light energy into chemical energy."
  }},
  {{
    "question": "What are the main components of a cell membrane?",
    "answer": "Cell membranes consist primarily of phospholipids arranged in a bilayer, with proteins embedded within or attached to the surface. The phospholipids have hydrophilic heads facing outward and hydrophobic tails facing inward."
  }},
  {{
    "question": "How does natural selection lead to adaptation?",
    "answer": "Natural selection occurs when individuals with certain heritable traits survive and reproduce more successfully than others in a specific environment. Over generations, this leads to the population becoming better adapted to that environment as favorable traits become more common."
  }}
]"""

        try:
            response = requests.post(
                self.ollama_api,
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": 0.7,
                    "top_p": 0.95,
                    "max_tokens": 2048
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                generated_text = result.get("response", "")
                
                # Try to parse as JSON
                try:
                    # Find JSON array in the response
                    json_match = re.search(r'\[\s*\{.*\}\s*\]', generated_text, re.DOTALL)
                    if json_match:
                        json_text = json_match.group(0)
                        qa_pairs = json.loads(json_text)
                        return qa_pairs
                    else:
                        print(f"  Warning: Could not find JSON array in response")
                        print(f"  Response: {generated_text[:150]}...")
                except json.JSONDecodeError:
                    print(f"  Warning: Invalid JSON format in response")
                    print(f"  Response: {generated_text[:150]}...")
            else:
                print(f"  Error: API returned status code {response.status_code}")
        
        except Exception as e:
            print(f"  Error calling Ollama API: {e}")
        
        return []
    
    def extract_questions(self):
        """Main method to extract questions from the textbook"""
        # Load the textbook
        content = self.load_textbook()
        if not content:
            return False
        
        # Print the first 200 characters to debug
        print(f"Textbook content starts with: {content[:200].replace(chr(10), ' ')}")
        
        # Split into chapters
        chapters = self.split_into_chapters(content)
        if not chapters:
            print("No chapters found in the textbook.")
            return False
        
        # Check if Ollama is available
        try:
            test_response = requests.get("http://localhost:11434/api/tags")
            if test_response.status_code != 200:
                print("Warning: Ollama API responded but returned non-200 status code")
        except Exception as e:
            print(f"Error: Could not connect to Ollama API. Make sure Ollama is running: {e}")
            return False
            
        # Process each chapter
        for chapter in chapters:
            # Skip already processed chapters if files exist
            output_file = self.output_dir / f"chapter_{chapter['number']}_qa.json"
            if output_file.exists():
                print(f"Skipping Chapter {chapter['number']} (already processed)")
                continue
            
            # Chunk the chapter
            chunks = self.chunk_chapter(chapter['content'])
            print(f"Chapter {chapter['number']} split into {len(chunks)} chunks")
            
            # Process each chunk and collect Q&A pairs
            all_qa_pairs = []
            for i, chunk in enumerate(tqdm(chunks, desc=f"Processing Chapter {chapter['number']} chunks")):
                qa_pairs = self.generate_questions_with_ollama(chunk, chapter)
                all_qa_pairs.extend(qa_pairs)
                
                # Optional: Save intermediate results after every few chunks
                if (i + 1) % 5 == 0 or i == len(chunks) - 1:
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(all_qa_pairs, f, indent=2)
                    print(f"  Saved {len(all_qa_pairs)} Q&A pairs so far...")
            
            print(f"Completed Chapter {chapter['number']} with {len(all_qa_pairs)} Q&A pairs")
        
        print("Question extraction complete!")
        return True


if __name__ == "__main__":
    # Use the path format that works for you
    textbook_path = "input_textbook/biology_textbook.txt"
    
    # Initialize and run the extractor
    extractor = TextbookQuestionExtractor(textbook_path, model_name="llama2:13b")
    extractor.extract_questions()
