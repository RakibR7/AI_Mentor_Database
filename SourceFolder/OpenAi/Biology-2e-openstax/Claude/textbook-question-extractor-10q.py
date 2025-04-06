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
        
        # More precise pattern for chapter headings to reduce false positives
        chapter_pattern = r"(?:^|\n)CHAPTER\s+(\d+)(?:\s+[A-Z]|\s*\n)"
        
        # Find all chapter headings
        chapter_matches = list(re.finditer(chapter_pattern, content, re.IGNORECASE))
        
        chapters = []
        if chapter_matches:
            print(f"Found {len(chapter_matches)} chapter headings")
            
            # Extract each chapter
            for i in range(len(chapter_matches)):
                start = chapter_matches[i].start()
                end = chapter_matches[i+1].start() if i < len(chapter_matches) - 1 else len(content)
                
                chapter_content = content[start:end].strip()
                chapter_title = chapter_matches[i].group(0).strip()
                
                # Extract chapter number
                chapter_num_match = re.search(r"CHAPTER\s+(\d+)", chapter_title, re.IGNORECASE)
                if chapter_num_match:
                    chapter_num = int(chapter_num_match.group(1))
                else:
                    chapter_num = i + 1
                
                chapters.append({
                    "number": chapter_num,
                    "title": chapter_title,
                    "content": chapter_content
                })
        
        # If no chapters found with standard pattern, try alternative
        if not chapters:
            print("No standard chapter headings found. Looking for alternative patterns...")
            
            # Try alternative pattern (e.g., chapter headings in another format)
            alt_pattern = r"Chapter\s+(\d+)"
            chapter_matches = list(re.finditer(alt_pattern, content))
            
            # Extract each chapter
            for i in range(len(chapter_matches)):
                start = chapter_matches[i].start()
                end = chapter_matches[i+1].start() if i < len(chapter_matches) - 1 else len(content)
                
                chapter_content = content[start:end].strip()
                chapter_title = chapter_matches[i].group(0).strip()
                
                # Extract chapter number
                chapter_num_match = re.search(r"Chapter\s+(\d+)", chapter_title)
                if chapter_num_match:
                    chapter_num = int(chapter_num_match.group(1))
                else:
                    chapter_num = i + 1
                
                # Prevent duplicate chapter numbers
                if not any(c["number"] == chapter_num for c in chapters):
                    chapters.append({
                        "number": chapter_num,
                        "title": chapter_title,
                        "content": chapter_content
                    })
        
        # If still no chapters, create artificial ones
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
        
        # Sort chapters by number
        chapters.sort(key=lambda x: x["number"])
        
        # Deduplicate chapters with the same number
        unique_chapters = []
        seen_numbers = set()
        
        for chapter in chapters:
            if chapter["number"] not in seen_numbers:
                unique_chapters.append(chapter)
                seen_numbers.add(chapter["number"])
        
        print(f"Found {len(unique_chapters)} unique chapters/sections")
        return unique_chapters
    
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
    
    def generate_questions_with_ollama(self, chunk, chapter_info, num_questions=10):
        """Generate questions from text chunk using Ollama API"""
        prompt = f"""You are a biology tutor creating study questions from a textbook.
        
Below is a section from Chapter {chapter_info['number']} of a biology textbook:

{chunk}

Based on this text, generate {num_questions} important conceptual questions and their answers that would appear on an exam.
Focus on key concepts, definitions, processes, and important relationships.
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
  }}
]

Your response should be in valid JSON format that can be parsed programmatically."""

        try:
            response = requests.post(
                self.ollama_api,
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": 0.7,
                    "top_p": 0.95,
                    "max_tokens": 4096  # Increased token limit for 10 questions
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
                        print(f"  Successfully generated {len(qa_pairs)} Q&A pairs")
                        return qa_pairs
                    else:
                        print(f"  Warning: Could not find JSON array in response")
                        print(f"  Response starts with: {generated_text[:150]}...")
                except json.JSONDecodeError:
                    print(f"  Warning: Invalid JSON format in response")
                    print(f"  Response starts with: {generated_text[:150]}...")
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
            # Output file for this chapter
            output_file = self.output_dir / f"chapter_{chapter['number']}_qa.json"
            
            # Check if we should overwrite existing file
            if output_file.exists():
                print(f"File exists for Chapter {chapter['number']}. Overwriting...")
                # Delete existing file to ensure fresh content
                output_file.unlink()
            
            # Chunk the chapter
            chunks = self.chunk_chapter(chapter['content'])
            print(f"Chapter {chapter['number']} split into {len(chunks)} chunks")
            
            # Process each chunk and collect Q&A pairs
            all_qa_pairs = []
            
            # Process multiple chunks if needed to reach 10 questions
            chunks_to_process = min(2, len(chunks))  # Process up to 2 chunks
            
            for i in range(chunks_to_process):
                chunk = chunks[i]
                print(f"Processing chunk {i+1}/{chunks_to_process} for Chapter {chapter['number']}")
                
                # Calculate how many questions to request from this chunk
                questions_needed = 10 - len(all_qa_pairs)
                if questions_needed <= 0:
                    break
                    
                qa_pairs = self.generate_questions_with_ollama(chunk, chapter, questions_needed)
                all_qa_pairs.extend(qa_pairs)
                
                # If we have at least 10 questions, stop processing chunks
                if len(all_qa_pairs) >= 10:
                    # Trim to exactly 10 questions if we have more
                    all_qa_pairs = all_qa_pairs[:10]
                    break
            
            # Save the results
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(all_qa_pairs, f, indent=2)
            
            print(f"Saved {len(all_qa_pairs)} Q&A pairs for Chapter {chapter['number']}")
        
        print("Question extraction complete!")
        return True


if __name__ == "__main__":
    # Use the path format that works for you
    textbook_path = "input_textbook/biology_textbook.txt"
    
    # Initialize and run the extractor
    extractor = TextbookQuestionExtractor(textbook_path, model_name="llama2:13b")
    extractor.extract_questions()
