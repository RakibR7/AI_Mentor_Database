import os
import json
import re
from pathlib import Path
from tqdm import tqdm

# You'll need a library to interact with your Llama model
# This example uses llama-cpp-python which is a common interface for local LLM use
# Install with: pip install llama-cpp-python
from llama_cpp import Llama

class TextbookQuestionExtractor:
    def __init__(self, textbook_path, output_dir="qa_results", model_path=None):
        """Initialize the extractor with paths to textbook and model"""
        self.textbook_path = Path(textbook_path)
        self.output_dir = Path(output_dir)
        self.model_path = model_path or self.find_model_path()
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        print(f"Textbook: {self.textbook_path}")
        print(f"Output directory: {self.output_dir}")
        print(f"Model path: {self.model_path}")
    
    def find_model_path(self):
        """Try to find a Llama 13B model in common locations"""
        possible_locations = [
            "./models/llama-13b.gguf",
            "./llama-13b.gguf",
            os.path.expanduser("~/models/llama-13b.gguf"),
            os.path.expanduser("~/.cache/huggingface/models/llama-13b.gguf")
        ]
        
        for loc in possible_locations:
            if os.path.exists(loc):
                return loc
        
        # If we can't find it, ask the user
        print("Llama 13B model not found in common locations.")
        return input("Please enter the full path to your Llama 13B model: ")
    
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
        
        # Pattern to identify chapter headings (adjust based on your textbook format)
        chapter_pattern = r"Chapter\s+\d+[\.:]\s+.+"
        
        # Find all chapter headings
        chapter_matches = list(re.finditer(chapter_pattern, content, re.IGNORECASE))
        
        chapters = []
        
        # Extract each chapter
        for i in range(len(chapter_matches)):
            start = chapter_matches[i].start()
            end = chapter_matches[i+1].start() if i < len(chapter_matches) - 1 else len(content)
            
            chapter_content = content[start:end].strip()
            chapter_title = chapter_matches[i].group(0).strip()
            
            # Clean up the chapter number
            chapter_num = re.search(r"Chapter\s+(\d+)", chapter_title, re.IGNORECASE)
            chapter_num = int(chapter_num.group(1)) if chapter_num else i + 1
            
            chapters.append({
                "number": chapter_num,
                "title": chapter_title,
                "content": chapter_content
            })
        
        print(f"Found {len(chapters)} chapters")
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
    
    def initialize_model(self):
        """Load the Llama model"""
        print("Initializing Llama 13B model... (this may take a while)")
        try:
            # Initialize with parameters suitable for question generation
            # Adjust n_ctx based on your available VRAM and needs
            model = Llama(
                model_path=self.model_path,
                n_ctx=4096,  # Context window
                n_threads=4,  # Adjust based on your CPU
                n_gpu_layers=0  # Set to non-zero if you have GPU support
            )
            return model
        except Exception as e:
            print(f"Error loading Llama model: {e}")
            return None
    
    def generate_questions(self, model, chapter_chunks, chapter_info):
        """Generate questions from each chunk using the model"""
        print(f"Generating questions for Chapter {chapter_info['number']}: {chapter_info['title']}")
        
        all_qa_pairs = []
        
        for i, chunk in enumerate(tqdm(chapter_chunks, desc=f"Processing Chapter {chapter_info['number']} chunks")):
            # Prompt the model to generate questions and answers
            prompt = f"""You are a biology tutor creating study questions from a textbook.
            
Below is a section from Chapter {chapter_info['number']} of a biology textbook:

{chunk}

Based on this text, generate 3 important conceptual questions and their answers that would appear on an exam.
Format your response as a JSON array with "question" and "answer" fields.
Example format:
[
  {{
    "question": "What is photosynthesis?",
    "answer": "Photosynthesis is the process by which plants convert light energy into chemical energy."
  }},
  ...
]

Questions:"""

            # Generate response from the model
            try:
                response = model(
                    prompt,
                    max_tokens=2048,
                    temperature=0.7,
                    top_p=0.95,
                    stop=["```"],  # In case the model uses code blocks
                    echo=False
                )
                
                # Extract the generated text
                generated_text = response["choices"][0]["text"].strip()
                
                # Try to parse as JSON
                try:
                    # Find JSON array in the response
                    json_match = re.search(r'\[\s*\{.*\}\s*\]', generated_text, re.DOTALL)
                    if json_match:
                        json_text = json_match.group(0)
                        qa_pairs = json.loads(json_text)
                        all_qa_pairs.extend(qa_pairs)
                    else:
                        print(f"  Warning: Could not find JSON array in response for chunk {i+1}")
                except json.JSONDecodeError:
                    print(f"  Warning: Invalid JSON format in response for chunk {i+1}")
                    print(f"  Response: {generated_text[:100]}...")
            
            except Exception as e:
                print(f"  Error generating questions for chunk {i+1}: {e}")
        
        return all_qa_pairs
    
    def extract_questions(self):
        """Main method to extract questions from the textbook"""
        # Load the textbook
        content = self.load_textbook()
        if not content:
            return False
        
        # Split into chapters
        chapters = self.split_into_chapters(content)
        if not chapters:
            print("No chapters found in the textbook.")
            return False
        
        # Initialize the model
        model = self.initialize_model()
        if not model:
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
            
            # Generate questions for this chapter
            qa_pairs = self.generate_questions(model, chunks, chapter)
            
            # Save the results
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(qa_pairs, f, indent=2)
            
            print(f"Saved {len(qa_pairs)} Q&A pairs for Chapter {chapter['number']}")
        
        print("Question extraction complete!")
        return True


if __name__ == "__main__":
    # Set the path to your textbook file
    textbook_path = "/SourceFolder/OpenAi/Biology-2e-openstax/Claude/input_textbook/biology_textbook.txt"
    
    # Initialize and run the extractor
    extractor = TextbookQuestionExtractor(textbook_path)
    extractor.extract_questions()
