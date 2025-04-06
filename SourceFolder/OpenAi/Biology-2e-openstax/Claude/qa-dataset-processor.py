import os
import json
import csv
from pathlib import Path
import random

class QADatasetProcessor:
    def __init__(self, input_dir="qa_results", output_dir="processed_data"):
        """Initialize the processor with input and output directories"""
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        print(f"Input directory: {self.input_dir}")
        print(f"Output directory: {self.output_dir}")
    
    def combine_all_files(self):
        """Combine all JSON files in the input directory into one dataset"""
        print("Combining all chapter files...")
        
        json_files = sorted(list(self.input_dir.glob("chapter_*_qa.json")), 
                           key=lambda x: int(x.stem.split('_')[1]))
        
        if not json_files:
            print("No chapter files found!")
            return None
        
        all_qa_pairs = []
        chapter_counts = {}
        
        # Process each file
        for file_path in json_files:
            try:
                # Extract chapter number
                chapter_num = int(file_path.stem.split('_')[1])
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Add chapter number to each Q&A pair
                for qa_pair in data:
                    qa_pair['chapter'] = chapter_num
                    all_qa_pairs.append(qa_pair)
                
                chapter_counts[chapter_num] = len(data)
                print(f"  Added {len(data)} Q&A pairs from Chapter {chapter_num}")
            
            except Exception as e:
                print(f"  Error processing {file_path.name}: {e}")
        
        # Save the combined dataset
        output_file = self.output_dir / "biology_qa_dataset.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_qa_pairs, f, indent=2)
        
        print(f"Combined dataset saved with {len(all_qa_pairs)} total Q&A pairs")
        print(f"File saved to: {output_file}")
        
        # Print chapter statistics
        print("\nQuestions per chapter:")
        for chapter, count in sorted(chapter_counts.items()):
            print(f"  Chapter {chapter}: {count} questions")
        
        return all_qa_pairs
    
    def export_as_csv(self, dataset=None):
        """Export the dataset as a CSV file"""
        if dataset is None:
            # Try to load the combined dataset if available
            combined_file = self.output_dir / "biology_qa_dataset.json"
            if combined_file.exists():
                with open(combined_file, 'r', encoding='utf-8') as f:
                    dataset = json.load(f)
            else:
                print("No dataset provided and no combined file found.")
                return False
        
        output_file = self.output_dir / "biology_qa_dataset.csv"
        
        with open(output_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Chapter', 'Question', 'Answer'])
            
            for qa_pair in dataset:
                writer.writerow([
                    qa_pair.get('chapter', ''),
                    qa_pair['question'],
                    qa_pair['answer']
                ])
        
        print(f"Exported CSV file to: {output_file}")
        return True
    
    def create_study_quiz(self, num_questions=20, dataset=None):
        """Create a randomized study quiz with a specific number of questions"""
        if dataset is None:
            # Try to load the combined dataset if available
            combined_file = self.output_dir / "biology_qa_dataset.json"
            if combined_file.exists():
                with open(combined_file, 'r', encoding='utf-8') as f:
                    dataset = json.load(f)
            else:
                print("No dataset provided and no combined file found.")
                return False
        
        # If there are fewer questions than requested, use all of them
        num_questions = min(num_questions, len(dataset))
        
        # Randomly select questions
        selected_questions = random.sample(dataset, num_questions)
        
        # Sort by chapter
        selected_questions.sort(key=lambda x: x.get('chapter', 0))
        
        # Create the quiz file
        output_file = self.output_dir / f"biology_study_quiz_{num_questions}q.txt"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"Biology Study Quiz - {num_questions} Questions\n\n")
            
            for i, qa_pair in enumerate(selected_questions, 1):
                chapter = qa_pair.get('chapter', '')
                f.write(f"Question {i} (Chapter {chapter}):\n")
                f.write(f"{qa_pair['question']}\n\n")
            
            f.write("\n--- ANSWERS ---\n\n")
            
            for i, qa_pair in enumerate(selected_questions, 1):
                f.write(f"Answer {i}:\n")
                f.write(f"{qa_pair['answer']}\n\n")
        
        print(f"Created study quiz with {num_questions} questions at: {output_file}")
        return True
    
    def create_anki_import(self, dataset=None):
        """Create a text file for importing into Anki flashcards"""
        if dataset is None:
            # Try to load the combined dataset if available
            combined_file = self.output_dir / "biology_qa_dataset.json"
            if combined_file.exists():
                with open(combined_file, 'r', encoding='utf-8') as f:
                    dataset = json.load(f)
            else:
                print("No dataset provided and no combined file found.")
                return False
        
        output_file = self.output_dir / "biology_flashcards_anki.txt"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for qa_pair in dataset:
                chapter = qa_pair.get('chapter', '')
                # Format: Question; Answer
                f.write(f"{qa_pair['question']} (Ch. {chapter}); {qa_pair['answer']}\n")
        
        print(f"Created Anki import file at: {output_file}")
        print("To import into Anki:")
        print("1. Open Anki and create a new deck")
        print("2. Click 'Import File' and select this file")
        print("3. Make sure the field separator is set to ';'")
        return True
    
    def create_openai_finetuning_data(self, dataset=None):
        """Create a JSONL file for OpenAI fine-tuning"""
        if dataset is None:
            # Try to load the combined dataset if available
            combined_file = self.output_dir / "biology_qa_dataset.json"
            if combined_file.exists():
                with open(combined_file, 'r', encoding='utf-8') as f:
                    dataset = json.load(f)
            else:
                print("No dataset provided and no combined file found.")
                return False
        
        output_file = self.output_dir / "biology_finetuning_data.jsonl"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for qa_pair in dataset:
                chapter = qa_pair.get('chapter', '')
                
                # Format for ChatGPT fine-tuning
                item = {
                    "messages": [
                        {"role": "system", "content": "You are a helpful biology tutor."},
                        {"role": "user", "content": qa_pair['question']},
                        {"role": "assistant", "content": qa_pair['answer']}
                    ]
                }
                
                f.write(json.dumps(item) + '\n')
        
        print(f"Created OpenAI fine-tuning file at: {output_file}")
        return True


if __name__ == "__main__":
    # Initialize the processor
    processor = QADatasetProcessor()
    
    # Combine all chapter files
    dataset = processor.combine_all_files()
    
    if dataset:
        # Export as CSV
        processor.export_as_csv(dataset)
        
        # Create a study quiz with 20 random questions
        processor.create_study_quiz(20, dataset)
        
        # Create Anki flashcard import file
        processor.create_anki_import(dataset)
        
        # Create OpenAI fine-tuning data
        processor.create_openai_finetuning_data(dataset)
    
    print("\nProcessing complete!")
