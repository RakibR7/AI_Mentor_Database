import os
import json
import re
from pathlib import Path


class TextbookProcessor:
    def __init__(self):
        """Initialize the processor with paths."""
        # Set up paths
        self.input_dir = Path("input_textbook")
        self.output_dir = Path("output_qa")
        self.output_dir.mkdir(exist_ok=True)

        print("Textbook processor initialized")

    def chunk_by_chapter(self, text):
        """Split text into chapters based on 'CHAPTER X' markers."""
        chapter_pattern = r'(CHAPTER \d+.*?)(?=CHAPTER \d+|$)'
        chapters = re.findall(chapter_pattern, text, re.DOTALL)

        # If no chapters found, try to find sections
        if not chapters:
            section_pattern = r'(\d+\.\d+.*?)(?=\d+\.\d+|$)'
            chapters = re.findall(section_pattern, text, re.DOTALL)

        # If still no chunks found, use a simple size-based approach
        if not chapters:
            # Split into chunks of ~100,000 characters
            chunk_size = 100000
            chapters = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

        return chapters

    def process_file(self, file_path):
        """Process a textbook file into chapter chunks."""
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()

        # Get chapters
        chapters = self.chunk_by_chapter(text)
        print(f"Split text into {len(chapters)} chapters/chunks")

        # Save individual chapter files
        chapter_files = []
        for i, chapter in enumerate(chapters):
            chapter_file = self.output_dir / f"chapter_{i + 1}.txt"
            with open(chapter_file, 'w', encoding='utf-8') as f:
                f.write(chapter)
            chapter_files.append(chapter_file)

        return chapter_files

    def prepare_paragraphs(self, chapter_file):
        """Extract meaningful paragraphs from a chapter file."""
        # Read chapter content
        with open(chapter_file, 'r', encoding='utf-8') as f:
            chapter_content = f.read()

        # Split chapter into paragraphs
        paragraphs = [p for p in chapter_content.split('\n\n') if len(p.strip()) > 100]

        # Filter to meaningful paragraphs
        meaningful_paragraphs = []
        for paragraph in paragraphs:
            # Skip if paragraph is too short or likely just a heading
            if len(paragraph) < 200 or paragraph.isupper():
                continue

            # Skip figure captions and other non-content paragraphs
            if "(Figure" in paragraph or "credit:" in paragraph or "(credit:" in paragraph:
                continue

            meaningful_paragraphs.append(paragraph.strip())

        # Save the meaningful paragraphs to a file
        output_file = self.output_dir / f"{os.path.basename(chapter_file).replace('.txt', '_paragraphs.txt')}"
        with open(output_file, 'w', encoding='utf-8') as f:
            for i, paragraph in enumerate(meaningful_paragraphs):
                f.write(f"PARAGRAPH {i + 1}:\n{paragraph}\n\n")

        return output_file, meaningful_paragraphs

    def create_qa_prompts(self, chapter_file, solutions_file=None):
        """Create prompts for generating Q&A pairs."""
        # Process paragraphs first
        _, paragraphs = self.prepare_paragraphs(chapter_file)

        # Read solutions content if available
        solutions_content = ""
        if solutions_file and os.path.exists(solutions_file):
            with open(solutions_file, 'r', encoding='utf-8') as f:
                solutions_content = f.read()

        # Create a file with prompts that can be fed to a model
        output_file = self.output_dir / f"{os.path.basename(chapter_file).replace('.txt', '_prompts.txt')}"
        with open(output_file, 'w', encoding='utf-8') as f:
            for i, paragraph in enumerate(paragraphs):
                # Only process every few paragraphs if there are too many
                if len(paragraphs) > 30 and i % 3 != 0:
                    continue

                # Create the prompt
                prompt = self._create_prompt(paragraph, solutions_content)

                # Write to file
                f.write(f"PROMPT {i + 1}:\n{prompt}\n\n{'=' * 80}\n\n")

        return output_file

    def _create_prompt(self, paragraph, solutions=""):
        """Create a prompt for the LLM."""
        return f"""You are a helpful AI assistant.
Your task is to generate high-quality biology question-answer (Q&A) pairs using content from a college-level textbook.
Instructions:
1. Generate 1 or 2 clear and standalone biology questions from the paragraph.
2. Provide correct and accurate answers.
3. Each Q&A must be output as a JSON object.
4. Do NOT include references to figures or images.

Textbook paragraph:
{paragraph}

Solutions context (use if relevant):
{solutions}

Generate 1-2 Q&A pairs in this format only:
[{{"question": "Your question here", "answer": "Your answer here"}}]"""

    def create_qa_template(self):
        """Create a template file for manually filling in Q&A pairs."""
        template_file = self.output_dir / "qa_template.json"

        template = [
            {
                "question": "What is biology?",
                "answer": "Biology is the study of life. It encompasses everything from the microscopic view of cells to entire ecosystems and the living planet."
            },
            {
                "question": "What are the steps of the scientific method?",
                "answer": "The scientific method includes defined steps with experiments and careful observation. Key aspects include testing hypotheses through repeatable experiments, making observations, and analyzing data to form conclusions."
            }
        ]

        with open(template_file, 'w', encoding='utf-8') as f:
            json.dump(template, f, indent=2)

        print(f"Created Q&A template file at {template_file}")
        return template_file

    def prepare_gpt_finetuning_format(self, qa_file):
        """Convert Q&A JSON file to GPT fine-tuning format."""
        # Read the Q&A pairs
        with open(qa_file, 'r', encoding='utf-8') as f:
            qa_pairs = json.load(f)

        # Convert to GPT fine-tuning format
        gpt_format = []
        for pair in qa_pairs:
            item = {
                "messages": [
                    {"role": "system", "content": "You are a helpful biology tutor."},
                    {"role": "user", "content": pair["question"]},
                    {"role": "assistant", "content": pair["answer"]}
                ]
            }
            gpt_format.append(item)

        # Save as JSONL file
        output_file = self.output_dir / "gpt_finetuning_data.jsonl"
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in gpt_format:
                f.write(json.dumps(item) + '\n')

        print(f"Created GPT fine-tuning file at {output_file}")
        return output_file

    def create_sample_qa_pairs(self, chapter_file):
        """Create sample Q&A pairs based on the first chapter."""
        # Process paragraphs
        _, paragraphs = self.prepare_paragraphs(chapter_file)

        # Create sample Q&A pairs
        sample_qa = []

        # From the biology chapter you uploaded
        if "biology" in chapter_file.name.lower():
            sample_qa.extend([
                {
                    "question": "What is the definition of biology and what range of topics does it encompass?",
                    "answer": "Biology is the study of life. It encompasses a vast range of topics, from the microscopic or submicroscopic view of cells to ecosystems and the entire living planet. Biologists study everything from the structure and function of cells to the behavior of organisms and the interactions between living things and their environment."
                },
                {
                    "question": "How is the scientific method applied in biology?",
                    "answer": "In biology, the scientific method is applied through a structured approach that includes making observations, forming hypotheses, conducting experiments with careful observation, and analyzing data. An important aspect of this method is testing hypotheses through repeatable experiments. Although the scientific method was used even in ancient times, it was first formally documented by England's Sir Francis Bacon. The process involves both inductive reasoning (drawing general conclusions from specific observations) and deductive reasoning (using general principles to predict specific results)."
                },
                {
                    "question": "What is the difference between inductive and deductive reasoning in scientific research?",
                    "answer": "Inductive reasoning involves formulating generalizations from careful observation and analysis of large amounts of data. It's a form of logical thinking that uses related observations to arrive at a general conclusion and is common in descriptive science. For example, a biologist might observe brain activity during specific tasks and infer which parts control certain responses. Deductive reasoning, on the other hand, is used in hypothesis-based science. It moves in the opposite direction, using general principles or laws to forecast specific results. Scientists might predict, for instance, that if climate becomes warmer in a region, then plant and animal distribution should change. Both types of reasoning are essential to scientific inquiry."
                }
            ])

        # Save the sample Q&A pairs
        output_file = self.output_dir / "sample_qa_pairs.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(sample_qa, f, indent=2)

        print(f"Created sample Q&A pairs at {output_file}")
        return output_file

    def process_all(self, textbook_path, solutions_dir=None):
        """Process the entire textbook."""
        # Process the textbook into chapters
        chapter_files = self.process_file(textbook_path)

        # For each chapter, extract paragraphs and create prompts
        for i, chapter_file in enumerate(chapter_files):
            # Look for corresponding solutions file
            solutions_file = None
            if solutions_dir:
                solutions_path = Path(solutions_dir)
                potential_solutions = list(solutions_path.glob(f"*chapter*{i + 1}*"))
                if potential_solutions:
                    solutions_file = potential_solutions[0]

            print(f"Processing Chapter {i + 1}...")

            # Prepare paragraphs
            paragraph_file, _ = self.prepare_paragraphs(chapter_file)
            print(f"  Extracted paragraphs saved to {paragraph_file}")

            # Create prompts
            prompt_file = self.create_qa_prompts(chapter_file, solutions_file)
            print(f"  Created prompts saved to {prompt_file}")

            # For the first chapter, also create sample Q&A pairs
            if i == 0:
                sample_file = self.create_sample_qa_pairs(chapter_file)
                print(f"  Created sample Q&A pairs at {sample_file}")

                # Create template for GPT fine-tuning
                template_file = self.create_qa_template()
                gpt_file = self.prepare_gpt_finetuning_format(sample_file)
                print(f"  Created GPT fine-tuning template at {gpt_file}")

        print("\nProcessing complete!")
        print("Next steps:")
        print("1. Review the prompts in the *_prompts.txt files")
        print("2. Use these prompts with your local LLaMA 2 model to generate responses")
        print("3. Convert the responses to the JSON format as shown in sample_qa_pairs.json")
        print("4. Use the prepare_gpt_finetuning_format method to convert your JSON to GPT fine-tuning format")


# Example usage
if __name__ == "__main__":
    processor = TextbookProcessor()
    processor.process_all("input_textbook/biology_textbook.txt")