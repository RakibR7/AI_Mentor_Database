import os
import json
import subprocess
import re
from pathlib import Path


def process_prompts_with_llama(prompts_directory, output_directory):
    # Create output directory if it doesn't exist
    output_dir = Path(output_directory)
    output_dir.mkdir(exist_ok=True)

    # Find all prompt files
    prompt_files = list(Path(prompts_directory).glob("*_prompts.txt"))

    for prompt_file in prompt_files:
        print(f"Processing {prompt_file.name}...")
        chapter_qa_pairs = []

        # Read the prompt file
        with open(prompt_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # Split by the separator
        prompts = content.split("=" * 80)

        for i, prompt in enumerate(prompts):
            if not prompt.strip():
                continue

            print(f"  Running prompt {i + 1}/{len(prompts)}...")

            # Write prompt to temporary file
            with open("temp_prompt.txt", 'w', encoding='utf-8') as f:
                f.write(prompt)

            # Call LLaMA 2 (adjust command based on your setup)
            result = subprocess.run(
                ["llama-cli", "--model", "path/to/model", "--prompt-file", "temp_prompt.txt"],
                capture_output=True,
                text=True
            )

            # Extract JSON from response
            response = result.stdout
            try:
                # Look for JSON content more robustly
                # First try to find content within square brackets
                json_pattern = r'\[(.*?)\]'
                match = re.search(json_pattern, response, re.DOTALL)

                if match:
                    try:
                        qa_json = json.loads('[' + match.group(1) + ']')
                        chapter_qa_pairs.extend(qa_json)
                        print(f"    Found {len(qa_json)} Q&A pairs in this prompt")
                    except json.JSONDecodeError:
                        print("    Invalid JSON format within brackets, trying alternative extraction...")

                # If that fails, try to find any JSON-like structure
                if not match or len(chapter_qa_pairs) == 0:
                    # Look for JSON objects (with curly braces)
                    json_objects = re.findall(r'({[^{}]*"question":[^{}]*"answer":[^{}]*})', response, re.DOTALL)

                    if json_objects:
                        for obj_str in json_objects:
                            try:
                                qa_item = json.loads(obj_str)
                                chapter_qa_pairs.append(qa_item)
                                print(f"    Found a Q&A pair using object extraction")
                            except json.JSONDecodeError:
                                print(f"    Failed to parse potential JSON object")

            except Exception as e:
                print(f"  Error processing response: {e}")
                # Save the problematic response for debugging
                debug_file = output_dir / f"debug_{prompt_file.stem}_{i}.txt"
                with open(debug_file, 'w', encoding='utf-8') as f:
                    f.write(response)
                print(f"  Saved problematic response to {debug_file}")

        # Save chapter Q&A pairs
        output_file = output_dir / f"{prompt_file.stem.replace('_prompts', '_qa')}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(chapter_qa_pairs, f, indent=2)

        print(f"  Saved {len(chapter_qa_pairs)} Q&A pairs to {output_file}")


# Example usage
process_prompts_with_llama("output_qa_old", "qa_results")