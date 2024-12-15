import json

# File path to the JSONL file
file_path = "CSV_V4.jsonl"


def validate_jsonl(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line_num, line in enumerate(file, start=1):
                try:
                    # Parse the JSON line
                    data = json.loads(line.strip())

                    # Check required keys
                    if not all(key in data for key in ["prompt", "completion"]):
                        print(f"Line {line_num}: Missing required keys.")
                        continue

                    # Check for non-empty values
                    if not data["prompt"].strip() or not data["completion"].strip():
                        print(f"Line {line_num}: Empty 'prompt' or 'completion'.")
                        continue

                except json.JSONDecodeError as e:
                    print(f"Line {line_num}: Invalid JSON - {e}")

        print("Validation completed!")

    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


# Run the validation
validate_jsonl(file_path)
