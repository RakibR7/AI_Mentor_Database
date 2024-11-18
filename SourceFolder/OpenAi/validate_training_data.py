import json
import sys
from typing import List, Dict
import os


def validate_messages_format(messages: List[Dict]) -> bool:
    """
    Validate that messages follow the required format for ChatGPT fine-tuning.
    """
    if not isinstance(messages, list):
        return False

    required_roles = {"system", "user", "assistant"}
    required_keys = {"role", "content"}

    # Check each message
    for msg in messages:
        # Check if message is a dict with required keys
        if not isinstance(msg, dict):
            return False
        if not all(key in msg for key in required_keys):
            return False

        # Check if role is valid
        if msg["role"] not in required_roles:
            return False

        # Check if content is string and not empty
        if not isinstance(msg["content"], str) or not msg["content"].strip():
            return False

    return True


def validate_jsonl_file(file_path: str) -> tuple:
    """
    Validate a JSONL file for ChatGPT fine-tuning.
    Returns (is_valid, error_details)
    """
    if not os.path.exists(file_path):
        return False, f"File not found: {file_path}"

    valid_entries = 0
    errors = []
    line_number = 0

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_number, line in enumerate(f, 1):
                try:
                    # Parse JSON
                    entry = json.loads(line.strip())

                    # Check basic structure
                    if not isinstance(entry, dict):
                        errors.append(f"Line {line_number}: Entry must be a dictionary")
                        continue

                    if "messages" not in entry:
                        errors.append(f"Line {line_number}: Missing 'messages' key")
                        continue

                    # Validate messages format
                    if not validate_messages_format(entry["messages"]):
                        errors.append(f"Line {line_number}: Invalid messages format")
                        continue

                    # Check message sequence
                    messages = entry["messages"]
                    if len(messages) < 2:
                        errors.append(f"Line {line_number}: Must have at least 2 messages")
                        continue

                    # Validate system message is first (optional but recommended)
                    if messages[0]["role"] != "system":
                        errors.append(f"Line {line_number}: First message should be system message")
                        continue

                    # Check token count (approximate)
                    total_tokens = sum(len(msg["content"].split()) for msg in messages)
                    if total_tokens > 4096:
                        errors.append(f"Line {line_number}: Total tokens may exceed 4096 limit")
                        continue

                    valid_entries += 1

                except json.JSONDecodeError as e:
                    errors.append(f"Line {line_number}: Invalid JSON - {str(e)}")
                except Exception as e:
                    errors.append(f"Line {line_number}: Unexpected error - {str(e)}")

    except Exception as e:
        return False, f"Failed to read file: {str(e)}"

    # Print summary
    print(f"\nValidation Summary:")
    print(f"Total entries processed: {line_number}")
    print(f"Valid entries: {valid_entries}")
    print(f"Errors found: {len(errors)}")

    if errors:
        print("\nDetailed Errors:")
        for error in errors[:10]:  # Show first 10 errors
            print(error)
        if len(errors) > 10:
            print(f"...and {len(errors) - 10} more errors")

    # Calculate basic statistics
    if valid_entries > 0:
        print("\nFile Statistics:")
        with open(file_path, 'r', encoding='utf-8') as f:
            first_entry = json.loads(f.readline())
            print(f"Sample system message: {first_entry['messages'][0]['content'][:100]}...")
            print(
                f"Average messages per entry: {sum(len(json.loads(line)['messages']) for line in f) / valid_entries:.1f}")

    is_valid = len(errors) == 0 and valid_entries > 0
    return is_valid, errors


if __name__ == "__main__":
    # Specify your JSONL file path
    file_path = "Wikipedia_dumps/training_data_2/biology_training_data.jsonl"

    print(f"Validating file: {file_path}")
    is_valid, errors = validate_jsonl_file(file_path)

    if is_valid:
        print("\nSuccess! File is valid for fine-tuning.")
        print("You can proceed with uploading this file to OpenAI.")
    else:
        print("\nValidation failed!")
        print("Please fix the errors before uploading to OpenAI.")

    # Additional file size check
    file_size = os.path.getsize(file_path) / (1024 * 1024)  # Convert to MB
    print(f"\nFile size: {file_size:.2f} MB")
    if file_size < 0.1:
        print("Warning: File seems very small. Make sure it contains enough training examples.")
    elif file_size > 100:
        print("Warning: File is quite large. Consider splitting into smaller files for easier handling.")