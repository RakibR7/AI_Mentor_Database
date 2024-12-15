import json
import random
from pathlib import Path

def validate_training_example(example):
    """
    Validate a single training example.
    """
    try:
        messages = example['messages']
        assert len(messages) == 3, "Incorrect number of messages in the example"
        assert messages[0]['role'] == 'system', "First message should have the 'system' role"
        assert messages[1]['role'] == 'user', "Second message should have the 'user' role"
        assert messages[2]['role'] == 'assistant', "Third message should have the 'assistant' role"
        assert len(messages[2]['content']) >= 200, "Assistant response is too short"
        return True
    except (KeyError, AssertionError):
        return False

def validate_training_data(file_path, sample_size=100):
    """
    Validate the training data file.
    """
    valid_examples = []
    total_examples = 0

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            total_examples += 1
            example = json.loads(line.strip())
            if validate_training_example(example):
                valid_examples.append(example)

    # Calculate validation statistics
    num_valid_examples = len(valid_examples)
    validity_percentage = num_valid_examples / total_examples * 100

    print(f"Total examples: {total_examples}")
    print(f"Valid examples: {num_valid_examples}")
    print(f"Validity percentage: {validity_percentage:.2f}%")

    # Randomly sample examples for manual review
    sample_examples = random.sample(valid_examples, min(sample_size, num_valid_examples))

    print(f"\nRandomly sampled {len(sample_examples)} examples for manual review:")
    for i, example in enumerate(sample_examples, start=1):
        print(f"Example {i}:")
        print(json.dumps(example, indent=2))
        print()

    return validity_percentage

if __name__ == "__main__":
    # Specify the path to your training data file
    training_data_file = "../Wikipedia_dumps/training_data_2/biology_training_data_high_quality.jsonl"

    # Validate the training data
    validity_score = validate_training_data(training_data_file)

    if validity_score >= 90:
        print("Training data validation passed. The data is of high quality.")
    else:
        print("Training data validation failed. Please review and improve the data quality.")

