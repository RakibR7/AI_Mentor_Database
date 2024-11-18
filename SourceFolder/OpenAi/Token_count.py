import tiktoken


def count_tokens_in_jsonl_file(file_path):
    encoding = tiktoken.get_encoding("cl100k_base")

    total_tokens = 0
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            tokens = encoding.encode(line)
            total_tokens += len(tokens)

    return total_tokens


# Specify the path to your training data file
file_path = "Wikipedia_dumps/training_data_2/biology_training_data.jsonl"

total_tokens = count_tokens_in_jsonl_file(file_path)
print(f"Total tokens in the file: {total_tokens}")