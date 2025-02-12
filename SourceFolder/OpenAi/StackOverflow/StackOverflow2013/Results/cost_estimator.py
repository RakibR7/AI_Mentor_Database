import json
import tiktoken
import argparse

# Install required library: pip install tiktoken

# OpenAI Pricing (update these values as needed)
PRICING = {
    "gpt-3.5-turbo": 0.0080,  # $ per 1K tokens
    "gpt-4o": 0.0300,  # $ per 1K tokens
    "gpt-4": 0.0600  # $ per 1K tokens
}


def count_tokens(text, encoding_name="cl100k_base"):
    """Count tokens using OpenAI's tokenizer"""
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(text))


def estimate_cost(jsonl_file, model_name="gpt-4o", epochs=1):
    """Estimate fine-tuning costs for a JSONL file"""
    total_tokens = 0
    example_counts = []

    try:
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                message_text = " ".join([msg["content"] for msg in data["messages"]])
                tokens = count_tokens(message_text)
                total_tokens += tokens
                example_counts.append(tokens)

                # Show first 3 examples for verification
                if len(example_counts) <= 3:
                    print(f"Example {len(example_counts)}: {tokens} tokens")

    except Exception as e:
        print(f"Error reading file: {e}")
        return

    # Calculate totals
    total_examples = len(example_counts)
    avg_tokens = sum(example_counts) / total_examples if total_examples > 0 else 0
    total_cost = (total_tokens * epochs / 1000) * PRICING.get(model_name, PRICING["gpt-4o"])

    print("\n=== Estimation Results ===")
    print(f"Total examples: {total_examples}")
    print(f"Total tokens: {total_tokens}")
    print(f"Average tokens/example: {avg_tokens:.1f}")
    print(f"Estimated cost ({model_name}, {epochs} epochs): ${total_cost:.2f}")
    print(f"Required quota: ${total_cost:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Estimate OpenAI Fine-Tuning Costs")
    parser.add_argument("jsonl_file", help="Path to JSONL training file")
    parser.add_argument("--model", default="gpt-4o",
                        choices=["gpt-4o", "gpt-4", "gpt-3.5-turbo"],
                        help="Model to use for pricing")
    parser.add_argument("--epochs", type=int, default=1,
                        help="Number of training epochs")

    args = parser.parse_args()

    if args.model not in PRICING:
        print(f"Warning: No pricing data for {args.model}. Using gpt-4o pricing.")
        args.model = "gpt-4o"

    estimate_cost(args.jsonl_file, args.model, args.epochs)