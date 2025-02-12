import json
import tiktoken
import argparse

# For special csv jsonl type (should be the same as the other one)
# run this python estimator.py DR1_CSV5_Shortened_20.csv.jsonl --model gpt-3.5-turbo --budget 5


# Updated pricing (as of July 2024)
PRICING = {
    "gpt-3.5-turbo": 0.0080,  # $ per 1K tokens
    "gpt-4o": 0.0300
}


def count_tokens(text):
    """Count tokens using OpenAI's tokenizer"""
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


def analyze_file(jsonl_file, model_name="gpt-3.5-turbo", budget=5):
    """Analyze JSONL file with budget-aware reporting"""
    total_examples = 0
    total_tokens = 0
    token_distribution = {
        "system": 0,
        "user": 0,
        "assistant": 0
    }

    try:
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                example_tokens = 0

                for msg in data["messages"]:
                    role = msg["role"]
                    content = msg["content"]
                    tokens = count_tokens(content)
                    token_distribution[role] += tokens
                    example_tokens += tokens

                total_tokens += example_tokens
                total_examples += 1

    except Exception as e:
        print(f"Error: {str(e)}")
        return

    # Calculate metrics
    avg_tokens = total_tokens / total_examples if total_examples > 0 else 0
    estimated_cost = (total_tokens / 1000) * PRICING[model_name]
    max_examples = int((budget / PRICING[model_name]) * 1000 / avg_tokens) if avg_tokens > 0 else 0

    # Print detailed report
    print(f"\n=== Analysis for {jsonl_file} ===")
    print(f"Total examples: {total_examples}")
    print(f"Total tokens: {total_tokens}")
    print(f"Average tokens/example: {avg_tokens:.1f}")
    print(f"\nToken distribution:")
    print(f"  System: {token_distribution['system']} ({token_distribution['system'] / total_tokens:.1%})")
    print(f"  User: {token_distribution['user']} ({token_distribution['user'] / total_tokens:.1%})")
    print(f"  Assistant: {token_distribution['assistant']} ({token_distribution['assistant'] / total_tokens:.1%})")
    print(f"\nEstimated cost ({model_name}): ${estimated_cost:.2f}")
    print(f"Max examples for ${budget}: ~{max_examples} (at current avg tokens/example)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Budget-aware Fine-Tuning Cost Estimator")
    parser.add_argument("jsonl_file", help="Path to JSONL file (e.g., DR1_CSV5_Shortened_20.jsonl)")
    parser.add_argument("--model", default="gpt-3.5-turbo", choices=["gpt-3.5-turbo", "gpt-4o"])
    parser.add_argument("--budget", type=float, default=5.0, help="Target budget in USD")

    args = parser.parse_args()

    analyze_file(args.jsonl_file, args.model, args.budget)