import json


def search_for_article(file_path, search_title):
    """
    Search for a specific article title in a JSONL file.

    Parameters:
    - file_path (str): Path to the JSONL file.
    - search_title (str): The title to search for (case-insensitive).

    Returns:
    - List of matching articles or an empty list if not found.
    """
    matching_articles = []
    search_title_lower = search_title.lower()

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                try:
                    article = json.loads(line.strip())
                    title = article.get('messages', [{}])[1].get('content', '').lower()
                    if search_title_lower in title:
                        matching_articles.append(article)
                except json.JSONDecodeError:
                    print("Skipping invalid JSON entry.")
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

    return matching_articles


if __name__ == "__main__":
    # Replace with your actual file path
    file_path = "C:\\Users\\Rakib\\Documents\\Ai Mentor\\SourceFolder\\OpenAi\\Wikipedia_dumps\\training_data_2\\biology_training_data_high_quality_2.6.jsonl"
    search_title = "Photosynthesis"

    results = search_for_article(file_path, search_title)
    if results:
        print(f"Found {len(results)} article(s) related to '{search_title}':")
        for i, article in enumerate(results, start=1):
            title = article['messages'][1]['content']
            print(f"{i}. Title: {title}")
    else:
        print(f"No articles found with title '{search_title}'.")
