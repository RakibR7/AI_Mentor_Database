import os
import json
from pathlib import Path

def search_for_article(base_directory, article_title):
    """
    Search for an article by title across all files in the Wikipedia dump.
    """
    found_articles = []
    article_title_lower = article_title.lower()

    # Iterate over all subdirectories and files
    for subdir in Path(base_directory).iterdir():
        if subdir.is_dir():
            print(f"Processing directory: {subdir.name}")

            for wiki_file in sorted(subdir.glob("wiki_*")):
                print(f"Searching in file: {wiki_file.name}")

                # Read each wiki file
                with open(wiki_file, 'r', encoding='utf-8') as file:
                    for line in file:
                        try:
                            article = json.loads(line.strip())
                            title = article.get('title', '').lower()

                            # Check if the title matches
                            if article_title_lower in title:
                                found_articles.append(article)
                                print(f"Found article: {title} - URL: {article.get('url')}")
                                break  # Stop searching after finding the first match

                        except (json.JSONDecodeError, KeyError):
                            continue

    return found_articles


if __name__ == "__main__":
    base_dir = r"C:\\Users\\Rakib\\Documents\\Ai Mentor\\SourceFolder\\OpenAi\\Wikipedia_dumps\\Wiki_extracted_data"
    article_title = "Photosynthesis"

    articles = search_for_article(base_dir, article_title)

    if articles:
        print(f"\nFound {len(articles)} matching articles.")
        # Optionally, write results to a file or display the first article's text
        for article in articles[:5]:  # Displaying top 5 found articles
            print(f"Title: {article.get('title')}, URL: {article.get('url')}")
    else:
        print(f"\nNo articles found for title: {article_title}.")
