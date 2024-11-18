import json

def find_article_in_jsonl(dump_file, article_title):
    with open(dump_file, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                article = json.loads(line.strip())
                if article.get('title', '').lower() == article_title.lower():
                    return article  # Found the article
            except json.JSONDecodeError:
                continue
    return None  # Article not found

article = find_article_in_jsonl('SourceFolder/OpenAi/Wikipedia_dumps/training_data_2/biology_training_data_high_quality_2.5.jsonl', 'Photosynthesis')

if article:
    print(f"Found the article: {article['title']}")
    print(f"URL: {article['url']}")
    print(f"Content: {article['text'][:1000]}")  # First 1000 characters of the article content
else:
    print("Article not found")
