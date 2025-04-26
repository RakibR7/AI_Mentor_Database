import re
import difflib

def extract_questions_from_solution(solution_text):
    chapter_pattern = re.compile(r'Chapter\s+(\d+)')
    question_pattern = re.compile(r'(\d+)\s+(?:Figure\s+[\d\.]+\s+)?(.*?)(?=\s+\d+\s+|$)', re.DOTALL)
    
    questions_by_chapter = {}
    current_chapter = None

    for line in solution_text.split('\n'):
        chapter_match = chapter_pattern.search(line)
        if chapter_match:
            current_chapter = int(chapter_match.group(1))
            questions_by_chapter[current_chapter] = {}
            
        elif current_chapter is not None:
            for match in question_pattern.finditer(line):
                question_num = int(match.group(1))
                answer_text = match.group(2).strip()
                questions_by_chapter[current_chapter][question_num] = answer_text
    
    return questions_by_chapter

def extract_content_from_cleaned_file(cleaned_text):
    chapter_pattern = re.compile(r'Chapter\s+(\d+)(?:\s+•\s+(.+?))?(?:\n|$)')
    figure_pattern = re.compile(r'Figure\s+([\d\.]+)')
    
    content_by_chapter = {}
    current_chapter = None
    current_content = ""
    
    lines = cleaned_text.split('\n')
    i = 0
    while i < len(lines):
        line = lines[i]
        chapter_match = chapter_pattern.search(line)
        
        if chapter_match:
            if current_chapter is not None:
                content_by_chapter[current_chapter] = current_content

            current_chapter = int(chapter_match.group(1))
            current_content = line + "\n"
        
        elif current_chapter is not None:
            current_content += line + "\n"
        
        i += 1

    if current_chapter is not None:
        content_by_chapter[current_chapter] = current_content
    
    return content_by_chapter

def match_questions_to_content(questions_by_chapter, content_by_chapter):
    matches = {}
    
    for chapter in questions_by_chapter:
        if chapter not in content_by_chapter:
            print(f"Chapter {chapter} not found in cleaned file")
            continue
        
        chapter_content = content_by_chapter[chapter]
        matches[chapter] = {}
        
        for question_num, answer in questions_by_chapter[chapter].items():
            figure_match = re.search(r'Figure\s+([\d\.]+)', answer)
            if figure_match:
                figure_ref = figure_match.group(0)
                paragraphs = chapter_content.split('\n\n')
                for paragraph in paragraphs:
                    if figure_ref in paragraph:
                        matches[chapter][question_num] = {
                            'answer': answer,
                            'content': paragraph.strip()
                        }
                        break
            else:
                key_terms = extract_key_terms(answer)
                best_match = None
                best_score = 0
                
                paragraphs = chapter_content.split('\n\n')
                for paragraph in paragraphs:
                    score = sum(1 for term in key_terms if term.lower() in paragraph.lower())
                    if score > best_score:
                        best_score = score
                        best_match = paragraph
                
                if best_match and best_score > 0:
                    matches[chapter][question_num] = {
                        'answer': answer,
                        'content': best_match.strip()
                    }
    
    return matches

def extract_key_terms(text):
    common_words = {'the', 'and', 'a', 'to', 'of', 'in', 'is', 'it', 'that', 'for', 'as', 'with', 'by', 'on', 'are'}
    terms = []
    for word in re.findall(r'\b\w+\b', text):
        if word.lower() not in common_words and len(word) > 3:
            terms.append(word)
    
    return terms

def format_matches(matches):
    result = []
    
    for chapter in sorted(matches.keys()):
        result.append(f"CHAPTER {chapter} MATCHES\n{'='*50}")
        
        for question_num in sorted(matches[chapter].keys()):
            match = matches[chapter][question_num]
            result.append(f"Question {question_num}")
            result.append(f"Answer: {match['answer']}")
            result.append(f"Matched Content: {match['content']}")
            result.append('-'*50)
        
        result.append('\n')
    
    return '\n'.join(result)

def main(solution_file, cleaned_file):
    with open(solution_file, 'r', encoding='utf-8') as f:
        solution_text = f.read()
    
    with open(cleaned_file, 'r', encoding='utf-8') as f:
        cleaned_text = f.read()

    questions_by_chapter = extract_questions_from_solution(solution_text)
    content_by_chapter = extract_content_from_cleaned_file(cleaned_text)
    matches = match_questions_to_content(questions_by_chapter, content_by_chapter)
    formatted_matches = format_matches(matches)
    with open('question_matches.txt', 'w', encoding='utf-8') as f:
        f.write(formatted_matches)
    
    print(f"Processed {sum(len(questions) for questions in questions_by_chapter.values())} questions")
    print(f"Found {sum(len(matches.get(chapter, {})) for chapter in questions_by_chapter)} matches")
    print("Results written to question_matches.txt")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python question_matcher.py solution_file.txt cleaned_file.txt")
        sys.exit(1)
    
    main(sys.argv[1], sys.argv[2])
