import re
import difflib
from collections import defaultdict
import json

def extract_questions_from_solution(solution_text):
    """
    Extract questions and answers from the solution manual.
    Returns a dictionary with chapter numbers as keys and a list of (question_num, answer) tuples as values.
    """
    chapter_pattern = re.compile(r'Chapter\s+(\d+)', re.IGNORECASE)
    questions_by_chapter = defaultdict(list)
    current_chapter = None

    lines = solution_text.strip().split('\n')
    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Detect new chapter
        chapter_match = chapter_pattern.match(line)
        if chapter_match:
            current_chapter = int(chapter_match.group(1))
            i += 1
            continue

        if current_chapter is not None and line:
            # Optionally pull out the figure (to keep for context)
            figure_ref = ''
            if 'Figure' in line:
                fig_match = re.search(r'(Figure\s+\d+\.\d+)', line)
                if fig_match:
                    figure_ref = fig_match.group(1)
                    line = line.replace(figure_ref, '').strip()  # remove it so it's not treated as an answer

            # Match multiple inline answers like "1: C; 2: F; 3: A"
            multi_qa_matches = re.findall(r'(\d+):?\s+([^;\n]+)', line)
            if multi_qa_matches:
                for match in multi_qa_matches:
                    question_num = int(match[0])
                    answer_text = match[1].strip().rstrip('.')
                    # Prepend the figure reference if available
                    final_answer = f"{figure_ref} {answer_text}".strip()
                    questions_by_chapter[current_chapter].append((question_num, final_answer))
                i += 1
                continue


            # Match single-line formats like "4 B" or "12 Answers will vary..."
            single_match = re.match(r'^(\d+)[\s\.]+(.+)', line)
            if single_match:
                question_num = int(single_match.group(1))
                answer_start = single_match.group(2).strip()

                # Try to gather full answer across multiple lines (until next question or chapter)
                answer_lines = [answer_start]
                j = i + 1
                while j < len(lines):
                    next_line = lines[j].strip()
                    if re.match(r'^(\d+)[\s\.]+', next_line) or chapter_pattern.match(next_line):
                        break
                    if next_line:
                        answer_lines.append(next_line)
                    j += 1

                full_answer = ' '.join(answer_lines)
                questions_by_chapter[current_chapter].append((question_num, full_answer))
                i = j
                continue

        i += 1

    return questions_by_chapter


def extract_questions_from_cleaned_file(cleaned_text):
    """
    Extract questions from the cleaned file.
    Returns a dictionary with chapter numbers as keys and a list of (question_num, question_text) tuples as values.
    """
    # Pattern to match chapter headers
    chapter_pattern = re.compile(r'CHAPTER\s+(\d+)')
    
    # Pattern to match question numbers
    question_pattern = re.compile(r'^\s*(\d+)\.\s+(.*?)$', re.MULTILINE)
    
    # Dictionary to store questions by chapter
    questions_by_chapter = defaultdict(list)
    current_chapter = None
    
    # Find all chapters
    chapter_matches = list(chapter_pattern.finditer(cleaned_text))
    
    # Process each chapter
    for i, chapter_match in enumerate(chapter_matches):
        chapter_num = int(chapter_match.group(1))
        
        # Determine the end of this chapter (start of next chapter or end of text)
        chapter_start = chapter_match.start()
        if i < len(chapter_matches) - 1:
            chapter_end = chapter_matches[i + 1].start()
        else:
            chapter_end = len(cleaned_text)
        
        # Extract chapter text
        chapter_text = cleaned_text[chapter_start:chapter_end]
        
        # Find all questions in this chapter
        for question_match in question_pattern.finditer(chapter_text):
            question_num = int(question_match.group(1))
            question_text = question_match.group(2).strip()
            
            # Add to our collection
            questions_by_chapter[chapter_num].append((question_num, question_text))
    
    return questions_by_chapter

def extract_critical_thinking_questions(cleaned_text):
    """
    Extract critical thinking questions which have a different format.
    """
    ct_questions = {}
    
    # Find Critical Thinking Questions sections
    ct_sections = re.finditer(r'CRITICAL\s+THINKING\s+QUESTIONS\s+(.*?)(?=CHAPTER|\Z)', cleaned_text, re.DOTALL)
    
    for section in ct_sections:
        section_text = section.group(1)
        
        # Find which chapter this section belongs to
        chapter_match = re.search(r'CHAPTER\s+(\d+)', cleaned_text[:section.start()], re.DOTALL)
        if not chapter_match:
            continue
            
        chapter_num = int(chapter_match.group(1))
        ct_questions[chapter_num] = []
        
        # Extract questions with their numbers
        question_matches = re.finditer(r'(\d+)\.\s+(.*?)(?=\d+\.\s+|\Z)', section_text, re.DOTALL)
        
        for question in question_matches:
            question_num = int(question.group(1))
            question_text = question.group(2).strip()
            ct_questions[chapter_num].append((question_num, question_text))
    
    return ct_questions

def match_figure_references(solution_answers, cleaned_text):
    """
    Match figure references between solution manual and cleaned file.
    """
    matches = {}
    
    # Find all figure references in cleaned file
    figure_pattern = re.compile(r'(Figure\s+(\d+\.\d+).*?)(?=Figure|\Z)', re.DOTALL)
    figures_in_text = {}
    
    for figure_match in figure_pattern.finditer(cleaned_text):
        figure_num = figure_match.group(2)
        figures_in_text[figure_num] = figure_match.group(1).strip()
    
    # Match solution answers that reference figures
    for chapter, questions in solution_answers.items():
        matches[chapter] = {}
        
        for question_num, answer in questions:
            figure_ref_match = re.search(r'Figure\s+(\d+\.\d+)', answer)
            if figure_ref_match:
                figure_num = figure_ref_match.group(1)
                if figure_num in figures_in_text:
                    matches[chapter][question_num] = {
                        'answer': answer,
                        'content': figures_in_text[figure_num]
                    }
    
    return matches

def match_by_question_number(solution_answers, cleaned_questions, critical_thinking_questions):
    """
    Match questions by their number between solution manual and cleaned file.
    """
    matches = {}
    
    for chapter, questions in solution_answers.items():
        matches[chapter] = {}
        
        # Regular questions
        chapter_questions = {q[0]: q[1] for q in cleaned_questions.get(chapter, [])}
        
        # Critical thinking questions
        ct_questions = {q[0]: q[1] for q in critical_thinking_questions.get(chapter, [])}
        
        # Combine all questions for this chapter
        all_questions = {**chapter_questions, **ct_questions}
        
        for question_num, answer in questions:
            if question_num in all_questions:
                matches[chapter][question_num] = {
                    'answer': answer,
                    'content': all_questions[question_num]
                }
    
    return matches

def match_by_content_similarity(solution_answers, cleaned_text, existing_matches):
    """
    Match questions by content similarity for those not already matched.
    """
    additional_matches = {}
    
    # Split cleaned text into paragraphs
    paragraphs = re.split(r'\n\s*\n', cleaned_text)
    
    for chapter, questions in solution_answers.items():
        additional_matches[chapter] = {}
        
        for question_num, answer in questions:
            # Skip if already matched
            if chapter in existing_matches and question_num in existing_matches[chapter]:
                continue
                
            # Extract key terms from answer
            key_terms = extract_key_terms(answer)
            
            # Find best matching paragraph
            best_match = None
            best_score = 0
            
            for paragraph in paragraphs:
                # Calculate similarity score
                score = sum(1 for term in key_terms if term.lower() in paragraph.lower())
                
                if score > best_score:
                    best_score = score
                    best_match = paragraph
            
            if best_match and best_score > 0:
                additional_matches[chapter][question_num] = {
                    'answer': answer,
                    'content': best_match.strip()
                }
    
    return additional_matches

def extract_key_terms(text):
    """
    Extract key terms from text that might be useful for matching.
    """
    # Remove common words
    common_words = {'the', 'and', 'a', 'to', 'of', 'in', 'is', 'it', 'that', 'for', 'as', 'with', 'by', 'on', 'are'}
    
    # Split by non-alphanumeric characters and filter out common words and short terms
    terms = []
    for word in re.findall(r'\b\w+\b', text):
        if word.lower() not in common_words and len(word) > 3:
            terms.append(word)
    
    return terms

def merge_matches(matches_dict_list):
    """
    Merge multiple match dictionaries.
    """
    merged = {}
    
    for matches in matches_dict_list:
        for chapter, questions in matches.items():
            if chapter not in merged:
                merged[chapter] = {}
                
            for question_num, match_data in questions.items():
                merged[chapter][question_num] = match_data
    
    return merged

def format_matches(matches):
    """
    Format the matches for easy reading.
    """
    result = []
    
    for chapter in sorted(matches.keys()):
        result.append(f"CHAPTER {chapter} MATCHES")
        result.append("=" * 50)
        
        for question_num in sorted(matches[chapter].keys()):
            match = matches[chapter][question_num]
            result.append(f"Question {question_num}")
            result.append(f"Answer: {match['answer']}")
            result.append(f"Matched Content: {match['content']}")
            result.append('-' * 50)
        
        result.append('\n')
    
    return '\n'.join(result)

def output_matches_jsonl(matches, output_file):
    """
    Output the matches in JSONL format.
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        for chapter in sorted(matches.keys()):
            for question_num in sorted(matches[chapter].keys()):
                match = matches[chapter][question_num]
                json_obj = {
                    'chapter': chapter,
                    'question_num': question_num,
                    'answer': match['answer'],
                    'content': match['content']
                }
                f.write(json.dumps(json_obj) + '\n')

def main(solution_file, cleaned_file):
    """
    Main function to process files and match questions to answers.
    """
    # Read input files
    with open(solution_file, 'r', encoding='utf-8') as f:
        solution_text = f.read()
    
    with open(cleaned_file, 'r', encoding='utf-8') as f:
        cleaned_text = f.read()
    
    # Extract questions and answers from solution manual
    solution_answers = extract_questions_from_solution(solution_text)
    
    # Extract questions from cleaned file
    cleaned_questions = extract_questions_from_cleaned_file(cleaned_text)
    critical_thinking_questions = extract_critical_thinking_questions(cleaned_text)
    
    # Match by figure references
    figure_matches = match_figure_references(solution_answers, cleaned_text)
    
    # Match by question number
    question_number_matches = match_by_question_number(solution_answers, cleaned_questions, critical_thinking_questions)
    
    # Match remaining by content similarity
    existing_matches = merge_matches([figure_matches, question_number_matches])
    content_matches = match_by_content_similarity(solution_answers, cleaned_text, existing_matches)
    
    # Merge all matches
    all_matches = merge_matches([figure_matches, question_number_matches, content_matches])
    
    # Format and output results
    formatted_matches = format_matches(all_matches)
    
    # Write results to file
    with open('question_matches.txt', 'w', encoding='utf-8') as f:
        f.write(formatted_matches)
    
    # Output results in JSONL format
    output_matches_jsonl(all_matches, 'question_matches.jsonl')
    
    # Count matches
    total_questions = sum(len(questions) for questions in solution_answers.values())
    total_matches = sum(len(matches) for matches in all_matches.values())
    
    print(f"Processed {total_questions} questions")
    print(f"Found {total_matches} matches")
    print("Results written to question_matches.txt and question_matches.jsonl")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python question_matcher.py solution_file.txt cleaned_file.txt")
        sys.exit(1)
    
    main(sys.argv[1], sys.argv[2])
