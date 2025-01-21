import pyodbc
import json
from tqdm import tqdm
import logging
import os
from datetime import datetime
import tiktoken
from typing import List, Dict, Any

class StackOverflowProcessor:
    def __init__(self, server: str, database: str):
        self.encoding = tiktoken.get_encoding("cl100k_base")
        self.total_tokens = 0
        self.max_tokens = 2000000  # 2 million token limit
        
        # Initialize SQL connection
        self.conn_str = (
            f'Driver={{SQL Server}};'
            f'Server={server};'
            f'Database={database};'
            f'Trusted_Connection=yes;'  # Windows authentication
        )
        
        self.setup_logging()
        
    def setup_logging(self):
        """Configure logging"""
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f"{log_dir}/stackoverflow_processing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
                logging.StreamHandler()
            ]
        )

    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.encoding.encode(text))

    def get_question_with_answers(self) -> List[Dict[str, Any]]:
        """
        Extract questions with their answers, focusing on well-received content
        Returns list of dictionaries containing questions and answers
        """
        query = """
        SELECT 
            q.Id as QuestionId,
            q.Title as QuestionTitle,
            q.Body as QuestionBody,
            q.Tags,
            q.Score as QuestionScore,
            a.Id as AnswerId,
            a.Body as AnswerBody,
            a.Score as AnswerScore
        FROM Posts q
        LEFT JOIN Posts a ON q.Id = a.ParentId
        WHERE 
            q.PostTypeId = 1  -- Questions
            AND q.Score >= 5  -- Well-received questions
            AND (a.PostTypeId = 2 OR a.PostTypeId IS NULL)  -- Answers
            AND (a.Score >= 5 OR a.Score IS NULL)  -- Well-received answers
        ORDER BY 
            q.Score DESC,
            a.Score DESC
        """
        
        try:
            with pyodbc.connect(self.conn_str) as conn:
                cursor = conn.cursor()
                cursor.execute(query)
                
                current_question = None
                qa_pairs = []
                
                for row in cursor:
                    if current_question is None or current_question['question_id'] != row.QuestionId:
                        if current_question is not None:
                            qa_pairs.append(current_question)
                        
                        current_question = {
                            'question_id': row.QuestionId,
                            'title': row.QuestionTitle,
                            'question': row.QuestionBody,
                            'tags': row.Tags,
                            'answers': []
                        }
                    
                    if row.AnswerId is not None:
                        current_question['answers'].append({
                            'answer_id': row.AnswerId,
                            'answer': row.AnswerBody,
                            'score': row.AnswerScore
                        })
                
                if current_question is not None:
                    qa_pairs.append(current_question)
                
                return qa_pairs
                
        except Exception as e:
            logging.error(f"Database error: {str(e)}")
            return []

    def convert_to_training_format(self, qa_pairs: List[Dict[str, Any]], output_file: str):
        """Convert Q&A pairs to JSONL training format"""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                for qa in tqdm(qa_pairs, desc="Converting to training format"):
                    # Create conversation flow
                    entry = {
                        "messages": [
                            {
                                "role": "system",
                                "content": "You are a knowledgeable programming mentor helping users solve technical problems."
                            },
                            {
                                "role": "user",
                                "content": f"{qa['title']}\n\n{qa['question']}"
                            }
                        ]
                    }
                    
                    # Add best answer if available
                    if qa['answers']:
                        best_answer = max(qa['answers'], key=lambda x: x['score'])
                        entry['messages'].append({
                            "role": "assistant",
                            "content": best_answer['answer']
                        })
                        
                        # Check token count
                        tokens = self.count_tokens(json.dumps(entry))
                        if self.total_tokens + tokens <= self.max_tokens:
                            f.write(json.dumps(entry) + '\n')
                            self.total_tokens += tokens
                        else:
                            logging.info("Reached token limit. Stopping conversion.")
                            break
                    
            logging.info(f"Processed {len(qa_pairs)} Q&A pairs")
            logging.info(f"Total tokens: {self.total_tokens:,}")
            
        except Exception as e:
            logging.error(f"Error converting to training format: {str(e)}")

    def process_database(self, output_file: str):
        """Main processing function"""
        logging.info("Starting Stack Overflow data processing")
        
        # Extract Q&A pairs
        qa_pairs = self.get_question_with_answers()
        logging.info(f"Extracted {len(qa_pairs)} Q&A pairs")
        
        # Convert to training format
        self.convert_to_training_format(qa_pairs, output_file)
        
        logging.info("Processing complete")

def main():
    # Configure these values
    server = "YOUR_SERVER_NAME"  # e.g., "localhost" or "DESKTOP-ABC123"
    database = "StackOverflow2010"  # Or whatever you named the database
    output_file = "training_data/stackoverflow_training.jsonl"
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Initialize and run processor
    processor = StackOverflowProcessor(server, database)
    processor.process_database(output_file)

if __name__ == "__main__":
    main()
