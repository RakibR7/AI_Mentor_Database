import json
import tiktoken
from tqdm import tqdm
import logging
from typing import List, Dict

logging.basicConfig(level=logging.INFO)

class EducationalContentGenerator:
    def __init__(self):
        self.encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

        self.teaching_patterns = {
            "explanation": [
                "Let me explain {topic} step by step...",
                "To understand {topic}, let's break it down...",
                "The key concepts of {topic} are..."
            ],
            "problem_solving": [
                "Let's solve this {topic} problem together...",
                "Here's how we can approach this {topic} question...",
                "Let's tackle this {topic} challenge step by step..."
            ],
            "interactive": [
                "Do you understand how {topic} works so far?",
                "Can you try to explain {topic} in your own words?",
                "What questions do you have about {topic}?"
            ],
            "encouragement": [
                "That's a great question about {topic}!",
                "You're making good progress understanding {topic}.",
                "Don't worry if {topic} seems difficult at first..."
            ]
        }

    def create_educational_conversation(self, article: Dict) -> List[Dict]:
        """Convert an article into educational conversations."""
        title = article.get('title', '')
        content = article.get('text', '').strip()

        conversations = []

        conversations.append({
            "messages": [
                {
                    "role": "system",
                    "content": "You are a knowledgeable and patient AI mentor helping users learn about various topics."
                },
                {
                    "role": "user",
                    "content": f"Can you explain {title} in simple terms?"
                },
                {
                    "role": "assistant",
                    "content": self._format_explanation(content)
                }
            ]
        })

        conversations.append({
            "messages": [
                {
                    "role": "system",
                    "content": "You are a knowledgeable and patient AI mentor helping users learn about various topics."
                },
                {
                    "role": "user",
                    "content": f"I'm having trouble understanding {title}. Can you break it down for me?"
                },
                {
                    "role": "assistant",
                    "content": self._format_step_by_step(content)
                },
                {
                    "role": "user",
                    "content": "Could you clarify that further?"
                },
                {
                    "role": "assistant",
                    "content": self._format_clarification(content)
                }
            ]
        })
        
        return conversations

    def _format_explanation(self, content: str) -> str:
        """Format content as a clear explanation."""
        paragraphs = content.split('\n\n')
        introduction = paragraphs[0] if paragraphs else content[:300]
        
        return f"Let me explain this in simple terms. {introduction}\n\n" + \
               "The key points to understand are:\n" + \
               self._extract_key_points(content)

    def _format_step_by_step(self, content: str) -> str:
        """Format content as step-by-step learning."""
        steps = []
        paragraphs = content.split('\n\n')
        
        for i, para in enumerate(paragraphs[:5], 1):
            steps.append(f"Step {i}: {para.strip()[:200]}")
            
        return "Let's break this down into manageable steps:\n\n" + \
               "\n\n".join(steps)

    def _format_clarification(self, content: str) -> str:
        """Format content as a helpful clarification."""
        return "Let me explain this another way:\n\n" + \
               self._extract_key_points(content) + \
               "\n\nDoes this help clarify things?"

    def _extract_key_points(self, content: str) -> str:
        """Extract and format key points from content."""
        sentences = [s.strip() for s in content.split('.') if len(s.strip()) > 20]
        key_points = []
        
        for i, sentence in enumerate(sentences[:5], 1):
            key_points.append(f"{i}. {sentence}.")
            
        return "\n".join(key_points)

    def process_dataset(self, input_file: str, output_file: str) -> None:
        """Process dataset into educational conversations."""
        educational_conversations = []
        
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                for line in tqdm(f):
                    article = json.loads(line)
                    conversations = self.create_educational_conversation(article)
                    educational_conversations.extend(conversations)

            with open(output_file, 'w', encoding='utf-8') as f:
                for conv in educational_conversations:
                    f.write(json.dumps(conv) + '\n')
                    
            logging.info(f"Processed {len(educational_conversations)} educational conversations")
            
        except Exception as e:
            logging.error(f"Error processing dataset: {e}")

def main():
    input_file = "training_data/wiki_training_data.jsonl"
    output_file = "training_data/mentor_training_data.jsonl"
    
    processor = EducationalContentGenerator()
    processor.process_dataset(input_file, output_file)

if __name__ == "__main__":
    main()
