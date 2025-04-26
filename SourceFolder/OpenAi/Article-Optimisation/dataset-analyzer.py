import json
import logging
from collections import defaultdict
from typing import Dict, List, Set
import re
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

class DatasetAnalyzer:
    def __init__(self):
        self.topic_keywords = {
            'mathematics': [
                'theorem', 'proof', 'equation', 'algebra', 'geometry',
                'calculus', 'probability', 'statistics', 'mathematical'
            ],
            'physics': [
                'force', 'energy', 'quantum', 'particle', 'mechanics',
                'relativity', 'electromagnetic', 'nuclear', 'physics'
            ],
            'computer_science': [
                'algorithm', 'programming', 'software', 'database', 'network',
                'computer', 'code', 'data structure', 'computation'
            ],
            'biology': [
                'cell', 'organism', 'gene', 'species', 'evolution',
                'protein', 'DNA', 'tissue', 'biological'
            ],
            'chemistry': [
                'molecule', 'reaction', 'compound', 'acid', 'element',
                'atomic', 'chemical', 'bond', 'solution'
            ],
            'history': [
                'century', 'war', 'empire', 'king', 'revolution',
                'ancient', 'medieval', 'historical', 'dynasty'
            ]
        }

    def analyze_dataset(self, file_path: str) -> Dict:
        """Perform detailed analysis of the dataset."""
        stats = {
            'total_articles': 0,
            'topic_distribution': defaultdict(int),
            'token_lengths': [],
            'subtopics': defaultdict(lambda: defaultdict(int)),
            'quality_scores': [],
            'cross_references': defaultdict(set)
        }

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Analyzing dataset"):
                try:
                    conversation = json.loads(line)

                    content = next((m['content'] for m in conversation['messages'] 
                                  if m['role'] == 'assistant'), '')

                    main_topic = self._identify_main_topic(content)
                    stats['topic_distribution'][main_topic] += 1

                    subtopics = self._identify_subtopics(content, main_topic)
                    for subtopic in subtopics:
                        stats['subtopics'][main_topic][subtopic] += 1

                    quality_score = self._calculate_quality_score(content)
                    stats['quality_scores'].append(quality_score)

                    stats['token_lengths'].append(len(content.split()))

                    refs = self._find_cross_references(content)
                    for ref_topic in refs:
                        if ref_topic != main_topic:
                            stats['cross_references'][main_topic].add(ref_topic)
                    
                    stats['total_articles'] += 1
                    
                except json.JSONDecodeError:
                    continue
        
        return stats

    def _identify_main_topic(self, content: str) -> str:
        """Identify the main topic of the content."""
        scores = {topic: sum(1 for keyword in keywords 
                           if keyword.lower() in content.lower())
                 for topic, keywords in self.topic_keywords.items()}
        return max(scores.items(), key=lambda x: x[1])[0] if any(scores.values()) else 'other'

    def _identify_subtopics(self, content: str, main_topic: str) -> Set[str]:
        """Identify subtopics within the main topic."""
        subtopics = set()
        content_lower = content.lower()

        patterns = [
            r'in ([a-zA-Z ]+) theory',
            r'field of ([a-zA-Z ]+)',
            r'branch of ([a-zA-Z ]+)',
            r'study of ([a-zA-Z ]+)'
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, content_lower)
            subtopics.update(match.group(1) for match in matches)
        
        return subtopics

    def _calculate_quality_score(self, content: str) -> float:
        """Calculate a quality score for the content."""
        score = 0
        

        words = len(content.split())
        length_score = min(words / 1000, 1)
        

        has_definition = any(phrase in content.lower() for phrase in ['is a', 'refers to', 'defined as'])
        has_examples = any(phrase in content.lower() for phrase in ['for example', 'such as', 'instance'])
        has_explanation = any(phrase in content.lower() for phrase in ['because', 'therefore', 'thus'])
        structure_score = (has_definition + has_examples + has_explanation) / 3
        

        technical_terms = sum(1 for topic, keywords in self.topic_keywords.items()
                            for keyword in keywords if keyword.lower() in content.lower())
        depth_score = min(technical_terms / 10, 1)  # Normalize to 10 terms
        
        score = (length_score + structure_score + depth_score) / 3
        return round(score, 2)

    def _find_cross_references(self, content: str) -> Set[str]:
        """Find references to other topics in the content."""
        references = set()
        content_lower = content.lower()
        
        for topic, keywords in self.topic_keywords.items():
            if any(keyword.lower() in content_lower for keyword in keywords):
                references.add(topic)
        
        return references

    def generate_recommendations(self, stats: Dict) -> List[str]:
        """Generate recommendations based on the analysis."""
        recommendations = []
        

        total = stats['total_articles']
        for topic, count in stats['topic_distribution'].items():
            percentage = (count / total) * 100
            if percentage < 10:
                recommendations.append(f"Consider adding more {topic} articles (currently {percentage:.1f}%)")
            elif percentage > 40:
                recommendations.append(f"Consider balancing dataset by reducing {topic} articles (currently {percentage:.1f}%)")

        avg_quality = sum(stats['quality_scores']) / len(stats['quality_scores'])
        if avg_quality < 0.6:
            recommendations.append("Consider improving content quality with more detailed explanations and examples")

        for topic, refs in stats['cross_references'].items():
            if len(refs) < 2:
                recommendations.append(f"Add more interdisciplinary content for {topic}")
        
        return recommendations

def main():
    analyzer = DatasetAnalyzer()
    

    dataset_path = "../Wikipedia_dumps/training_data/validated_training_data.jsonl"
    

    print("Analyzing dataset...")
    stats = analyzer.analyze_dataset(dataset_path)
    recommendations = analyzer.generate_recommendations(stats)

    print("\n=== Dataset Analysis Results ===")
    print(f"Total articles: {stats['total_articles']}")
    print("\nTopic Distribution:")
    for topic, count in stats['topic_distribution'].items():
        print(f"- {topic}: {count} articles ({(count/stats['total_articles'])*100:.1f}%)")
    
    print("\nQuality Metrics:")
    print(f"Average quality score: {sum(stats['quality_scores'])/len(stats['quality_scores']):.2f}")
    
    print("\nRecommendations:")
    for rec in recommendations:
        print(f"- {rec}")

if __name__ == "__main__":
    main()
