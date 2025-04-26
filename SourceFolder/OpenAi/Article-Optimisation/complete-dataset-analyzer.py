
import json
import logging
from collections import defaultdict
from typing import Dict, List, Set, Tuple
import re

import self
import stats
import timestamp
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import networkx as nx
from datetime import datetime
import pandas as pd
from wordcloud import WordCloud
import os

class DatasetAnalyzer:
    def __init__(self):
        self.topic_keywords = {
            'mathematics': [
                'theorem', 'proof', 'equation', 'algebra', 'geometry',
                'calculus', 'probability', 'statistics', 'mathematical',
                'matrix', 'vector', 'numerical', 'computation'
            ],
            'physics': [
                'force', 'energy', 'quantum', 'particle', 'mechanics',
                'relativity', 'electromagnetic', 'nuclear', 'physics',
                'velocity', 'acceleration', 'mass', 'momentum', 'gravity'
            ],
            'computer_science': [
                'algorithm', 'programming', 'software', 'database', 'network',
                'computer', 'code', 'data structure', 'computation',
                'operating system', 'memory', 'CPU', 'hardware', 'internet'
            ],
            'biology': [
                'cell', 'organism', 'gene', 'species', 'evolution',
                'protein', 'DNA', 'tissue', 'biological', 'enzyme',
                'metabolism', 'ecology', 'chromosome', 'bacteria', 'virus'
            ],
            'chemistry': [
                'molecule', 'reaction', 'compound', 'acid', 'element',
                'atomic', 'chemical', 'bond', 'solution', 'organic',
                'inorganic', 'polymer', 'catalyst', 'electron'
            ],
            'history': [
                'century', 'war', 'empire', 'king', 'revolution',
                'ancient', 'medieval', 'historical', 'dynasty',
                'civilization', 'period', 'era', 'modern', 'conquest'
            ]
        }
        
        self.complexity_indicators = {
            'basic': [
                'basic', 'simple', 'elementary', 'introduction', 'fundamental',
                'beginner', 'starting', 'overview', 'primer'
            ],
            'intermediate': [
                'advanced', 'complex', 'detailed', 'comprehensive',
                'thorough', 'in-depth', 'extensive', 'complete'
            ],
            'advanced': [
                'specialized', 'theoretical', 'advanced', 'research',
                'technical', 'expert', 'professional', 'academic'
            ]
        }
        
        self.interaction_patterns = {
            'explanation': [
                'explain', 'describe', 'define', 'elaborate',
                'clarify', 'illustrate', 'demonstrate'
            ],
            'analysis': [
                'analyze', 'compare', 'evaluate', 'examine',
                'investigate', 'study', 'assess', 'review'
            ],
            'application': [
                'apply', 'solve', 'implement', 'use',
                'practice', 'utilize', 'employ', 'execute'
            ]
        }

    def identify_main_topic(self, content: str) -> str:
        """Identify the main topic based on keyword frequency."""
        content_lower = content.lower()
        scores = defaultdict(int)
        
        for topic, keywords in self.topic_keywords.items():
            for keyword in keywords:
                scores[topic] += content_lower.count(keyword.lower())
        
        if not any(scores.values()):
            return 'other'
        
        return max(scores.items(), key=lambda x: x[1])[0]

    def identify_all_topics(self, content: str) -> Set[str]:
        """Identify all topics mentioned in the content."""
        content_lower = content.lower()
        topics = set()
        
        for topic, keywords in self.topic_keywords.items():
            if any(keyword.lower() in content_lower for keyword in keywords):
                topics.add(topic)
        
        return topics

    def count_keywords(self, content: str, stats: Dict):
        """Count frequency of all keywords in content."""
        content_lower = content.lower()
        for topic_keywords in self.topic_keywords.values():
            for keyword in topic_keywords:
                count = content_lower.count(keyword.lower())
                if count > 0:
                    stats['keyword_frequencies'][keyword] += count

    def analyze_dataset(self, file_path: str) -> Dict:
        """Enhanced analysis with additional metrics."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        stats = {
            'total_articles': 0,
            'topic_distribution': defaultdict(int),
            'token_lengths': [],
            'subtopics': defaultdict(lambda: defaultdict(int)),
            'quality_scores': [],
            'cross_references': defaultdict(set),
            'complexity_levels': defaultdict(int),
            'interaction_types': defaultdict(int),
            'temporal_distribution': defaultdict(int),
            'keyword_frequencies': defaultdict(int),
            'content_patterns': defaultdict(int),
            'topic_correlations': defaultdict(lambda: defaultdict(int)),
            'readability_scores': [],
            'technical_density': []
        }

        print(f"Processing file: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Analyzing dataset"):
                try:
                    conversation = json.loads(line)
                    content = next((m['content'] for m in conversation['messages'] 
                                  if m['role'] == 'assistant'), '')
                    
                    stats['total_articles'] += 1
                    main_topic = self.identify_main_topic(content)
                    stats['topic_distribution'][main_topic] += 1
                    self._analyze_complexity(content, stats)
                    self._analyze_interactions(content, stats)
                    self._analyze_technical_density(content, stats)
                    self._analyze_readability(content, stats)
                    self._analyze_content_patterns(content, stats)
                    topics = self.identify_all_topics(content)
                    for t1, t2 in self._get_topic_pairs(topics):
                        stats['topic_correlations'][t1][t2] += 1

                    self.count_keywords(content, stats)
                    
                except json.JSONDecodeError as e:
                    print(f"Error processing line: {e}")
                    continue
                except Exception as e:
                    print(f"Unexpected error: {e}")
                    continue

        return stats

    def _analyze_complexity(self, content: str, stats: Dict):
        """Analyze content complexity level."""
        content_lower = content.lower()
        for level, indicators in self.complexity_indicators.items():
            if any(indicator in content_lower for indicator in indicators):
                stats['complexity_levels'][level] += 1

    def _analyze_interactions(self, content: str, stats: Dict):
        """Analyze interaction patterns in content."""
        content_lower = content.lower()
        for pattern_type, patterns in self.interaction_patterns.items():
            if any(pattern in content_lower for pattern in patterns):
                stats['interaction_types'][pattern_type] += 1

    def _analyze_technical_density(self, content: str, stats: Dict):
        """Calculate technical term density."""
        words = content.lower().split()
        if not words:
            return
        
        technical_terms = sum(1 for word in words 
                            for keywords in self.topic_keywords.values()
                            if word in keywords)
        density = technical_terms / len(words)
        stats['technical_density'].append(density)

    def _analyze_readability(self, content: str, stats: Dict):
        """Calculate readability score."""
        sentences = [s for s in content.split('.') if s.strip()]
        if not sentences:
            return
            
        words = content.split()
        avg_sentence_length = len(words) / len(sentences)
        stats['readability_scores'].append(avg_sentence_length)

    def _analyze_content_patterns(self, content: str, stats: Dict):
        """Analyze common content patterns."""
        patterns = {
            'definition': r'(?:is|are) (?:defined as|a type of|a kind of)',
            'example': r'(?:for example|such as|instance)',
            'comparison': r'(?:compared to|similar to|differs from)',
            'cause_effect': r'(?:because|therefore|as a result)',
        }
        
        for pattern_type, pattern in patterns.items():
            if re.search(pattern, content, re.IGNORECASE):
                stats['content_patterns'][pattern_type] += 1

    def _get_topic_pairs(self, topics: Set[str]) -> List[Tuple[str, str]]:
        """Get all possible topic pairs for correlation analysis."""
        return [(t1, t2) for t1 in topics for t2 in topics if t1 < t2]

    def _create_correlation_matrix(self, correlations: Dict) -> pd.DataFrame:
        """Create correlation matrix from topic correlations."""
        topics = sorted(set(correlations.keys()) | 
                      set(t for d in correlations.values() for t in d.keys()))
        matrix = pd.DataFrame(0, index=topics, columns=topics)
        
        for t1, t2_dict in correlations.items():
            for t2, count in t2_dict.items():
                matrix.loc[t1, t2] = count
                matrix.loc[t2, t1] = count
        
        return matrix

    def visualize_results(self, stats: Dict, output_dir: str):
        """Generate comprehensive visualizations of the analysis results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        os.makedirs(output_dir, exist_ok=True)

        plt.style.use('seaborn')

        plt.figure(figsize=(12, 8))
        topic_data = stats['topic_distribution']
        topics = list(topic_data.keys())
        values = list(topic_data.values())

        total = sum(values)
        labels = [f'{t}\n({v/total*100:.1f}%)' if v/total > 0.02 else '' 
                 for t, v in zip(topics, values)]
        
        plt.pie(values, labels=labels, autopct=lambda pct: f'{pct:.1f}%' if pct > 2 else '')
        plt.title('Topic Distribution in Dataset')
        plt.savefig(os.path.join(output_dir, f'topic_distribution_pie_{timestamp}.png'))
        plt.close()

        plt.figure(figsize=(15, 8))
        colors = plt.cm.viridis(np.linspace(0, 1, len(topics)))
        bars = plt.bar(topics, values, color=colors)
        plt.xticks(rotation=45, ha='right')
        plt.title('Topic Distribution (Bar Chart)')
        plt.xlabel('Topics')
        plt.ylabel('Number of Articles')

        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'topic_distribution_bar_{timestamp}.png'))
        plt.close()

        if stats['technical_density']:
            plt.figure(figsize=(12, 6))
            sns.histplot(data=stats['technical_density'], bins=30)
            plt.title('Distribution of Technical Term Density')
            plt.xlabel('Technical Term Density')
            plt.ylabel('Count')
            plt.savefig(os.path.join(output_dir, f'technical_density_{timestamp}.png'))
            plt.close()

        if stats['complexity_levels']:
            plt.figure(figsize=(10, 6))
            complexity_data = pd.DataFrame.from_dict(
                stats['complexity_levels'], 
                orient='index', 
                columns=['Count']
            )
            complexity_data.plot(kind='bar')
            plt.title('Content Complexity Distribution')
            plt.xlabel('Complexity Level')
            plt.ylabel('Count')

            for i, v in enumerate(complexity_data['Count']):
                plt.text(i, v, str(v), ha='center', va='bottom')
                
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'complexity_levels_{timestamp}.png'))
            plt.close()
        
        # 5. Interaction Types Distribution
        if stats['interaction_types']:
            plt.figure(figsize=(10, 6))
            interaction_data = pd.DataFrame.from_dict(
                stats['interaction_types'], 
                orient='index', 
                columns=['Count']
            )
            interaction_data.plot(kind='bar')
            plt.title('Interaction Types Distribution')
            plt.xlabel('Interaction Type')
            plt.ylabel('Count')
            
            # Add value labels
            for i, v in enumerate(interaction_data['Count']):
                plt.text(i, v, str(v), ha='center', va='bottom')
                
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'interaction_types_{timestamp}.png'))
            plt.close()
        
        # 6. Content Patterns Distribution
        if stats['content_patterns']:
            plt.figure(figsize=(12, 6))
            patterns_data = pd.DataFrame.from_dict(
                stats['content_patterns'], 
                orient='index', 
                columns=['Count']
            )
            patterns_data.plot(kind='bar')
            plt.title('Content Patterns Distribution')
            plt.xlabel('Pattern Type')
            plt.ylabel('Count')
            
            # Add value labels
            for i, v in enumerate(patterns_data['Count']):
                plt.text(i, v, str(v), ha='center', va='bottom')
                
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'content_patterns_{timestamp}.png'))
            plt.close()
        
        # 7. Topic Correlation Heatmap
        if stats['topic_correlations']:
            correlation_matrix = self._create_correlation_matrix(stats['topic_correlations'])
            plt.figure(figsize=(12, 10))
            sns.heatmap(correlation_matrix, annot=True, fmt='d', cmap='YlOrRd')
            plt.title('Topic Correlations')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'topic_correlations_{timestamp}.png'))
            plt.close()
        
        # 8. Keyword Frequencies WordCloud
        if stats['keyword_frequencies']:
            wordcloud = WordCloud(
                width=1600, 
                height=800,
                background_color='white',
                min_font_size=10
            ).generate_from_frequencies(stats['keyword_frequencies'])
            
            plt.figure(figsize=(20, 10))


            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title('Keyword Frequencies Word Cloud')
            plt.savefig(os.path.join(output_dir, f'keyword_wordcloud_{timestamp}.png'))
            plt.close()

        #Generate summary report
        self._generate_summary_report(stats, output_dir, timestamp)


    def _generate_summary_report(self, stats: Dict, output_dir: str, timestamp: str):
        """Generate a detailed summary report of the analysis."""
        report_path = os.path.join(output_dir, f'analysis_report_{timestamp}.txt')

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=== Dataset Analysis Report ===\n\n")

            #Basic Statistics
            f.write("Basic Statistics:\n")
            f.write(f"Total Articles: {stats['total_articles']}\n\n")

            #Topic Distribution
            f.write("Topic Distribution:\n")
            total_articles = stats['total_articles']
            for topic, count in sorted(stats['topic_distribution'].items(),
                                       key=lambda x: x[1], reverse=True):
                percentage = (count / total_articles) * 100
                f.write(f"{topic}: {count} articles ({percentage:.1f}%)\n")
            f.write("\n")

            #Technical Density
            if stats['technical_density']:
                f.write("Technical Density Statistics:\n")
                td_array = np.array(stats['technical_density'])
                f.write(f"Mean: {np.mean(td_array):.3f}\n")
                f.write(f"Median: {np.median(td_array):.3f}\n")
                f.write(f"Std Dev: {np.std(td_array):.3f}\n")
                f.write(f"Min: {np.min(td_array):.3f}\n")
                f.write(f"Max: {np.max(td_array):.3f}\n")
                f.write("\n")

            #Complexity Levels
            if stats['complexity_levels']:
                f.write("Content Complexity Distribution:\n")
                total_complexity = sum(stats['complexity_levels'].values())
                for level, count in stats['complexity_levels'].items():
                    percentage = (count / total_complexity) * 100 if total_complexity > 0 else 0
                    f.write(f"{level}: {count} ({percentage:.1f}%)\n")
                f.write("\n")

            #Interaction Types
            if stats['interaction_types']:
                f.write("Interaction Types Distribution:\n")
                total_interactions = sum(stats['interaction_types'].values())
                for itype, count in stats['interaction_types'].items():
                    percentage = (count / total_interactions) * 100 if total_interactions > 0 else 0
                    f.write(f"{itype}: {count} ({percentage:.1f}%)\n")
                f.write("\n")

            #Content Patterns
            if stats['content_patterns']:
                f.write("Content Patterns Distribution:\n")
                total_patterns = sum(stats['content_patterns'].values())
                for pattern, count in stats['content_patterns'].items():
                    percentage = (count / total_patterns) * 100 if total_patterns > 0 else 0
                    f.write(f"{pattern}: {count} ({percentage:.1f}%)\n")
                f.write("\n")

            if stats['keyword_frequencies']:
                f.write("Top 20 Keywords:\n")
                sorted_keywords = sorted(
                    stats['keyword_frequencies'].items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:20]
                max_freq = sorted_keywords[0][1] if sorted_keywords else 0
                for keyword, freq in sorted_keywords:
                    percentage = (freq / max_freq) * 100
                    f.write(f"{keyword}: {freq} ({percentage:.1f}% of max frequency)\n")
                f.write("\n")

            #Cross-References Analysis
            if stats['topic_correlations']:
                f.write("Topic Cross-References:\n")
                for topic1, refs in stats['topic_correlations'].items():
                    if refs:
                        f.write(f"\n{topic1} references:\n")
                        for topic2, count in sorted(refs.items(), key=lambda x: x[1], reverse=True):
                            f.write(f"  - {topic2}: {count} references\n")
                f.write("\n")

            #Recommendations
            f.write("\nRecommendations:\n")

            #Topic balance recommendations
            for topic, count in stats['topic_distribution'].items():
                percentage = (count / total_articles) * 100
                if percentage < 10 and topic != 'other':
                    f.write(f"- Consider adding more {topic} content (currently {percentage:.1f}%)\n")
                elif percentage > 30 and topic != 'other':
                    f.write(f"- Consider balancing {topic} content (currently {percentage:.1f}%)\n")

            #Complexity balance recommendations
            if stats['complexity_levels']:
                complexity_total = sum(stats['complexity_levels'].values())
                for level, count in stats['complexity_levels'].items():
                    percentage = (count / complexity_total) * 100 if complexity_total > 0 else 0
                    if percentage < 20:
                        f.write(f"- Consider adding more {level} complexity content\n")
                    elif percentage > 50:
                        f.write(f"- Consider balancing {level} complexity content\n")


    def main():
        analyzer = DatasetAnalyzer()
        script_dir = os.path.dirname(os.path.abspath(__file__))
        dataset_path = os.path.join(script_dir, "training_data", "validated_training_data.jsonl")
        output_dir = os.path.join(script_dir, "analysis_results")
        os.makedirs(output_dir, exist_ok=True)
        print("Analyzing dataset...")
        try:
            stats = analyzer.analyze_dataset(dataset_path)

            print("\nAnalysis Summary:")
            print(f"Total articles processed: {stats['total_articles']}")
            print("\nTopic Distribution:")
            for topic, count in sorted(stats['topic_distribution'].items(),
                                       key=lambda x: x[1], reverse=True):
                print(f"- {topic}: {count} articles")
            print("\nGenerating visualizations and report...")
            analyzer.visualize_results(stats, output_dir)

            print(f"\nAnalysis complete. Results saved in {output_dir}")
        except Exception as e:
            print(f"Error during analysis: {e}")
            raise

    if __name__ == "__main__":
        main()
