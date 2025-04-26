class DatasetSizeCalculator:
    """Calculate and validate dataset sizes for different learning depths."""
    
    def __init__(self):
        # Minimum articles needed for different coverage levels
        self.coverage_levels = {
            "basic": {
                "min_articles": 500,
                "recommended": 1000,
                "description": "Core concepts and fundamental principles",
                "estimated_topics": 50,
                "articles_per_topic": 20
            },
            "intermediate": {
                "min_articles": 2000,
                "recommended": 3000,
                "description": "Detailed understanding with practical applications",
                "estimated_topics": 100,
                "articles_per_topic": 30
            },
            "advanced": {
                "min_articles": 5000,
                "recommended": 8000,
                "description": "Comprehensive coverage including specialized topics",
                "estimated_topics": 200,
                "articles_per_topic": 40
            }
        }
        
        # Subject-specific adjustments (multipliers)
        self.subject_adjustments = {
            "mathematics": {
                "multiplier": 1.2,
                "critical_topics": [
                    "Algebra", "Calculus", "Geometry", "Statistics",
                    "Number Theory", "Linear Algebra", "Mathematical Analysis"
                ]
            },
            "physics": {
                "multiplier": 1.2,
                "critical_topics": [
                    "Classical Mechanics", "Quantum Mechanics", "Thermodynamics",
                    "Electromagnetism", "Relativity", "Nuclear Physics"
                ]
            },
            "computer_science": {
                "multiplier": 1.3,
                "critical_topics": [
                    "Programming", "Data Structures", "Algorithms",
                    "Computer Architecture", "Operating Systems", "Databases",
                    "Networking", "Software Engineering"
                ]
            },
            "biology": {
                "multiplier": 1.1,
                "critical_topics": [
                    "Cell Biology", "Genetics", "Evolution", "Physiology",
                    "Ecology", "Molecular Biology", "Biochemistry"
                ]
            },
            "chemistry": {
                "multiplier": 1.1,
                "critical_topics": [
                    "Organic Chemistry", "Inorganic Chemistry", "Physical Chemistry",
                    "Analytical Chemistry", "Biochemistry", "Chemical Bonding"
                ]
            },
            "history": {
                "multiplier": 1.4,
                "critical_topics": [
                    "Ancient History", "Medieval History", "Modern History",
                    "World Wars", "Political History", "Social History",
                    "Economic History", "Cultural History"
                ]
            }
        }
    
    def get_recommended_size(self, subject: str, level: str = "intermediate") -> dict:
        """Calculate recommended dataset size for a subject and coverage level."""
        if subject not in self.subject_adjustments or level not in self.coverage_levels:
            return None
            
        base_coverage = self.coverage_levels[level]
        subject_adj = self.subject_adjustments[subject]
        
        recommended_articles = int(base_coverage["recommended"] * subject_adj["multiplier"])
        min_articles = int(base_coverage["min_articles"] * subject_adj["multiplier"])
        
        return {
            "subject": subject,
            "level": level,
            "minimum_articles": min_articles,
            "recommended_articles": recommended_articles,
            "critical_topics": subject_adj["critical_topics"],
            "description": base_coverage["description"],
            "estimated_topics": base_coverage["estimated_topics"],
            "articles_per_topic": base_coverage["articles_per_topic"]
        }

# Example usage and recommendations
calculator = DatasetSizeCalculator()

# Print recommendations for each subject
for subject in calculator.subject_adjustments.keys():
    for level in ["basic", "intermediate", "advanced"]:
        rec = calculator.get_recommended_size(subject, level)
        print(f"\n{subject.upper()} - {level.title()} Level:")
        print(f"Minimum articles: {rec['minimum_articles']}")
        print(f"Recommended articles: {rec['recommended_articles']}")
        print(f"Coverage: {rec['description']}")
        print("Critical topics to cover:")
        for topic in rec['critical_topics']:
            print(f"  - {topic}")
