# AI Mentor Project Development Guide

## Project Overview
AI mentor system that processes Wikipedia data for specialized subject training, with a focus on creating separate models for different academic subjects (mathematics, physics, biology, chemistry, computer science, and history).

## Step 1: Data Extraction and Processing

### Initial Data Preparation
1. Use Wiki-extractor to process Wikipedia dump files
2. Convert extracted data to proper format for fine-tuning
3. Implement subject-specific filtering

### Data Format Requirements
- Input format: JSONL
- Structure:
```json
{
    "messages": [
        {"role": "system", "content": "You are a knowledgeable AI mentor helping users learn about various topics."},
        {"role": "user", "content": "What is [Topic]?"},
        {"role": "assistant", "content": "Detailed explanation..."}
    ]
}
```

### Subject Distribution Found
Initial distribution across subjects:
- Mathematics: 23,898 examples
- History: 11,814 examples
- Biology: 605 examples
- Chemistry: 655 examples
- Computer Science: 454 examples
- Physics: 424 examples

## Step 2: Subject-Specific Processing

### Biology Content Filter Implementation
Create a sophisticated filtering system with:

1. Subject Patterns:
```python
biology_patterns = {
    'molecular_biology': {
        'primary': {'dna', 'rna', 'protein', 'gene', 'genome', 'chromosome'},
        'secondary': {'transcription', 'translation', 'enzyme', 'mutation'}
    },
    'cell_biology': {
        'primary': {'cell', 'membrane', 'nucleus', 'mitochondria', 'organelle'},
        'secondary': {'cytoplasm', 'ribosome', 'golgi', 'endoplasmic'}
    },
    # [Add other biology subfields]
}
```

2. Validation Criteria:
- Minimum content length: 20 characters
- Maximum content length: 15,000 characters
- Required keywords from primary/secondary lists
- Subfield coverage requirements
- Confidence scoring system

### Quality Control Steps
1. Data Validation:
- Check for subject relevance
- Verify content quality
- Ensure proper formatting
- Remove non-subject content

2. Content Balance:
- Aim for 3,000+ examples per subject
- Maintain quality while increasing quantity
- Even distribution across subfields

## Step 3: Data Preparation for Training

### Processing Pipeline
1. Extract Wikipedia content
2. Filter by subject
3. Format for fine-tuning
4. Validate content
5. Balance dataset
6. Generate training files

### Validation Requirements
Create a validation system that checks:
1. Content relevance to subject
2. Proper formatting
3. Content quality
4. Topic coverage
5. Data distribution

## Step 4: Model Training Setup

### Model Requirements
- Base Model: GPT-4
- Training Data Format: JSONL
- Token Limits: Consider model-specific limitations
- Subject-Specific Models: Separate models for each subject

### Training Parameters
```python
training_config = {
    'model_type': 'gpt-4',
    'training_format': 'jsonl',
    'max_tokens': 2048,
    'temperature': 0.7,
    'learning_rate': 1e-5,
    'batch_size': 4,
    'epochs': 3
}
```

## Step 5: Quality Assurance

### Validation Metrics
Track and verify:
1. Subject accuracy
2. Content relevance
3. Response quality
4. Knowledge coverage
5. Educational value

### Testing Requirements
Implement testing for:
1. Subject matter accuracy
2. Response coherence
3. Educational effectiveness
4. Content coverage
5. Interactive capabilities

## Implementation Notes

### Code Structure
Maintain separate modules for:
1. Data extraction
2. Subject filtering
3. Content validation
4. Training preparation
5. Quality assurance

### Best Practices
1. Regular validation of processed data
2. Continuous quality monitoring
3. Iterative improvement of filters
4. Balanced dataset maintenance
5. Comprehensive testing

## Success Criteria
1. Achieve minimum 3,000 quality examples per subject
2. Maintain high subject matter accuracy
3. Ensure comprehensive topic coverage
4. Validate educational effectiveness
5. Verify interactive capability

## Special Considerations
1. Handle subject overlap appropriately
2. Maintain educational focus
3. Ensure content accuracy
4. Balance quantity vs. quality
5. Regular quality validation

When implementing this project, follow the steps sequentially and maintain rigorous quality control throughout the process. Adjust thresholds and criteria based on validation results to achieve optimal performance for each subject area.
