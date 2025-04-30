import json
import os
from transformers import AutoTokenizer, AutoModelForCausalLM

# Your Hugging Face API key
api_key = "huggingface-api-key"

# Load the tokenizer and model with the API key
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B", token=api_key)
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B", token=api_key)

# Set the pad token to eos token if not already set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Define the file name where notes will be stored
NOTES_FILE = '../Output/study_notes.json'

# Load existing notes from the JSON file if it exists
def load_notes():
    if os.path.exists(NOTES_FILE):
        with open(NOTES_FILE, 'r') as file:
            return json.load(file)
    return {}

# Save notes to the JSON file
def save_notes(notes):
    with open(NOTES_FILE, 'w') as file:
        json.dump(notes, file, indent=4)

# Use the AI to generate topic, subheading, and description
def generate_summary(input_text):
    prompt = f"Summarize the following text into a topic, subheading, and description:\n\n{input_text}\n\nTopic:"
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    
    # Generate an answer using the model
    outputs = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_length=00,
        pad_token_id=tokenizer.eos_token_id
    )

    # Decode the generated text
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract topic, subheading, and description from the AI response
    try:
        topic_part = summary.split("Topic:")[-1].strip()
        subheading_part = topic_part.split("Subheading:")[0].strip()
        subheading = topic_part.split("Subheading:")[-1].split("Description:")[0].strip()
        description = topic_part.split("Description:")[-1].strip()
        return subheading_part, subheading, description
    except IndexError:
        print("Failed to extract topic, subheading, and description. Please refine the input text or try again.")
        return None, None, None

# Add or update notes using AI-generated summary
def add_note():
    notes = load_notes()

    # Get input text from the user
    input_text = input("Enter the text to summarize: ").strip()
    
    # Use AI to generate topic, subheading, and description
    topic, subheading, description = generate_summary(input_text)
    
    # Check if the AI generated valid outputs
    if topic and subheading and description:
        # Organize notes into JSON structure
        if topic not in notes:
            notes[topic] = {}  # Create a new topic if it doesn't exist
        
        # Add or update the subheading with the description
        notes[topic][subheading] = description

        # Save updated notes
        save_notes(notes)
        print(f"Note added under topic '{topic}' with subheading '{subheading}'.")
    else:
        print("Could not generate a valid summary. Please try with a different input.")

# Display notes in an organized format
def display_notes():
    notes = load_notes()

    if not notes:
        print("No notes available.")
        return
    
    print("\nCurrent Study Notes:")
    for topic, subheadings in notes.items():
        print(f"Topic: {topic}")
        for subheading, description in subheadings.items():
            print(f"  Subheading: {subheading}")
            print(f"    Description: {description}")
        print()

# Ask the AI using the notes as context
def ask_ai():
    notes = load_notes()
    if not notes:
        print("No notes available. Please add notes first.")
        return

    # Prepare the notes into a readable context for the AI
    notes_context = ""
    for topic, subheadings in notes.items():
        notes_context += f"Topic: {topic}\n"
        for subheading, description in subheadings.items():
            notes_context += f"  Subheading: {subheading}\n    Description: {description}\n"

    # Get the user's question
    question = input("\nEnter your question: ").strip()
    prompt = f"Here are some study notes:\n{notes_context}\n\nQuestion: {question}\nAnswer:"

    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    
    # Generate an answer using the model
    outputs = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_length=600,
        pad_token_id=tokenizer.eos_token_id
    )

    # Decode the generated text
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract and return only the answer part
    print(f"AI Tutor: {answer.split('Answer:')[-1].strip()}")

# Main program loop
def main():
    while True:
        print("\n1. Add a note")
        print("2. Display notes")
        print("3. Ask the AI")
        print("4. Exit")
        choice = input("Choose an option (1, 2, 3, 4): ")

        if choice == '1':
            add_note()
        elif choice == '2':
            display_notes()
        elif choice == '3':
            ask_ai()
        elif choice == '4':
            break
        else:
            print("Invalid choice. Please select 1, 2, 3, or 4.")

if __name__ == "__main__":
    main()
