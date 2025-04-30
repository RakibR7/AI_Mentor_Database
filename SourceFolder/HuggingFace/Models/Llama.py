import json
import os
from transformers import AutoTokenizer, AutoModelForCausalLM

api_key = "hugging-face-api"

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B", token=api_key)
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B", token=api_key)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

NOTES_FILE = '../Output/study_notes.json'

def load_notes():
    if os.path.exists(NOTES_FILE):
        with open(NOTES_FILE, 'r') as file:
            return json.load(file)
    return {}

def save_notes(notes):
    with open(NOTES_FILE, 'w') as file:
        json.dump(notes, file, indent=4)

def generate_summary(input_text):
    prompt = f"Summarize the following text into a topic, subheading, and description:\n\n{input_text}\n\nTopic:"
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)

    outputs = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_length=00,
        pad_token_id=tokenizer.eos_token_id
    )

    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)

    try:
        topic_part = summary.split("Topic:")[-1].strip()
        subheading_part = topic_part.split("Subheading:")[0].strip()
        subheading = topic_part.split("Subheading:")[-1].split("Description:")[0].strip()
        description = topic_part.split("Description:")[-1].strip()
        return subheading_part, subheading, description
    except IndexError:
        print("Failed to extract topic, subheading, and description. Please refine the input text or try again.")
        return None, None, None

def add_note():
    notes = load_notes()

    input_text = input("Enter the text to summarize: ").strip()
    topic, subheading, description = generate_summary(input_text)

    if topic and subheading and description:
        if topic not in notes:
            notes[topic] = {}

        notes[topic][subheading] = description
        save_notes(notes)
        print(f"Note added under topic '{topic}' with subheading '{subheading}'.")
    else:
        print("Could not generate a valid summary. Please try with a different input.")

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

def ask_ai():
    notes = load_notes()
    if not notes:
        print("No notes available. Please add notes first.")
        return

    notes_context = ""
    for topic, subheadings in notes.items():
        notes_context += f"Topic: {topic}\n"
        for subheading, description in subheadings.items():
            notes_context += f"  Subheading: {subheading}\n    Description: {description}\n"

    question = input("\nEnter your question: ").strip()
    prompt = f"Here are some study notes:\n{notes_context}\n\nQuestion: {question}\nAnswer:"

    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)

    outputs = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_length=600,
        pad_token_id=tokenizer.eos_token_id
    )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"AI Tutor: {answer.split('Answer:')[-1].strip()}")


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
