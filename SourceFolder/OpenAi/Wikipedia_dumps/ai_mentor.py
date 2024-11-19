import openai

# Replace 'your-api-key' with your actual OpenAI API key
openai.api_key = 'your-api-key'

def ask_ai(question):
    response = openai.Completion.create(
        engine="text-davinci-003",  # Choose the appropriate model
        prompt=question,
        max_tokens=200,  # Limits the length of the response
        temperature=0.7  # Adjusts the creativity of the response
    )
    return response.choices[0].text.strip()

if __name__ == "__main__":
    while True:
        user_input = input("Ask a study-related question (or type 'quit' to exit): ")
        if user_input.lower() == 'quit':
            break
        answer = ask_ai(user_input)
        print(f"AI Mentor: {answer}")
