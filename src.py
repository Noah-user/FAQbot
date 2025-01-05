import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI

# Load FAQ dataset
faq_data = pd.read_csv("faq_dataset.csv")

# Vectorize FAQ questions
vectorizer = TfidfVectorizer()
faq_vectors = vectorizer.fit_transform(faq_data["question"])

# OpenAI API key (replace with your key)
client = OpenAI(
    api_key="your_openai_api_key"
)

def find_relevant_question(user_input):
    """Find the most relevant FAQ question using TF-IDF and cosine similarity."""
    user_input_vector = vectorizer.transform([user_input])
    similarities = cosine_similarity(user_input_vector, faq_vectors)
    max_similarity_index = similarities.argmax()
    max_similarity_score = similarities[0, max_similarity_index]
    
    # Set a similarity threshold for matching
    if max_similarity_score > 0.3:
        return faq_data.iloc[max_similarity_index]["question"]
    return None

def generate_fallback_answer(user_input):
    """Generate a dynamic answer using OpenAI GPT with the updated interface."""
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "user", "content": user_input}
            ],
            model="gpt-4",  # Use "gpt-4" or "gpt-3.5-turbo"
        )
        # Access the content of the first choice
        return chat_completion.choices[0].message.content.strip()
    except Exception as e:
        return f"I'm sorry, I couldn't generate an answer. Error: {e}"

def get_answer(user_input):
    """Retrieve an answer from the FAQ or generate one dynamically."""
    best_question = find_relevant_question(user_input)
    if best_question:
        return faq_data[faq_data["question"] == best_question]["answer"].values[0]
    else:
        # Generate fallback answer
        return generate_fallback_answer(user_input)

# Start FAQ Bot
print("FAQ Bot: Hi! Ask me anything. Type 'exit' to quit.")
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("FAQ Bot: Goodbye!")
        break
    response = get_answer(user_input)
    print(f"FAQ Bot: {response}")
