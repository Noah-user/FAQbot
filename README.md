Overview:

The FAQ Bot is an intelligent question-answering system designed to respond to user queries. It leverages a combination of TF-IDF and cosine similarity for retrieving answers from a predefined FAQ dataset and OpenAI GPT for generating fallback answers when no relevant FAQ is found.

Features:

FAQ Matching: Uses TF-IDF vectorization and cosine similarity to find the most relevant question in the FAQ dataset.
Dynamic Answers: Employs OpenAI GPT to generate fallback answers for unmatched questions.
Interactive Interface: Provides a simple conversational experience for users.
Customizable: Easily update the FAQ dataset and adjust thresholds to fine-tune performance.

Prerequisites:

Required Libraries

Install the following Python libraries: pandas, scikit-learn, openai

You can install these dependencies with: pip install pandas scikit-learn openai

Files and Structure:
faq_dataset.csv: The FAQ dataset containing question and answer columns. Ensure it is structured as follows:
question,answer
What is your refund policy?,We offer a 30-day refund policy for unused products.
How do I reset my password?,You can reset your password by clicking 'Forgot Password' on the login page.
...
Script File: The Python script implementing the FAQ Bot logic.

How It Works:

Load FAQ Dataset: The bot reads a CSV file containing FAQ questions and answers.
Preprocess Questions: Uses TF-IDF Vectorizer to transform FAQ questions into numerical vectors.
Match Questions:
Computes cosine similarity between the user's query and FAQ questions to find the best match.
A similarity threshold (default: 0.3) determines if a match is considered relevant.
Fallback with GPT: If no FAQ question meets the threshold, the bot uses OpenAI GPT to dynamically generate an answer.
Interactive Mode: Users can ask questions in a conversational loop and receive responses until they type "exit."

Setup and Configuration:

Prepare the FAQ Dataset:
Create a faq_dataset.csv file with question and answer columns.

Set OpenAI API Key:
Replace "your_openai_api_key" in the script with your OpenAI API key.
Sign up for OpenAI API access if you don’t have a key: OpenAI API

Run the Script: Execute the script to start the FAQ Bot: python faq_bot.py

Limitations: 
Requires an active internet connection for GPT-based fallback answers.
Performance depends on the quality of the FAQ dataset and the OpenAI model.

License: This project is open-source and can be modified and distributed freely.
