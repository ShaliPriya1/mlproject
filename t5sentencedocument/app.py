from flask import Flask, render_template, request, jsonify
import json
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Initialize Flask app
app = Flask(__name__)

# Load the T5 model and tokenizer for answering questions
t5_model = T5ForConditionalGeneration.from_pretrained('t5-small')
t5_tokenizer = T5Tokenizer.from_pretrained('t5-small')

# Load the Sentence-BERT model for folder name embeddings
sentence_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Load preprocessed embeddings and folder information
with open('embeddings_t5.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
    folder_name_embeddings = data['folder_name_embeddings']
    documents = data['documents']
    filenames = data['filenames']

# Function to find the best matching folder based on cosine similarity
def get_best_folder(query):
    # Check if the query is empty
    if not query.strip():
        return "Please provide a valid query."  # Handle empty query case

    # Get the embedding for the user's query using Sentence-BERT
    query_embedding = sentence_model.encode([query])[0]

    # Debug: Check the shape of query_embedding
    print(f"Query embedding shape: {query_embedding.shape if hasattr(query_embedding, 'shape') else 'No shape attribute'}")

    folder_similarities = {}

    # Calculate cosine similarity with each folder's name embedding
    for folder, folder_embedding in folder_name_embeddings.items():
        try:
            # Check the shape of folder_embedding
            print(f"Folder: {folder}, Folder embedding shape: {np.array(folder_embedding).shape}")
            
            # Calculate cosine similarity between query and folder embedding
            cosine_sim = cosine_similarity([query_embedding], [folder_embedding])[0][0]
            folder_similarities[folder] = cosine_sim
        except Exception as e:
            print(f"Error processing folder '{folder}': {e}")

    # Debug: Print folder similarities to help debug
    print(f"Folder similarities: {folder_similarities}")

    # Find the folder with the highest cosine similarity
    if folder_similarities:
        best_folder = max(folder_similarities, key=folder_similarities.get)
        print(f"Best matching folder: {best_folder} with similarity: {folder_similarities[best_folder]}")
        return best_folder
    else:
        return "No matching folder found."

# Function to get specific information from a folder's documents
def get_document_info(folder, query):
    folder_docs = documents[folder]
    document = "\n".join(folder_docs)  # Combine all documents into one context for the question-answering task
    input_text = f"question: {query} context: {document}"

    # Tokenize the input text
    inputs = t5_tokenizer(input_text, return_tensors="pt", truncation=True, padding="max_length", max_length=512)

    # Generate the answer from the model
    outputs = t5_model.generate(inputs["input_ids"], max_length=150, num_beams=4, early_stopping=True)

    # Decode the output tokens to get the final answer
    answer = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Debug: Print the answer to help debug
    print(f"Generated answer: {answer}")
    
    # Check if answer is empty or too generic
    if not answer or "I don't know" in answer:
        return "Sorry, I couldn't find an answer based on the documents."

    return answer

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_input = request.json['user_input']  # Fetch user input from JSON data

        # Step 1: Get the best folder based on cosine similarity (find the folder)
        best_folder = get_best_folder(user_input)

        if best_folder == "No matching folder found.":
            return jsonify({"response": "Sorry, no relevant folder found for the query."})

        # Step 2: Retrieve the answer from the documents in the selected folder
        response = get_document_info(best_folder, user_input)

        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
