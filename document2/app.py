from flask import Flask, render_template, request, jsonify
import json
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Initialize Flask app and load SentenceTransformer model
app = Flask(__name__)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load preprocessed embeddings and folder information
with open('embeddings.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
    embeddings = data['embeddings']
    filenames = data['filenames']
    documents = data['documents']

# Function to find the best matching folder based on cosine similarity
def get_best_folder(query):
    query_embedding = model.encode([query])  # Generate embedding for the query
    query_embedding = query_embedding.tolist()  # Convert numpy array to list

    folder_similarities = {}
    
    # Calculate cosine similarity with each folder's embeddings
    for folder, folder_embeddings in embeddings.items():
        folder_embeddings = np.array(folder_embeddings)  # Convert list back to numpy array if necessary
        cosine_sim = cosine_similarity([query_embedding], folder_embeddings)
        folder_similarities[folder] = cosine_sim.max()  # Get max similarity for each folder
    
    # Find the folder with the highest similarity
    best_folder = max(folder_similarities, key=folder_similarities.get)
    return best_folder

# Function to get complete steps or specific info from a document
def get_document_info(folder, query, return_complete=False):
    folder_docs = documents[folder]
    
    # If complete steps are requested, return all content from the folder
    if return_complete:
        return "\n".join(folder_docs)
    
    # Otherwise, use SentenceTransformer to answer the user's question based on the documents
    document = "\n".join(folder_docs)  # Combine all documents into one context for the question-answering task
    input_text = f"question: {query} context: {document}"
    
    # Generate embedding for the context and query
    inputs = model.encode(input_text)
    
    # Generate the answer using the model (placeholder logic here, could be improved)
    answer = "This is a placeholder answer based on the query."
    
    return answer

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json['user_input']  # Fix the key from 'user_input' in JSON
    # Get the best folder based on cosine similarity
    best_folder = get_best_folder(user_input)
    
    # If the user asks for complete steps
    if "complete steps" in user_input.lower():
        response = get_document_info(best_folder, user_input, return_complete=True)
    else:
        # Get specific info from the most relevant document
        response = get_document_info(best_folder, user_input, return_complete=False)
    
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(debug=True)
