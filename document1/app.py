from flask import Flask, render_template, request, jsonify
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Initialize Flask app and load SentenceTransformer model
app = Flask(__name__)
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')  # You can use other models as well

# Load preprocessed embeddings and folder information
with open('embeddings.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
    embeddings = data['embeddings']
    filenames = data['filenames']
    documents = data['documents']

# Function to find the best matching folder based on cosine similarity
def get_best_folder(query):
    # Get the embedding for the query using Sentence-Transformer
    query_embedding = model.encode([query])

    folder_similarities = {}

    # Calculate cosine similarity with each folder's embeddings
    for folder, folder_embeddings in embeddings.items():
        # Calculate cosine similarity between the query embedding and folder embeddings
        cosine_sim = cosine_similarity(query_embedding, folder_embeddings)
        folder_similarities[folder] = cosine_sim.max()

    # Find the folder with the highest similarity
    best_folder = max(folder_similarities, key=folder_similarities.get)
    return best_folder

# Function to get complete steps or specific info from a document
def get_document_info(folder, query, return_complete=False):
    folder_docs = documents[folder]
    
    # If complete steps are requested, return all content from the folder
    if return_complete:
        return "\n".join(folder_docs)
    
    # Otherwise, use the query and the documents to generate an answer
    document = "\n".join(folder_docs)  # Combine all documents into one context for the question-answering task
    input_text = f"question: {query} context: {document}"
    
    # Get the embedding for the input text using Sentence-Transformer (we can treat this as the "context")
    input_embedding = model.encode([input_text])

    # Find the most similar document section to the query using cosine similarity
    # (You could split the document into smaller chunks for better precision)
    document_embeddings = model.encode(folder_docs)  # Generate embeddings for each document part
    
    # Calculate cosine similarity between the input text and the document parts
    cosine_similarities = cosine_similarity(input_embedding, document_embeddings)
    
    # Get the most relevant document part
    most_relevant_idx = np.argmax(cosine_similarities)
    return folder_docs[most_relevant_idx]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.form['user_input']
    
    # Get the best folder based on cosine similarity
    best_folder = get_best_folder(user_input)
    
    # If the user asks for complete steps
    if "complete steps" in user_input.lower():
        response = get_document_info(best_folder, user_input, return_complete=True)
    else:
        # Get specific info from the most relevant document using Sentence-Transformer
        response = get_document_info(best_folder, user_input, return_complete=False)
    
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(debug=True)
