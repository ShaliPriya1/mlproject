from flask import Flask, render_template, request, jsonify
import json
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Initialize Flask app and load T5 model
app = Flask(__name__)
model = T5ForConditionalGeneration.from_pretrained('t5-small')
tokenizer = T5Tokenizer.from_pretrained('t5-small')

# Load preprocessed embeddings and folder information
with open('embeddings_t5.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
    embeddings = data['embeddings']
    filenames = data['filenames']
    documents = data['documents']

# Function to generate embeddings using T5 for a given text
def get_embeddings(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
    with torch.no_grad():
        outputs = model.encode(inputs['input_ids'])  # Generate embedding
        embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()  # Averaging over the token embeddings
    return embedding

# Function to find the best matching folder based on cosine similarity
def get_best_folder(query):
    query_embedding = get_embeddings(query)  # Get embedding for the query
    folder_similarities = {}

    # Compare query embedding with each folder's embeddings
    for folder, folder_embeddings in embeddings.items():
        cosine_sim = cosine_similarity(query_embedding, np.array(folder_embeddings))
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
    document = "\n".join(folder_docs)
    input_text = f"question: {query} context: {document}"

    # Get the embedding for the input text (question + context)
    input_embedding = get_embeddings(input_text)

    # Find the most similar document section to the query using cosine similarity
    document_embeddings = [get_embeddings(doc) for doc in folder_docs]
    cosine_similarities = cosine_similarity(input_embedding, np.array(document_embeddings))

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
        # Get specific info from the most relevant document using T5
        response = get_document_info(best_folder, user_input, return_complete=False)

    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(debug=True)
