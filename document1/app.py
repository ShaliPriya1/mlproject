from flask import Flask, render_template, request, jsonify
import json
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA

# Initialize Flask app and load T5 model
app = Flask(__name__)
model = T5ForConditionalGeneration.from_pretrained('t5-small')  # You can use a larger model like 't5-base' or 't5-large'
tokenizer = T5Tokenizer.from_pretrained('t5-small')

# Load preprocessed embeddings and folder information
with open('embeddings.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
    embeddings = data['embeddings']
    filenames = data['filenames']
    documents = data['documents']

# Function to find the best matching folder based on cosine similarity
def get_best_folder(query):
    # Tokenize the query using T5 tokenizer
    inputs = tokenizer(query, return_tensors="pt", truncation=True, padding=True, max_length=512)

    # Get the encoder output for the query (no generation required)
    with torch.no_grad():
        encoder_outputs = model.encoder(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])

    # We use the last hidden state as the query embedding
    query_embedding = encoder_outputs.last_hidden_state.mean(dim=1)  # Take the mean of the last hidden state
    
    # Ensure the query embedding is valid
    query_embedding = query_embedding.cpu().numpy()
    if query_embedding.shape[1] == 0:
        raise ValueError("Query embedding has zero features. Check the encoder output.")

    # Reshape the query embedding to 2D (1, embedding_size)
    query_embedding = query_embedding.reshape(1, -1)

    folder_similarities = {}

    # Apply PCA to match dimensions of query and folder embeddings
    pca = PCA(n_components=query_embedding.shape[1])  # Match the number of components to query embedding size

    # Calculate cosine similarity with each folder's embeddings
    for folder, folder_embeddings in embeddings.items():
        # Ensure folder embeddings are not empty and reshape them if needed
        folder_embeddings = np.array(folder_embeddings)
        
        if folder_embeddings.shape[0] == 0:
            raise ValueError(f"Folder embeddings for '{folder}' are empty.")
        
        # Reshape folder embeddings to 2D (n_samples, embedding_size) if necessary
        folder_embeddings = folder_embeddings.reshape(1, -1)

        # Apply PCA to reduce folder embedding dimensionality to match query embedding
        folder_embeddings_pca = pca.fit_transform(folder_embeddings)

        # Calculate cosine similarity
        cosine_sim = cosine_similarity(query_embedding, folder_embeddings_pca)
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
    
    # Otherwise, use T5 to answer the user's question based on the documents
    document = "\n".join(folder_docs)  # Combine all documents into one context for the question-answering task
    input_text = f"question: {query} context: {document}"
    
    # Tokenize the input text
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
    
    # Generate the answer from the model
    outputs = model.generate(inputs["input_ids"], max_length=150, num_beams=4, early_stopping=True)
    
    # Decode the output tokens to get the final answer
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return answer

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
