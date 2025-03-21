from flask import Flask, render_template, request, jsonify
import json
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Initialize Flask app and load T5 model
app = Flask(__name__)
model = T5ForConditionalGeneration.from_pretrained('t5-small')  # You can use 't5-base' or 't5-large'
tokenizer = T5Tokenizer.from_pretrained('t5-small')

# Load preprocessed embeddings and folder information
with open('embeddings_t5.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
    embeddings = data['embeddings']
    filenames = data['filenames']
    documents = data['documents']
    folder_names_embeddings = data['folder_names_embeddings']  # Load the folder name embeddings

# Function to find the best matching folder based on cosine similarity
def get_best_folder(query):
    # Check if the query is empty
    if not query.strip():
        return "Please provide a valid query."  # Handle empty query case

    # Tokenize and encode the query text using T5
    inputs = tokenizer(query, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
    
    with torch.no_grad():
        # Get the hidden states (token embeddings) from the encoder
        outputs = model.encoder(inputs['input_ids'])[0]
        
        # Average the token embeddings to get a single vector
        query_embedding = outputs.mean(dim=1).cpu().numpy()
    
    # Ensure query embedding is a 2D array (1, 512)
    query_embedding = query_embedding.reshape(1, -1)
    
    print(f"Query embedding shape: {query_embedding.shape}")
    if query_embedding.shape[1] == 0:
        return "Error: Query embedding has no features."

    folder_similarities = {}
    
    # Calculate cosine similarity with each folder's name embedding
    for folder, folder_name_embedding in folder_names_embeddings.items():
        folder_embedding = np.array(folder_name_embedding)  # Convert list to numpy array
        
        # Ensure folder embedding is a 2D array (1, 512)
        folder_embedding = folder_embedding.reshape(1, -1)
        
        # Log folder name embedding and query embedding for debugging
        print(f"Folder: {folder}, Folder name embedding: {folder_embedding}")
        print(f"Query embedding: {query_embedding}")

        if folder_embedding.shape[1] == 0:
            return f"Error: Folder embedding for {folder} has no features."

        # Calculate cosine similarity
        cosine_sim = cosine_similarity(query_embedding, folder_embedding)  # Calculate similarity
        folder_similarities[folder] = cosine_sim.max()  # Use the maximum cosine similarity for each folder
    
    # Log folder similarities for debugging
    print(f"Folder similarities: {folder_similarities}")

    # Find the folder with the highest similarity
    if folder_similarities:
        best_folder = max(folder_similarities, key=folder_similarities.get)
        print(f"Best folder: {best_folder} with similarity {folder_similarities[best_folder]}")
        return best_folder
    else:
        return "No matching folder found."

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
    
    print(f"Generated answer: {answer}")
    
    return answer

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_input = request.json['user_input']  # Fetch user input from JSON data
        
        # Check if input is received properly
        print(f"User input received: {user_input}")
        
        # Step 1: Get the best folder based on cosine similarity (find the folder)
        best_folder = get_best_folder(user_input)
        print(f"Best folder: {best_folder}")
        
        if best_folder == "No matching folder found.":
            return jsonify({"response": "Sorry, no relevant folder found for the query."})
        
        # Step 2: Retrieve the answer from the documents in the selected folder
        response = get_document_info(best_folder, user_input, return_complete=False)
        
        print(f"Response: {response}")
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
