from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import pickle
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the models
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
with open('document_embeddings.pkl', 'rb') as f:
    paragraphs, embeddings = pickle.load(f)

# Load the fine-tuned T5 model
t5_tokenizer = T5Tokenizer.from_pretrained('fine_tuned_t5')
t5_model = T5ForConditionalGeneration.from_pretrained('fine_tuned_t5')

# Function to get the most relevant paragraph using semantic search
def semantic_search(query):
    query_embedding = sentence_model.encode([query])
    
    # Compute cosine similarity
    cosine_similarities = np.dot(embeddings, query_embedding.T).flatten()
    
    # Get the most similar paragraph
    best_match_idx = np.argmax(cosine_similarities)
    return paragraphs[best_match_idx]

# Function to generate answer using T5
def generate_answer(query, context):
    input_text = f"question: {query} context: {context}"
    inputs = t5_tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
    
    # Generate answer
    outputs = t5_model.generate(input_ids=inputs['input_ids'], max_length=150)
    answer = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return answer

# API endpoint to handle user query
@app.route("/ask", methods=["POST"])
def ask():
    query = request.json.get("query")
    
    # Use semantic search to find relevant context
    context = semantic_search(query)
    
    # Use T5 to generate an answer
    answer = generate_answer(query, context)
    
    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(debug=True)
