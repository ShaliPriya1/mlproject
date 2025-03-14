from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import pickle
import numpy as np
import json
import os

# Initialize Flask app
app = Flask(__name__)

# Load the models
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
with open('document_embeddings.pkl', 'rb') as f:
    paragraphs, embeddings = pickle.load(f)

# Load the fine-tuned T5 model
t5_tokenizer = T5Tokenizer.from_pretrained('fine_tuned_t5')
t5_model = T5ForConditionalGeneration.from_pretrained('fine_tuned_t5')

# Store user feedback in a JSON file (for simplicity)
feedback_file = 'feedback.json'
if not os.path.exists(feedback_file):
    with open(feedback_file, 'w') as f:
        json.dump([], f)

# Function to get the most relevant paragraph using semantic search
def semantic_search(query):
    try:
        query_embedding = sentence_model.encode([query])
        
        # Compute cosine similarity
        cosine_similarities = np.dot(embeddings, query_embedding.T).flatten()
        
        # Get the most similar paragraph
        best_match_idx = np.argmax(cosine_similarities)
        return paragraphs[best_match_idx]
    except Exception as e:
        return str(e)

# Function to generate answer using T5
def generate_answer(query, context):
    try:
        input_text = f"question: {query} context: {context}"
        inputs = t5_tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
        
        # Generate answer
        outputs = t5_model.generate(input_ids=inputs['input_ids'], max_length=150)
        answer = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return answer
    except Exception as e:
        return str(e)

# API endpoint to handle user query
@app.route("/ask", methods=["POST"])
def ask():
    try:
        query = request.json.get("query")
        
        # Validate user input
        if not query:
            return jsonify({"error": "Query cannot be empty"}), 400
        
        # Use semantic search to find relevant context
        context = semantic_search(query)
        
        # Use T5 to generate an answer
        answer = generate_answer(query, context)
        
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": f"Error processing the request: {str(e)}"}), 500

# API endpoint to handle user feedback
@app.route("/feedback", methods=["POST"])
def feedback():
    try:
        data = request.json
        query = data.get("query")
        answer = data.get("answer")
        rating = data.get("rating")  # Rating from 1 (bad) to 5 (good)

        # Validate feedback data
        if not query or not answer or not rating:
            return jsonify({"error": "Missing feedback data"}), 400

        # Save feedback to a file
        feedback_data = {"query": query, "answer": answer, "rating": rating}
        with open(feedback_file, 'r') as f:
            feedback_list = json.load(f)
        
        feedback_list.append(feedback_data)

        with open(feedback_file, 'w') as f:
            json.dump(feedback_list, f)
        
        return jsonify({"message": "Feedback saved successfully"})
    except Exception as e:
        return jsonify({"error": f"Error saving feedback: {str(e)}"}), 500

# Function to retrain the model based on feedback (for periodic retraining)
def retrain_model():
    try:
        with open(feedback_file, 'r') as f:
            feedback_list = json.load(f)
        
        if len(feedback_list) > 0:
            print(f"Retraining the model with {len(feedback_list)} feedback samples.")
            # Implement retraining logic using feedback and original data (similar to fine-tuning)
            # You would update the fine-tuning process here with the feedback data
            # This is an advanced step and could involve further processing, model retraining, etc.

            # Save the new fine-tuned model after retraining (you may need to adapt the logic)
            # model.save_pretrained('fine_tuned_t5')
            # tokenizer.save_pretrained('fine_tuned_t5')

            print("Model retrained and saved.")
        else:
            print("No feedback data available to retrain the model.")
    except Exception as e:
        print(f"Error during retraining: {str(e)}")

# Run the Flask application
if __name__ == "__main__":
    app.run(debug=True)
