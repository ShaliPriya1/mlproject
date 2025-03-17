import json
import os
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import numpy as np

# Load T5 model and tokenizer
model = T5ForConditionalGeneration.from_pretrained('t5-small')  # You can use 't5-base' or 't5-large' for better performance
tokenizer = T5Tokenizer.from_pretrained('t5-small')

# Directory containing the folders of documents (Use the current directory)
root_folder = os.getcwd()

# Initialize data storage
documents = {}
embeddings = {}
filenames = {}

# Function to generate embeddings for a document using T5
def get_embeddings(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
    with torch.no_grad():
        outputs = model.encode(inputs['input_ids'])  # Using model.encode might need changes as T5 is generative
        embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()  # Take the mean of the hidden states as the embedding
    return embedding

# Loop through directories (each folder contains documents)
for folder_name in os.listdir(root_folder):
    folder_path = os.path.join(root_folder, folder_name)
    
    if os.path.isdir(folder_path):  # Ensure we're working with folders only
        documents[folder_name] = []
        embeddings[folder_name] = []
        filenames[folder_name] = []
        
        # Loop through the files inside each folder
        for filename in os.listdir(folder_path):
            if filename.endswith(".pdf"):  # Process only PDF files
                file_path = os.path.join(folder_path, filename)
                filenames[folder_name].append(filename)

                # Open and read the PDF content
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()

                documents[folder_name].append(text)

                # Generate the embedding for the document using T5
                embedding = get_embeddings(text)
                embeddings[folder_name].append(embedding)

# Save embeddings to a file for later use in app.py
with open('embeddings_t5.json', 'w', encoding='utf-8') as f:
    json.dump({'embeddings': embeddings, 'filenames': filenames, 'documents': documents}, f, ensure_ascii=False, indent=4)

print("Model training and embedding generation complete!")
