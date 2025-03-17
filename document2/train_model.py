import json
import os
import fitz  # PyMuPDF for reading PDF files
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
    # Use T5 tokenizer to encode the input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
    
    # Generate the embeddings using the T5 model (the outputs contain hidden states)
    with torch.no_grad():
        outputs = model.encoder(inputs['input_ids'])[0]  # Encoder outputs the hidden states
        embedding = outputs.mean(dim=1).cpu().numpy()  # Averaging over the token embeddings
    
    return embedding

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    
    # Iterate over each page and extract text
    for page in doc:
        text += page.get_text()

    return text

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

                # Extract text from the PDF
                text = extract_text_from_pdf(file_path)

                documents[folder_name].append(text)

                # Generate the embedding for the document using T5
                embedding = get_embeddings(text)
                embeddings[folder_name].append(embedding)

# Save embeddings to a file for later use in app.py
with open('embeddings_t5.json', 'w', encoding='utf-8') as f:
    json.dump({'embeddings': embeddings, 'filenames': filenames, 'documents': documents}, f, ensure_ascii=False, indent=4)

print("Model training and embedding generation complete!")
