import os
import json
from transformers import T5ForConditionalGeneration, T5Tokenizer
from PyPDF2 import PdfReader
import torch
import numpy as np
from sentence_transformers import SentenceTransformer

# Initialize T5 model and tokenizer
t5_model = T5ForConditionalGeneration.from_pretrained('t5-small')
t5_tokenizer = T5Tokenizer.from_pretrained('t5-small')

# Initialize Sentence-BERT model for folder name matching
sentence_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Directory containing the folders of documents
root_folder = os.getcwd()  # Using the current working directory

# Function to read PDF files and extract text
def extract_text_from_pdf(pdf_path):
    try:
        with open(pdf_path, "rb") as file:
            reader = PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            return text.strip()
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        return ""

# Initialize data structures
folder_name_embeddings = {}
documents = {}
filenames = {}

# Function to get embeddings from T5 model
def get_embedding_from_t5(text):
    inputs = t5_tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
    with torch.no_grad():
        # Get the hidden states (token embeddings) from the encoder
        outputs = t5_model.encoder(inputs['input_ids'])[0]
        # Average the token embeddings to get a single vector
        embedding = outputs.mean(dim=1).cpu().numpy()
    return embedding

# Function to get folder name embedding using Sentence-BERT
def get_folder_embedding(folder_name):
    return sentence_model.encode([folder_name])[0]  # Return the embedding as a list

# Loop through the folders in the root folder
for folder_name in os.listdir(root_folder):
    folder_path = os.path.join(root_folder, folder_name)
    
    if os.path.isdir(folder_path):
        # List to hold the folder's document texts and embeddings
        folder_documents = []
        folder_embeddings = []
        
        # Get folder embedding (folder name)
        folder_name_embedding = get_folder_embedding(folder_name)
        folder_name_embeddings[folder_name] = folder_name_embedding.tolist()  # Convert numpy array to list
        
        # Check if folder contains PDF files
        pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]
        if not pdf_files:
            print(f"No PDF files found in folder: {folder_name}")
            continue  # Skip folder if no PDF files are found

        # Process PDF files in the folder
        for pdf_file in pdf_files:
            pdf_path = os.path.join(folder_path, pdf_file)
            text = extract_text_from_pdf(pdf_path)
            
            if text:
                folder_documents.append(text)
                embedding = get_embedding_from_t5(text)  # Generate embedding using T5
                folder_embeddings.append(embedding.tolist())  # Convert numpy array to list

        # Only save the folder if it has valid PDFs
        if folder_documents:
            documents[folder_name] = folder_documents
            filenames[folder_name] = pdf_files

# Save embeddings, documents, and filenames to a JSON file
with open('embeddings_t5.json', 'w', encoding='utf-8') as f:
    json.dump({
        'folder_name_embeddings': folder_name_embeddings,
        'documents': documents,
        'filenames': filenames
    }, f, ensure_ascii=False, indent=4)

print("Embedding extraction complete.")
