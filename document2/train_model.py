import os
import json
from transformers import T5ForConditionalGeneration, T5Tokenizer
from PyPDF2 import PdfReader
import torch
import numpy as np

# Initialize T5 model and tokenizer
model = T5ForConditionalGeneration.from_pretrained('t5-small')
tokenizer = T5Tokenizer.from_pretrained('t5-small')

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
embeddings = {}
documents = {}
filenames = {}
folder_names_embeddings = {}  # To store folder name embeddings

# Function to get embeddings from T5 model
def get_embedding_from_t5(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
    with torch.no_grad():
        # Get the hidden states (token embeddings) from the encoder
        outputs = model.encoder(inputs['input_ids'])[0]
        # Average the token embeddings to get a single vector
        embedding = outputs.mean(dim=1).cpu().numpy()
    return embedding

# Loop through the folders in the root folder
for folder_name in os.listdir(root_folder):
    folder_path = os.path.join(root_folder, folder_name)
    
    if os.path.isdir(folder_path):
        # List to hold the folder's document texts and embeddings
        folder_documents = []
        folder_embeddings = []
        
        # Embed the folder name itself
        folder_name_embedding = get_embedding_from_t5(folder_name)  # Generate embedding for folder name
        folder_names_embeddings[folder_name] = folder_name_embedding.tolist()  # Save it
        
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
                folder_embeddings.append(embedding.tolist())  # Convert the numpy array to a list

        # Only save the folder if it has valid PDFs
        if folder_documents:
            embeddings[folder_name] = folder_embeddings
            documents[folder_name] = folder_documents
            filenames[folder_name] = pdf_files

# Save embeddings, documents, and filenames to a JSON file
with open('embeddings_t5.json', 'w', encoding='utf-8') as f:
    json.dump({
        'embeddings': embeddings,
        'documents': documents,
        'filenames': filenames,
        'folder_names_embeddings': folder_names_embeddings  # Include the folder name embeddings
    }, f, ensure_ascii=False, indent=4)

print("Embedding extraction complete.")
