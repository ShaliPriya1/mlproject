import os
import json
from sentence_transformers import SentenceTransformer
from transformers import T5ForConditionalGeneration, T5Tokenizer
from PyPDF2 import PdfReader
import torch
import numpy as np

# Initialize Sentence Transformer model for folder name embeddings
sentence_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')  # Sentence-BERT model

# Initialize T5 model and tokenizer for document answering
t5_model = T5ForConditionalGeneration.from_pretrained('t5-small')
t5_tokenizer = T5Tokenizer.from_pretrained('t5-small')

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
folder_name_embeddings = {}  # To store embeddings for folder names (Sentence-BERT)
documents = {}  # To store documents in each folder
filenames = {}  # To store filenames for each folder

# Function to get embeddings from T5 model (for document answering)
def get_t5_embedding(text):
    inputs = t5_tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
    with torch.no_grad():
        # Get the hidden states (token embeddings) from the encoder
        outputs = t5_model.encoder(inputs['input_ids'])[0]
        # Average the token embeddings to get a single vector
        embedding = outputs.mean(dim=1).cpu().numpy()
    return embedding

# Function to get Sentence-BERT embeddings (for folder name matching)
def get_sentence_embedding(text):
    return sentence_model.encode(text)

# Loop through the folders in the root folder
for folder_name in os.listdir(root_folder):
    folder_path = os.path.join(root_folder, folder_name)
    
    if os.path.isdir(folder_path):
        # List to hold the folder's document texts and their embeddings
        folder_documents = []
        folder_document_embeddings = []  # Embeddings for documents within the folder
        
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
                embedding = get_t5_embedding(text)  # Generate embedding using T5
                folder_document_embeddings.append(embedding.tolist())  # Convert the numpy array to a list

        # Generate and store the folder embedding (using Sentence-BERT for folder name)
        folder_name_embeddings[folder_name] = get_sentence_embedding(folder_name)

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
