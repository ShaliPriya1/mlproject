import os
import json
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader

# Initialize sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

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

# Loop through the folders in the root folder
for folder_name in os.listdir(root_folder):
    folder_path = os.path.join(root_folder, folder_name)
    
    if os.path.isdir(folder_path):
        # List to hold the folder's document texts and embeddings
        folder_documents = []
        folder_embeddings = []
        
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
                embedding = model.encode(text)  # Generate embedding for the document text
                folder_embeddings.append(embedding)

        # Only save the folder if it has valid PDFs
        if folder_documents:
            embeddings[folder_name] = folder_embeddings
            documents[folder_name] = folder_documents
            filenames[folder_name] = pdf_files

# Save embeddings, documents, and filenames to a JSON file
with open('embeddings.json', 'w', encoding='utf-8') as f:
    json.dump({
        'embeddings': embeddings,
        'documents': documents,
        'filenames': filenames
    }, f, ensure_ascii=False, indent=4)

print("Embedding extraction complete.")
