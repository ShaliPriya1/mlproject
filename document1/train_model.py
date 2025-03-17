import os
import json
import fitz  # PyMuPDF
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize Sentence Transformer model for embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')  # You can change this model to any other sentence transformer model.

# Directory containing the folders of documents
root_folder = "path_to_your_document_folders"  # Replace this with the path to your folders

# Function to read all PDF documents in a folder and store their contents
def load_documents_from_folder(folder_path):
    documents = []
    filenames = []
    
    # Iterate over all files in the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        # Check if it's a PDF file
        if os.path.isfile(file_path) and file_path.endswith(".pdf"):  # Ensure we process only PDFs
            with open(file_path, 'rb') as file:
                pdf_document = fitz.open(file)
                pdf_text = ""
                
                # Iterate over each page and extract text
                for page_num in range(pdf_document.page_count):
                    page = pdf_document.load_page(page_num)
                    pdf_text += page.get_text()
                
                documents.append(pdf_text)  # Append the extracted text of the PDF
                filenames.append(filename)  # Store the filename
    
    return documents, filenames

# Function to create embeddings for a set of documents
def create_embeddings_for_documents(documents):
    embeddings = model.encode(documents)  # Generate embeddings for the documents
    return embeddings

# Function to process each folder and generate embeddings
def process_folders(root_folder):
    all_embeddings = {}
    all_filenames = {}
    all_documents = {}
    
    # Iterate through each folder in the root directory
    for folder_name in os.listdir(root_folder):
        folder_path = os.path.join(root_folder, folder_name)
        
        # Check if it's a directory (folder)
        if os.path.isdir(folder_path):
            # Check if the folder contains any PDF files
            pdf_files = [f for f in os.listdir(folder_path) if f.endswith(".pdf")]
            
            # Skip the folder if it contains no PDF files
            if not pdf_files:
                continue
            
            # Load documents from the folder
            documents, filenames = load_documents_from_folder(folder_path)
            
            # Create embeddings for the documents
            embeddings = create_embeddings_for_documents(documents)
            
            # Store the embeddings, filenames, and documents
            all_embeddings[folder_name] = embeddings
            all_filenames[folder_name] = filenames
            all_documents[folder_name] = documents
    
    return all_embeddings, all_filenames, all_documents

# Function to save the embeddings and documents to a JSON file
def save_embeddings_to_json(embeddings, filenames, documents, output_file="embeddings.json"):
    data = {
        "embeddings": embeddings,
        "filenames": filenames,
        "documents": documents
    }
    
    # Save the embeddings as a JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"Embeddings saved to {output_file}")

# Main function to execute the entire process
def main():
    # Process the folders to generate embeddings and document data
    all_embeddings, all_filenames, all_documents = process_folders(root_folder)
    
    # Save the embeddings, filenames, and documents to a JSON file
    save_embeddings_to_json(all_embeddings, all_filenames, all_documents)

if __name__ == "__main__":
    main()
