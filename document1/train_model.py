import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize Sentence Transformer model for embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')  # You can change this model to any other sentence transformer model.

# Directory containing the folders of documents
root_folder = "path_to_your_document_folders"  # Replace this with the path to your folders

# Function to read all documents in a folder and store their contents
def load_documents_from_folder(folder_path):
    documents = []
    filenames = []
    
    # Iterate over all files in the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        # Check if it's a text file
        if os.path.isfile(file_path) and file_path.endswith(".txt"):  # Modify this to handle other formats if needed
            with open(file_path, 'r', encoding='utf-8') as file:
                documents.append(file.read())  # Read the entire content of the document
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
        
        # Only process folders (skip files)
        if os.path.isdir(folder_path):
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
