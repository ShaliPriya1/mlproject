import PyPDF2
import os
import torch
from sentence_transformers import SentenceTransformer
from transformers import T5Tokenizer, T5ForConditionalGeneration
import pickle

# Step 1: Extract text from PDF
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text

# Step 2: Embed document with Sentence-Transformer
def create_embeddings(text):
    # Split text into chunks (e.g., paragraphs or sections)
    paragraphs = text.split('\n')
    
    # Load pre-trained Sentence-Transformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Create embeddings for each paragraph
    embeddings = model.encode(paragraphs)
    
    # Save embeddings and corresponding paragraphs
    with open('document_embeddings.pkl', 'wb') as f:
        pickle.dump((paragraphs, embeddings), f)

# Step 3: Fine-tune T5 for generative question answering
def fine_tune_t5(text):
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    model = T5ForConditionalGeneration.from_pretrained('t5-small')
    
    # Sample training data, you need to extract questions and answers from the text.
    question = "What is the process for data extraction?"  # Example
    answer = "Data extraction involves parsing the document and converting it into a structured format."  # Example
    
    # Tokenize input and output
    input_text = f"question: {question} context: {text}"
    target_text = answer
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
    targets = tokenizer(target_text, return_tensors="pt", padding=True, truncation=True)
    
    # Fine-tune model (this is just an example, you'd need a proper dataset for actual training)
    model.train()
    outputs = model(input_ids=inputs['input_ids'], labels=targets['input_ids'])
    
    # Save the fine-tuned T5 model
    model.save_pretrained('fine_tuned_t5')
    tokenizer.save_pretrained('fine_tuned_t5')

# Step 4: Running the entire pipeline
def train_models(pdf_path):
    text = extract_text_from_pdf(pdf_path)
    create_embeddings(text)
    fine_tune_t5(text)

if __name__ == "__main__":
    pdf_path = 'your_document.pdf'  # Path to the process documentation PDF
    train_models(pdf_path)
