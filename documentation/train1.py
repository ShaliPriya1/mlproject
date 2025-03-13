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

# Step 3: Fine-tune T5 for generative question answering using custom question-answer pairs
def fine_tune_t5(qa_pairs):
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    model = T5ForConditionalGeneration.from_pretrained('t5-small')
    
    # Fine-tuning loop for each question-answer pair
    for question, answer in qa_pairs:
        # Prepare the input text and target text
        input_text = f"question: {question} context: {answer}"  # Context comes from the answer itself
        target_text = answer  # The answer is the target for the model
        
        # Tokenize input and output
        inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
        targets = tokenizer(target_text, return_tensors="pt", padding=True, truncation=True)
        
        # Fine-tune the model
        model.train()
        outputs = model(input_ids=inputs['input_ids'], labels=targets['input_ids'])

        # Backpropagation (optimizer step can be included here if you have a proper training loop)
        loss = outputs.loss  # Calculate the loss
        loss.backward()  # Perform backpropagation to calculate gradients
        print(f"Loss for question: {question}, Loss: {loss.item()}")
        
        # In real-world fine-tuning, you should use an optimizer like Adam and step through batches
        # optimizer = Adam(model.parameters(), lr=1e-5)
        # optimizer.step()
        # optimizer.zero_grad()

    # Save the fine-tuned T5 model
    model.save_pretrained('fine_tuned_t5')
    tokenizer.save_pretrained('fine_tuned_t5')

# Step 4: Running the entire pipeline
def train_models(pdf_path):
    text = extract_text_from_pdf(pdf_path)
    create_embeddings(text)
    
    # Custom question-answer pairs related to your document
    qa_pairs = [
        ("What is the process for data extraction?", "Data extraction involves parsing the document and converting it into a structured format."),
        ("How do we handle errors during data processing?", "Errors during data processing are handled by logging them and providing detailed error messages for troubleshooting."),
        ("What tools are used for data analysis?", "We use Python and R for data analysis, with libraries such as Pandas, Numpy, and Scikit-learn."),
        ("What is the purpose of the document?", "The document outlines the process for data extraction, error handling, and analysis of data."),
        # Add more questions and answers as needed
    ]
    
    fine_tune_t5(qa_pairs)  # Pass the qa_pairs to the fine-tuning function

if __name__ == "__main__":
    pdf_path = 'your_document.pdf'  # Path to the process documentation PDF
    train_models(pdf_path)
