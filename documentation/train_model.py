import PyPDF2
import os
import torch
from transformers import BertTokenizer, BertForQuestionAnswering, Trainer, TrainingArguments
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text

# Tokenization and vectorization
def prepare_data(text):
    # Example: Use simple TF-IDF Vectorizer or advanced NLP methods
    vectorizer = TfidfVectorizer(stop_words='english')
    text_data = [text]
    tfidf_matrix = vectorizer.fit_transform(text_data)
    return vectorizer, tfidf_matrix

# Create and train model
def train_model(text):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForQuestionAnswering.from_pretrained('bert-base-uncased')

    # Use a simple example for training
    question = "What is the process for data extraction?" # Replace this with actual training data
    answers = ["Data extraction involves parsing the document."]  # Replace with real data

    # Tokenize the question and answer
    encoding = tokenizer(question, text, return_tensors='pt')
    labels = tokenizer(answers[0], return_tensors='pt')

    # Fine-tuning Bert for QA
    model.train()
    training_args = TrainingArguments(
        output_dir='./results',          # output directory
        num_train_epochs=3,              # number of training epochs
        per_device_train_batch_size=4,   # batch size for training
        per_device_eval_batch_size=8,    # batch size for evaluation
        warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir='./logs',            # directory for storing logs
    )

    trainer = Trainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=None,                  # Can create a custom dataset
    )

    trainer.train()

    # Save the model
    model.save_pretrained("model.bin")
    tokenizer.save_pretrained("tokenizer")

    # Save the vectorizer to use later in app.py
    vectorizer, tfidf_matrix = prepare_data(text)
    with open("vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

if __name__ == "__main__":
    pdf_path = 'your_document.pdf'  # Path to the process documentation PDF
    text = extract_text_from_pdf(pdf_path)
    train_model(text)
