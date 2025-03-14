import PyPDF2
import os
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
import torch
from sklearn.model_selection import train_test_split
import glob
import pathlib
import re

# Step 1: Extract text from multiple PDF files
def extract_text_from_pdfs(pdf_paths):
    text = ""
    for pdf_path in pdf_paths:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text()
    return text

pdf_paths = ['file1.pdf', 'file2.pdf', 'file3.pdf', 'file4.pdf', 'file5.pdf']
text = extract_text_from_pdfs(pdf_paths)

# Save the extracted text for future use
with open('processed_text.txt', 'w') as f:
    f.write(text)


def fine_tune_t5_model(text):
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    model = T5ForConditionalGeneration.from_pretrained('t5-small')

    # Example question-answer pairs
    qa_pairs = [
        {"question": "What is the process for data extraction?", "answer": "Data extraction involves parsing the document and converting it into a structured format."},
        {"question": "What is the format of data?", "answer": "The data is usually in CSV or JSON format."},
        # Add more QA pairs related to your document...
    ]
    
    # Prepare the inputs and targets
    inputs = [f"question: {qa['question']} context: {text}" for qa in qa_pairs]
    targets = [qa['answer'] for qa in qa_pairs]

    # Split into train and validation sets (80% train, 20% validation)
    train_inputs, eval_inputs, train_targets, eval_targets = train_test_split(inputs, targets, test_size=0.2)

    # Tokenize the train and eval inputs/targets
    train_input_encodings = tokenizer(train_inputs, truncation=True, padding=True)
    train_target_encodings = tokenizer(train_targets, truncation=True, padding=True)

    eval_input_encodings = tokenizer(eval_inputs, truncation=True, padding=True)
    eval_target_encodings = tokenizer(eval_targets, truncation=True, padding=True)

    # Define a dataset class
    class QA_Dataset(torch.utils.data.Dataset):
        def __init__(self, input_encodings, target_encodings):
            self.input_encodings = input_encodings
            self.target_encodings = target_encodings

        def __getitem__(self, idx):
            return {
                'input_ids': torch.tensor(self.input_encodings['input_ids'][idx]),
                'attention_mask': torch.tensor(self.input_encodings['attention_mask'][idx]),
                'labels': torch.tensor(self.target_encodings['input_ids'][idx])
            }

        def __len__(self):
            return len(self.input_encodings['input_ids'])

    # Create datasets for train and eval
    train_dataset = QA_Dataset(train_input_encodings, train_target_encodings)
    eval_dataset = QA_Dataset(eval_input_encodings, eval_target_encodings)

    # Set training arguments
    training_args = TrainingArguments(
        output_dir='./results', 
        evaluation_strategy="epoch", 
        learning_rate=2e-5, 
        per_device_train_batch_size=4,
        num_train_epochs=3, 
        weight_decay=0.01
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset  # Pass the eval dataset
    )

    # Train the model
    trainer.train()

    # Save the fine-tuned model
    model.save_pretrained('fine_tuned_t5')
    tokenizer.save_pretrained('fine_tuned_t5')

# Assuming 'text' is the extracted document text
text = "Here is your document text. Add more text as needed."
fine_tune_t5_model(text)
