import PyPDF2
import os
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
import torch

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

# Function to fine-tune T5 model on the extracted text
def fine_tune_t5_model(text):
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    model = T5ForConditionalGeneration.from_pretrained('t5-small')

    # Example question-answer pairs (you will need to manually create these from your document)
    qa_pairs = [
        {"question": "What is the process for data extraction?", "answer": "Data extraction involves parsing the document and converting it into a structured format."},
        {"question": "What is the format of data?", "answer": "The data is usually in CSV or JSON format."},
        # Add more QA pairs related to your document...
    ]
    
    # Tokenizing the input and output for fine-tuning
    inputs = [f"question: {qa['question']} context: {text}" for qa in qa_pairs]
    targets = [qa['answer'] for qa in qa_pairs]

    input_encodings = tokenizer(inputs, truncation=True, padding=True)
    target_encodings = tokenizer(targets, truncation=True, padding=True)

    # Prepare dataset
    class QA_Dataset(torch.utils.data.Dataset):
        def __init__(self, input_encodings, target_encodings):
            self.input_encodings = input_encodings
            self.target_encodings = target_encodings

        def __getitem__(self, idx):
            return {'input_ids': torch.tensor(self.input_encodings['input_ids'][idx]),
                    'attention_mask': torch.tensor(self.input_encodings['attention_mask'][idx]),
                    'labels': torch.tensor(self.target_encodings['input_ids'][idx])}

        def __len__(self):
            return len(self.input_encodings['input_ids'])

    dataset = QA_Dataset(input_encodings, target_encodings)

    # Training arguments
    training_args = TrainingArguments(
        output_dir='./results', 
        evaluation_strategy="epoch", 
        learning_rate=2e-5, 
        per_device_train_batch_size=4,
        num_train_epochs=3, 
        weight_decay=0.01
    )

    trainer = Trainer(
        model=model, 
        args=training_args, 
        train_dataset=dataset
    )

    trainer.train()
    
    # Save the fine-tuned model
    model.save_pretrained('fine_tuned_t5')
    tokenizer.save_pretrained('fine_tuned_t5')

# Call the fine-tuning function
fine_tune_t5_model(text)
