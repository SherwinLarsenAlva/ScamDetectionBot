import torch
import pandas as pd
from transformers import AutoTokenizer, DebertaV2ForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding, EarlyStoppingCallback
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import os
from dotenv import load_dotenv
from huggingface_hub import HfApi

# Load environment variables (for Hugging Face token)
load_dotenv()

# Load the synthetic dataset
df = pd.read_csv("Data\synthetic_scam_dataset.csv")

# Convert labels (Ensure consistent mapping)
df["label_id"] = df["label"].astype(int)  # Already labeled as 0 (Ham), 1 (Spam), 2 (Phishing)

# Train-validation split
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["text"].tolist(), df["label_id"].tolist(), test_size=0.2, random_state=42
)

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("Models\deberta_scam_detector_v3")
model = DebertaV2ForSequenceClassification.from_pretrained("Models\deberta_scam_detector_v3").to("cuda" if torch.cuda.is_available() else "cpu")

# Unfreeze part of the model for gradual fine-tuning
for name, param in model.named_parameters():
    if "classifier" in name or "pooler" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

# Dataset class
class ScamDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        inputs = tokenizer(
            self.texts[idx], truncation=True, padding="max_length", max_length=256, return_tensors="pt"
        )
        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }

train_dataset = ScamDataset(train_texts, train_labels)
val_dataset = ScamDataset(val_texts, val_labels)

# Data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Training arguments with early stopping and learning rate scheduler
training_args = TrainingArguments(
    output_dir="./Models/deberta_scam_detector_v4",  
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
    load_best_model_at_end=True,
    logging_dir="./logs",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    gradient_accumulation_steps=4,
    fp16=True,
    logging_steps=10,
    eval_steps=100,
    gradient_checkpointing=True,
    max_grad_norm=1.0,
    lr_scheduler_type="cosine",
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model
trainer.save_model("./Models/deberta_scam_detector_v4")
tokenizer.save_pretrained("./Models/deberta_scam_detector_v4")

print("Fine-tuning complete. Model saved to './Models/deberta_scam_detector_v4'")

# Upload the fine-tuned model to Hugging Face
token = os.getenv("HUGGINGFACE_TOKEN")
if not token:
    raise ValueError("Hugging Face token not found in .env file.")

api = HfApi()
repo_id = "SparkyPilot/ScamDetector"  # Your Hugging Face repository

# Upload all files in the fine-tuned model directory
for root, _, files in os.walk("./Models/deberta_scam_detector_v4"):
    for file_name in files:
        file_path = os.path.join(root, file_name)
        relative_path = os.path.relpath(file_path, "./Models/deberta_scam_detector_v4")  # Get relative path
        api.upload_file(
            path_or_fileobj=file_path,
            path_in_repo=relative_path,
            repo_id=repo_id,
            token=token,
        )
        print(f"Uploaded: {file_path} -> {relative_path}")

print("Fine-tuned model uploaded to Hugging Face successfully!")