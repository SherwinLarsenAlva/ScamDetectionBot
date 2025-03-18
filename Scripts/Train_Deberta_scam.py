import torch
import pandas as pd
import numpy as np
from transformers import (
    DebertaV2ForSequenceClassification, 
    DebertaV2Tokenizer, 
    Trainer, 
    TrainingArguments,
    EarlyStoppingCallback
)
from datasets import Dataset
from sklearn.utils.class_weight import compute_class_weight
from torch.nn import CrossEntropyLoss
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Define compute_metrics function
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")

# Load the dataset
full_data = pd.read_csv("updated_scam_dataset.csv")

# Convert to Hugging Face Dataset
dataset = Dataset.from_pandas(full_data)

# Load tokenizer
tokenizer = DebertaV2Tokenizer.from_pretrained("microsoft/deberta-v3-base")

# Tokenization function
def tokenize_function(examples):
    return tokenizer(
        examples["text"], 
        padding="max_length", 
        truncation=True, 
        max_length=128  # Increased max_length for better context capture
    )

# Apply tokenization
dataset = dataset.map(tokenize_function, batched=True)
dataset = dataset.remove_columns(["text"])
dataset = dataset.rename_column("label", "labels")
dataset.set_format("torch")

# Ensure labels are of type long for CrossEntropyLoss
dataset = dataset.map(lambda x: {"labels": x["labels"].long()}, batched=True)

# Split into train and test sets with shuffle enabled
train_test = dataset.train_test_split(test_size=0.2, shuffle=True, seed=42)
train_dataset, test_dataset = train_test["train"], train_test["test"]

# Compute class weights
class_weights = compute_class_weight("balanced", classes=np.array([0, 1, 2]), y=full_data["label"])
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
print(f"📊 Class Weights: {class_weights}")

# Define custom Trainer with class-weighted loss
class CustomTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels").to(torch.long)  # Ensure labels are int64
        outputs = model(**inputs)
        logits = outputs.logits.to(torch.float32)  # Ensure logits are float32
        
        loss_fct = CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


# Load the model and move it to GPU
model = DebertaV2ForSequenceClassification.from_pretrained("microsoft/deberta-v3-base", num_labels=3).to(device)

# Freeze base model layers for efficient fine-tuning
for param in model.base_model.parameters():
    param.requires_grad = False

# Unfreeze classifier layers
for param in model.classifier.parameters():
    param.requires_grad = True

# Training Arguments
training_args = TrainingArguments(
    output_dir="./deberta_scam_detector_v1",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    num_train_epochs=5,  # Increased for better fine-tuning
    per_device_train_batch_size=8,  # Increased batch size for stability
    per_device_eval_batch_size=8,
    learning_rate=3e-5,  # Slightly increased LR for better adaptation
    weight_decay=0.01,  # Regularization to prevent overfitting
    logging_dir="./logs",
    logging_steps=200,  # More frequent logging
    fp16=True,  # Mixed precision for faster training
    push_to_hub=False,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False
)

# Initialize trainer with EarlyStopping
trainer = CustomTrainer(
    class_weights=class_weights,
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]  # Improved patience setting
)

# Train the model
trainer.train()

# Save the model and tokenizer
trainer.save_model("deberta_scam_detector_v1")
tokenizer.save_pretrained("deberta_scam_detector_v1")
print("✅ Model retrained and saved successfully!")
