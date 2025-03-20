import torch
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from transformers import AutoTokenizer, DebertaV2ForSequenceClassification
import seaborn as sns
import matplotlib.pyplot as plt

# Load the benchmark dataset
benchmark_df = pd.read_csv("Data/benchmark_dataset.csv")  # Load your benchmark dataset
benchmark_df = benchmark_df[["label", "text"]]  # Keep only necessary columns
benchmark_df.columns = ["label", "text"]  # Rename columns for consistency

# Check for NaN values in the label column
if benchmark_df["label"].isna().any():
    print("Warning: NaN values found in the 'label' column. Removing them.")
    benchmark_df = benchmark_df.dropna(subset=["label"])

# Load the pre-trained model and tokenizer
model_path = "Models/deberta_scam_detector_v3"  # Path to your trained model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = DebertaV2ForSequenceClassification.from_pretrained(model_path).to("cuda" if torch.cuda.is_available() else "cpu")
model.eval()  # Set the model to evaluation mode

# Tokenize the benchmark dataset
def tokenize_data(df):
    return tokenizer(df["text"].tolist(), truncation=True, padding="max_length", max_length=256, return_tensors="pt")

benchmark_encodings = tokenize_data(benchmark_df)

# Convert labels to tensors
benchmark_labels = torch.tensor(benchmark_df["label"].tolist())

# Create a PyTorch dataset
class ScamDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)

benchmark_dataset = ScamDataset(benchmark_encodings, benchmark_labels)

# Function to predict labels
def predict(dataset):
    predictions = []
    for i in range(len(dataset)):
        inputs = {key: val[i].unsqueeze(0).to("cuda" if torch.cuda.is_available() else "cpu") for key, val in benchmark_encodings.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(probs).item()
            predictions.append(predicted_class)
    return predictions

# Run predictions on the benchmark dataset
print("Running predictions on the benchmark dataset...")
predicted_labels = predict(benchmark_dataset)

# Print evaluation metrics
print("Classification Report:")
print(classification_report(benchmark_df["label"], predicted_labels, target_names=["Ham", "Spam", "Phishing"], labels=[0, 1, 2]))

# Confusion Matrix
conf_matrix = confusion_matrix(benchmark_df["label"], predicted_labels, labels=[0, 1, 2])
print("Confusion Matrix:")
print(conf_matrix)

# Plot Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Ham", "Spam", "Phishing"], yticklabels=["Ham", "Spam", "Phishing"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Additional Metrics
accuracy = accuracy_score(benchmark_df["label"], predicted_labels)
precision = precision_score(benchmark_df["label"], predicted_labels, average="weighted")
recall = recall_score(benchmark_df["label"], predicted_labels, average="weighted")
f1 = f1_score(benchmark_df["label"], predicted_labels, average="weighted")

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

# Test the model on real-world sample texts
def predict_text(texts):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=256).to("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        outputs = model(**inputs)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
    predictions = torch.argmax(outputs.logits, dim=1)
    return predictions.cpu().tolist(), probabilities.cpu().tolist()

# Sample texts for testing
sample_texts = {
    "Ham (Legitimate)": [
        "Hey, are we still meeting at 5 PM today?",
        "Your package has been delivered. Check your doorstep.",
        "Reminder: Your doctor's appointment is tomorrow at 10 AM.",
    ],
    "Spam (Promotional)": [
        "Win a free iPhone! Click here to claim your prize.",
        "Limited time offer: Get 50% off on all products. Shop now!",
        "Congratulations! You've been selected for a $1000 Walmart gift card.",
    ],
    "Phishing (Fraudulent)": [
        "Your account has been compromised. Click this link to secure it.",
        "We detected unusual activity in your bank account. Verify your details here.",
        "Your PayPal account has been locked. Click here to unlock it.",
    ]
}

# Test the model with sample texts
for category, texts in sample_texts.items():
    print(f"\nTesting {category} Messages:")
    predictions, probabilities = predict_text(texts)
    for i, text in enumerate(texts):
        pred_class = predictions[i]
        confidence = probabilities[i][pred_class] * 100
        print(f"Text: {text}")
        print(f"Prediction: {pred_class} ({'Ham (Legit)' if pred_class == 0 else 'Spam' if pred_class == 1 else 'Phishing'})")
        print(f"Confidence: {confidence:.2f}%")
        print("-" * 50)