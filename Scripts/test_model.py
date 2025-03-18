import torch
from transformers import AutoTokenizer, DebertaV2ForSequenceClassification

# Load trained model
model_path = "deberta_scam_detector_v3"
tokenizer = AutoTokenizer.from_pretrained(model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DebertaV2ForSequenceClassification.from_pretrained(model_path).to(device)
model.eval()  # Set model to evaluation mode

# Function to predict the class of a given text
def predict_text(texts):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)  # Convert logits to probabilities
    predictions = torch.argmax(outputs.logits, dim=1)  # Get predicted class indices

    return predictions.cpu().tolist(), probabilities.cpu().tolist()  # Convert to lists for easier handling

# Sample texts for testing
sample_texts = {
    "Ham (Legitimate)": [
        "Hey, are we still meeting at 5 PM today?",
        "Your package has been delivered. Check your doorstep.",
        "Reminder: Your doctor's appointment is tomorrow at 10 AM.",
        "Thanks for your payment of $50. Your transaction ID is 12345.",
        "Congratulations! You've won a free ticket to the event. Click here to claim: [link].",
        "Your flight has been confirmed. Check your email for details.",
        "Your subscription to Netflix has been renewed successfully.",
        "Your order #12345 has been shipped and will arrive by Friday.",
        "Your recent transaction of $20 at Starbucks was successful.",
        "Your account balance is $500. Log in to view details."
    ],
    "Spam (Promotional)": [
        "Win a free iPhone! Click here to claim your prize.",
        "Limited time offer: Get 50% off on all products. Shop now!",
        "Congratulations! You've been selected for a $1000 Walmart gift card.",
        "Earn money from home! Start today: [link].",
        "Hot singles in your area are waiting to meet you: [link].",
        "Get a free trial of our premium service. Sign up now!",
        "You've won a luxury vacation! Click here to claim: [link].",
        "Exclusive deal: Buy one, get one free on all items!",
        "Your chance to win a $500 Amazon gift card! Click here: [link].",
        "Limited stock: Get the latest smartphone at 70% off!"
    ],
    "Phishing (Fraudulent)": [
        "Your account has been compromised. Click this link to secure it.",
        "We detected unusual activity in your bank account. Verify your details here.",
        "Your PayPal account has been locked. Click here to unlock it. toll fee:12319823u198",
        "Your Netflix subscription has expired. Update your payment info here: [link].",
        "Your tax refund is pending. Click here to claim it: [link].",
        "Your Amazon account has been suspended. Click here to reactivate: [link].",
        "Your Google account has been hacked. Secure it now: [link].",
        "Your credit card has been charged $500. Click here to dispute: [link].",
        "Your social security number has been compromised. Verify it here: [link].",
        "Your email account will be deleted. Click here to stop it: [link]."
    ]
}

# Test the model with sample texts
for category, texts in sample_texts.items():
    print(f"\nTesting {category} Messages:")
    
    predictions, probabilities = predict_text(texts)  # Process batch at once

    for i, text in enumerate(texts):
        pred_class = predictions[i]
        confidence = probabilities[i][pred_class] * 100  # Get confidence for predicted class

        print(f"Text: {text}")
        print(f"Prediction: {pred_class} ({'Ham (Legit)' if pred_class == 0 else 'Spam' if pred_class == 1 else 'Phishing'})")
        print(f"Confidence: {confidence:.2f}%")
        print("-" * 50)
