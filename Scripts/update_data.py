import pandas as pd

# Load the original dataset
existing_data = pd.read_csv("merged_scam_dataset.csv")

# Load the new phishing dataset
phishing_data = pd.read_csv("phishing_email1.csv")

# Map labels in the new dataset
phishing_data["label"] = phishing_data["label"].replace({0: 0.0, 1: 2.0})

# Rename the text column (if necessary)
phishing_data.rename(columns={"text_combined": "text"}, inplace=True)

# Merge the datasets
full_data = pd.concat([existing_data, phishing_data], ignore_index=True)

# Save the combined dataset
full_data.to_csv("updated_scam_dataset.csv", index=False)
print("✅ Combined dataset saved as 'updated_scam_dataset.csv'")