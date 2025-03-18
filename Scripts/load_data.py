import pandas as pd
from datasets import load_dataset

# Load Kaggle SMS Spam dataset [https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset]
df_sms = pd.read_csv("D:\PROJECTS\AI\spam.csv", encoding="latin-1")
df_sms = df_sms[['v1', 'v2']]  # Keep only necessary columns [This data set has 4 columns]
df_sms.columns = ['label', 'text']
df_sms['label'] = df_sms['label'].map({'ham': 0, 'spam': 1})  # Convert labels [ham means legitimate and spam means spam]

# Load SMS Phishing dataset [https://data.mendeley.com/datasets/f45bkkt8pr/1]
df_smishing = pd.read_csv("D:\PROJECTS\AI\Dataset_5971.csv") 
df_smishing = df_smishing[['LABEL', 'TEXT']]  # Keep only necessary columns [This data set has 5 columns] 
df_smishing.columns = ['text', 'label']
df_smishing['label'] = df_smishing['label'].map({'ham': 0, 'spam': 1, 'Smishing': 2})  # Convert labels [ham means legitimate, spam means spam, and smishing means SMS phishing]

# Load Hugging Face Phishing dataset [https://huggingface.co/datasets/ealvaradob/phishing-dataset]
df_phishing = pd.read_csv("D:\PROJECTS\AI\Phishing_Email.csv") # Because from this we are using only phishing email and sms phishing dataset [sms is already called previously]
df_phishing = df_phishing[['Email Text', 'Email Type']] # Keep only necessary columns [This data set has 3 columns]
df_phishing.columns = ['text', 'label']
df_phishing['label'] = df_phishing['label'].map({'Safe Mail': 0, 'Phishing Email': 2})  # Convert labels [ham means legitimate and spam means spam]

# Merge all datasets
df = pd.concat([df_sms, df_smishing, df_phishing], ignore_index=True)

# Remove duplicates and empty rows
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

# Save cleaned dataset
df.to_csv("merged_scam_dataset.csv", index=False)
print("Dataset loaded and saved successfully!")
