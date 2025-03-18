import os
from dotenv import load_dotenv
from huggingface_hub import HfApi

# Get the token from the .env file
token = os.getenv("HUGGINGFACE_TOKEN")
if not token:
    raise ValueError("Hugging Face token not found in .env file.")

# Initialize the API
api = HfApi()

# Define your dataset directory and repository details
dataset_dir = "Data"  # Directory containing your datasets
repo_id = "SherwinLarsenAlva/scam-detection-data"  # Your Hugging Face dataset repository

# Upload all files in the dataset directory
for file_name in os.listdir(dataset_dir):
    file_path = os.path.join(dataset_dir, file_name)
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_name,
        repo_id=repo_id,
        token=token,
    )
    print(f"Uploaded: {file_path} -> {file_name}")

print("Datasets uploaded to Hugging Face successfully!")