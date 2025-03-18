import os
from dotenv import load_dotenv
from huggingface_hub import HfApi

# Load the .env file
load_dotenv()

# Get the token from the .env file
token = os.getenv("HUGGINGFACE_TOKEN")
if not token:
    raise ValueError("Hugging Face token not found in .env file.")

# Initialize the API
api = HfApi()

# Define your model directory and repository details
model_dir = "Models"  # Directory containing your pre-trained models
repo_id = "SparkyPilot/ScamDetector"  # Your Hugging Face repository

# Function to upload files recursively
def upload_files(directory, repo_id, token):
    for root, _, files in os.walk(directory):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            relative_path = os.path.relpath(file_path, directory)  # Get relative path
            api.upload_file(
                path_or_fileobj=file_path,
                path_in_repo=relative_path,  # Use relative path in the repo
                repo_id=repo_id,
                token=token,
            )
            print(f"Uploaded: {file_path} -> {relative_path}")

# Upload all files in the model directory
upload_files(model_dir, repo_id, token)

print("Models uploaded successfully!")