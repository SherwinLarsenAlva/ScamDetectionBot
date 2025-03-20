import os
import time
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

# Define your dataset directory and repository details
dataset_dir = "Data"  # Directory containing your datasets
repo_id = "SparkyPilot/scam-detection-data"  # Your Hugging Face dataset repository

# Get the repository information
repo_info = api.repo_info(repo_id, repo_type="dataset", token=token)

# Get the list of files in the Hugging Face repository
hf_files = {file.rfilename for file in repo_info.siblings}  # List of files in the Hugging Face repository

# Get the list of files in the local Data/ folder
local_files = set(os.listdir(dataset_dir))

# Find missing files in Hugging Face
missing_files = local_files - hf_files
if missing_files:
    print(f"Found {len(missing_files)} missing dataset(s) in Hugging Face. Uploading...")
    for file_name in missing_files:
        file_path = os.path.join(dataset_dir, file_name)
        print(f"Uploading: {file_path}")
        api.upload_file(
            path_or_fileobj=file_path,
            path_in_repo=file_name,
            repo_id=repo_id,
            repo_type="dataset",  # Specify repo_type as "dataset"
            token=token,
        )
        print(f"Uploaded: {file_path} -> {file_name}")
else:
    print("All datasets are up-to-date in Hugging Face.")

# Track uploaded files to avoid duplicates
uploaded_files = local_files

def upload_new_datasets():
    """Upload new datasets to Hugging Face."""
    global uploaded_files
    current_files = set(os.listdir(dataset_dir))

    # Debug: Print current files
    print("Current files in Data/: ", current_files)

    # Find new files
    new_files = current_files - uploaded_files
    if new_files:
        print(f"Found {len(new_files)} new dataset(s). Uploading...")
        for file_name in new_files:
            file_path = os.path.join(dataset_dir, file_name)
            print(f"Uploading: {file_path}")
            api.upload_file(
                path_or_fileobj=file_path,
                path_in_repo=file_name,
                repo_id=repo_id,
                repo_type="dataset",  # Specify repo_type as "dataset"
                token=token,
            )
            print(f"Uploaded: {file_path} -> {file_name}")
        uploaded_files = current_files  # Update the set of uploaded files
    else:
        print("No new datasets found.")

# Upload all existing files initially
upload_new_datasets()

# Monitor the Data/ folder for new files
if __name__ == "__main__":
    print("Monitoring the Data/ folder for new datasets...")
    while True:
        upload_new_datasets()
        time.sleep(60)  # Check for new files every 60 seconds