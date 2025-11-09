from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
import os

# Environment variable must be set in GitHub Actions secrets
token = os.getenv("HF_TOKEN")

if not token:
    raise ValueError("Missing Hugging Face token. Please set HF_TOKEN in GitHub Actions secrets.")

repo_id = "Shramik121/Superkart-dataset"
repo_type = "dataset"
data_folder = "SuperKart_project/data"

# Initialize API client
api = HfApi(token=token)

# Step 1: Check if the dataset repo exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f" Dataset '{repo_id}' already exists. Uploading new data...")
except RepositoryNotFoundError:
    print(f" Dataset '{repo_id}' not found. Creating it...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False, token=token)
    print(f" Dataset '{repo_id}' created successfully.")

# Step 2: Upload your dataset folder
api.upload_folder(
    folder_path=data_folder,
    repo_id=repo_id,
    repo_type=repo_type,
)
print(f"Uploaded data from {data_folder} to Hugging Face Hub.")
