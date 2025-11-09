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
api = HfApi(token=os.getenv("HF_TOKEN"))

# Step 1: Check if the space exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Space '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Space '{repo_id}' not found. Creating new space...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Space '{repo_id}' created.")

api.upload_folder(
    folder_path="SuperKart_project/data",
    repo_id=repo_id,
    repo_type=repo_type,
)
