from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
import os

# Initialize API
api = HfApi(token=os.getenv("HF_TOKEN"))

# Space details
repo_id = "Shramik121/Superkart"
repo_type = "space"

# Step 1: Check and create Space if needed
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f" Space '{repo_id}' already exists.")
except RepositoryNotFoundError:
    print(f" Creating Space '{repo_id}'...")
    create_repo(
        repo_id=repo_id,
        repo_type=repo_type,
        space_sdk="streamlit",  # For your Streamlit app.py
        private=False,          # Public; set True for private
        exist_ok=False
    )
    print(f" Created! Wait 1-2 min for init: https://huggingface.co/spaces/{repo_id}")
except Exception as e:
    print(f" Creation error: {e}")
    raise

# Step 2: Upload files (now safe)
api.upload_folder(
    folder_path="SuperKart_project/deployment",
    repo_id=repo_id,
    repo_type=repo_type,
    path_in_repo="",  # Root of Space
    commit_message="Deploy SuperKart Sales Prediction App"
)

print(" Upload complete! App will build in 2-5 min.")
print(f"Visit: https://huggingface.co/spaces/{repo_id}")
