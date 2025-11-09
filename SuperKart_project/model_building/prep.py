# for data manipulation
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi

# ----------------------------
# Setup
# ----------------------------
api = HfApi(token=os.getenv("HF_TOKEN"))

# Load dataset
Superkart_df = pd.read_csv("SuperKart_project/data/SuperKart.csv")
print("Dataset loaded successfully.")

# ----------------------------
# Define the target variable (Revenue)
# ----------------------------
target = 'Product_Store_Sales_Total'
Superkart_df = Superkart_df.dropna(subset=[target])

# ----------------------------
# Define features
# ----------------------------
numeric_features = [
    'Product_Weight',
    'Product_Allocated_Area',
    'Product_MRP',
    'Store_Establishment_Year'
]

categorical_features = [
    'Product_Id',
    'Product_Sugar_Content',
    'Product_Type',
    'Store_Id',
    'Store_Size',
    'Store_Location_City_Type',
    'Store_Type'
]

# ----------------------------
# Split data into features (X) and target (y)
# ----------------------------
X = Superkart_df[numeric_features + categorical_features]
y = Superkart_df[target]

# ----------------------------
# Split dataset into training and test sets
# ----------------------------
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Data split done: Train({Xtrain.shape}), Test({Xtest.shape})")

# Save locally
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)
print("Train/Test datasets saved locally.")

# ----------------------------
# Upload to Hugging Face dataset repo
# ----------------------------
files = ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path,
        repo_id="Shramik121/Superkart",
        repo_type="dataset",
    )

print("All split files uploaded successfully to Hugging Face.")
