# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = "https://huggingface.co/datasets/Shramik121/superkart/SuperKart.csv"
tourism_df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# ----------------------------
# Define the target variable
# ----------------------------
target = 'ProdTaken'   # 1 if the customer purchased the package, else 0

# ----------------------------
# List of numerical features
# ----------------------------
numeric_features = [
    'Product_Weight',                     # Product's weight
    'Product_Allocated_Area',                # Product Allocated Area
    'Product_MRP',         # 
    'Store_Establishment_Year',  # 
    'Product_Store_Sales_Total'       # 
]

# ----------------------------
# List of categorical features
# ----------------------------
categorical_features = [
    'Product_Id',   # 
    'Product_Sugar_Content',      # 
    'Product_Type',          # 
    'Store_Id',  # 
    'Store_Size',   # 
    'Store_Location_City_Type',      # 
    'Store_Type' 
 ]


# ----------------------------
# Combine features to form X (feature matrix)
# ----------------------------
X = tourism_df[numeric_features + categorical_features]

# ----------------------------
# Define target vector y
# ----------------------------
y = tourism_df[target]

# ----------------------------
# Split dataset into training and test sets
# ----------------------------
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

Xtrain.to_csv("Xtrain.csv",index=False)
Xtest.to_csv("Xtest.csv",index=False)
ytrain.to_csv("ytrain.csv",index=False)
ytest.to_csv("ytest.csv",index=False)


files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id="Shramik121/superkart-dataset",
        repo_type="dataset",
    )
