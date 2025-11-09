import pandas as pd
import numpy as np
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib
from huggingface_hub import login, HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
import mlflow
import os

#mlflow.set_tracking_uri("http://localhost:8080")
mlflow.set_experiment("SuperKart-Prediction-Experiment")

# Hugging Face API authentication
api = HfApi(token=os.getenv("HF_TOKEN"))
Xtrain_path = "hf://datasets/Shramik121/superkart/Xtrain.csv"
Xtest_path = "hf://datasets/Shramik121/superkart/Xtest.csv"
ytrain_path = "hf://datasets/Shramik121/superkart/ytrain.csv"
ytest_path = "hf://datasets/Shramik121/superkart/ytest.csv"

# Load datasets
Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path)
ytest = pd.read_csv(ytest_path)

# If y is stored as DataFrame with one column, convert to Series
if isinstance(ytrain, pd.DataFrame):
    ytrain = ytrain.iloc[:, 0]
if isinstance(ytest, pd.DataFrame):
    ytest = ytest.iloc[:, 0]
# Feature definitions
numeric_features = [
    'Product_Weight',                     # Product's weight
    'Product_Allocated_Area',                # Product Allocated Area
    'Product_MRP',         # 
    'Store_Establishment_Year',  # 
    'Product_Store_Sales_Total'
]
categorical_features = [
    'Product_Id',   # 
    'Product_Sugar_Content',      # 
    'Product_Type',          # 
    'Store_Id',  # 
    'Store_Size',   # 
    'Store_Location_City_Type',      # 
    'Store_Type'
]

# Preprocessing
# --------------------------
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown='ignore'), categorical_features)
)
# Model definition
# --------------------------
xgb_model = xgb.XGBRegressor(random_state=42, objective='reg:squarederror')

# Pipeline
# ----------------------------
model_pipeline = make_pipeline(preprocessor, xgb_model)

# Hyperparameter grid
# --------------------------
param_grid = {
    'xgbregressor__n_estimators': [100, 200],
    'xgbregressor__max_depth': [3, 5],
    'xgbregressor__learning_rate': [0.05, 0.1],
    'xgbregressor__subsample': [0.8, 1.0],
}


# Train and log with MLflow
# --------------------------
with mlflow.start_run():
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=3, n_jobs=-1)
    grid_search.fit(Xtrain, ytrain)

    best_model = grid_search.best_estimator_

    # Predictions
    y_pred_train = best_model.predict(Xtrain)
    y_pred_test = best_model.predict(Xtest)

    # Metrics
    train_rmse = np.sqrt(mean_squared_error(ytrain, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(ytest, y_pred_test))
    train_mae = mean_absolute_error(ytrain, y_pred_train)
    test_mae = mean_absolute_error(ytest, y_pred_test)
    r2 = r2_score(ytest, y_pred_test)

    # Log metrics to MLflow
    mlflow.log_metrics({
        "train_rmse": train_rmse,
        "test_rmse": test_rmse,
        "train_mae": train_mae,
        "test_mae": test_mae,
        "r2_score": r2
    })

    # Save model
    model_path = "best_superkart_sales_model_v1.joblib"
    joblib.dump(best_model, model_path)
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved as artifact at: {model_path}")

  
  

    # Upload to Hugging Face
    repo_id = "Shramik121/superkart"
    repo_type = "model"

    # Step 1: Check if the space exists
    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        print(f"Space '{repo_id}' already exists. Using it.")
    except RepositoryNotFoundError:
        print(f"Space '{repo_id}' not found. Creating new space...")
        create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
        print(f"Space '{repo_id}' created.")

    # create_repo("churn-model", repo_type="model", private=False)
    api.upload_file(
        path_or_fileobj="best_superkart_model_v1.joblib",
        path_in_repo="best_superkart_model_v1.joblib",
        repo_id=repo_id,
        repo_type=repo_type,
    )
    print("✅ Model uploaded to Hugging Face Hub successfully!")
