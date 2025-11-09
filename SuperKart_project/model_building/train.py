import pandas as pd
import numpy as np
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error  # ← SPACE ADDED
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import xgboost as xgb
import joblib
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
import mlflow
import os

# MLflow setup
mlflow.set_experiment("SuperKart-Prediction-Experiment")

# Hugging Face API
api = HfApi(token=os.getenv("HF_TOKEN"))

# Dataset paths
Xtrain_path = "hf://datasets/Shramik121/superkart/Xtrain.csv"
Xtest_path = "hf://datasets/Shramik121/superkart/Xtest.csv"
ytrain_path = "hf://datasets/Shramik121/superkart/ytrain.csv"
ytest_path = "hf://datasets/Shramik121/superkart/ytest.csv"

# Load data
X train = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path).iloc[:, 0]
ytest = pd.read_csv(ytest_path).iloc[:, 0]

# Features
numeric_features = ['Product_Weight', 'Product_Allocated_Area', 'Product_MRP',
                    'Store_Establishment_Year', 'Product_Store_Sales_Total']
categorical_features = ['Product_Id', 'Product_Sugar_Content', 'Product_Type',
                        'Store_Id', 'Store_Size', 'Store_Location_City_Type', 'Store_Type']

# Preprocessor
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown='ignore'), categorical_features)
)

# Model
xgb_model = xgb.XGBRegressor(random_state=42, objective='reg:squarederror')
model_pipeline = make_pipeline(preprocessor, xgb_model)

# Hyperparameter tuning
param_grid = {
    'xgbregressor__n_estimators': [100, 200],
    'xgbregressor__max_depth': [3, 5],
    'xgbregressor__learning_rate': [0.05, 0.1],
    'xgbregressor__subsample': [0.8, 1.0],
}

# Train with MLflow
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

    # Log metrics
    mlflow.log_metrics({
        "train_rmse": train_rmse,
        "test_rmse": test_rmse,
        "train_mae": train_mae,
        "test_mae": test_mae,
        "r2_score": r2
    })

    # Save and log model
    model_path = "best_superkart_sales_model_v1.joblib"
    joblib.dump(best_model, model_path)
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved: {model_path}")

    # Upload to Hugging Face
    repo_id = "Shramik121/Superkart"
    repo_type = "model"

    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        print(f"Repo {repo_id} exists.")
    except RepositoryNotFoundError:
        create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
        print(f"Created repo: {repo_id}")

    # FIXED: Use same filename
    api.upload_file(
        path_or_fileobj=model_path,  # ← Use the saved file
        path_in_repo="best_superkart_sales_model_v1.joblib",
        repo_id=repo_id,
        repo_type=repo_type,
    )
    print("Model uploaded to Hugging Face!")
