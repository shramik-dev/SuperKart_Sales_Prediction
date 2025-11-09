import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download the model from the Model Hub
model_path = hf_hub_download(repo_id="Shramik121/superkart-model", filename="best_superkart_model_v1.joblib")

# Load the model
model = joblib.load(model_path)

# Streamlit UI for Customer Churn Prediction
st.title("Superkart Prediction")
st.write("
Predict future **product sales** based on product features, store type, and location details.
An accurate sales forecast helps optimize procurement, plan logistics, and support decision-making across departments.
")

# Collect user input
Product_Id = st.text_input("Product ID", value="FD6114")
Product_Weight = st.number_input("Product Weight (kg)", min_value=0.0, value=12.0)
Product_Sugar_Content = st.selectbox("Product Sugar Content", ["Low Sugar", "Regular", "No Sugar"])
Product_Allocated_Area = st.number_input("Allocated Shelf Area (in sq.m)", min_value=0.0, value=0.05)
Product_Type = st.selectbox("Product Type",["Dairy", "Frozen Foods", "Baking Goods", "Canned", "Health and Hygiene", "Meat", "Snack Foods", "Soft Drinks"])
Product_MRP = st.number_input("Product MRP (₹)", min_value=10.0, value=150.0)
Store_Id = st.text_input("Store ID", value="OUT001")
Store_Establishment_Year = st.number_input("Store Establishment Year", min_value=1980, max_value=2025, value=2000)
Store_Size = st.selectbox("Store Size", ["Small", "Medium", "High"])
Store_Location_City_Type = st.selectbox("Store Location City Type", ["Tier 1", "Tier 2", "Tier 3"])
Store_Type = st.selectbox("Store Type", ["Supermarket Type1", "Supermarket Type2", "Departmental Store", "Food Mart"])

# ----------------------------
# Prepare input data
# ----------------------------
input_data = pd.DataFrame([{
    'Product_Id': Product_Id,
    'Product_Weight': Product_Weight,
    'Product_Sugar_Content': Product_Sugar_Content,
    'Product_Allocated_Area': Product_Allocated_Area,
    'Product_Type': Product_Type,
    'Product_MRP': Product_MRP,
    'Store_Id': Store_Id,
    'Store_Establishment_Year': Store_Establishment_Year,
    'Store_Size': Store_Size,
    'Store_Location_City_Type': Store_Location_City_Type,
    'Store_Type': Store_Type
}])

# Set the classification threshold
classification_threshold = 0.45

# Predict button
if st.button("Predict"):
    prob = model.predict_proba(input_data)[0,1]
    pred = int(prob >= classification_threshold)
    result = "Prediction sales" if pred == 1 else "prediction failed"
    st.write(f"Prediction: Sales {result}")
