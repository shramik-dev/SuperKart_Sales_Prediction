import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download model
model_path = hf_hub_download(
    repo_id="Shramik121/superkart",
    filename="best_superkart_sales_model_v1.joblib"  # Ensure this matches train.py save
)
model = joblib.load(model_path)

st.title("SuperKart Sales Prediction")
st.write("Predict product sales based on features and store details.")

# Inputs (as before)
Product_Id = st.text_input("Product ID", "FD6114")
Product_Weight = st.number_input("Product Weight (kg)", min_value=0.0, value=12.0)
Product_Sugar_Content = st.selectbox("Sugar Content", ["Low Sugar", "Regular", "No Sugar"])
Product_Allocated_Area = st.number_input("Allocated Area (sq.m)", min_value=0.0, value=0.05)
Product_Type = st.selectbox("Product Type", [
    "Dairy", "Frozen Foods", "Baking Goods", "Canned",
    "Health and Hygiene", "Meat", "Snack Foods", "Soft Drinks"
])
Product_MRP = st.number_input("MRP (₹)", min_value=10.0, value=150.0)
Store_Id = st.text_input("Store ID", "OUT001")
Store_Establishment_Year = st.number_input("Establishment Year", 1980, 2025, 2000)
Store_Size = st.selectbox("Store Size", ["Small", "Medium", "High"])
Store_Location_City_Type = st.selectbox("City Type", ["Tier 1", "Tier 2", "Tier 3"])
Store_Type = st.selectbox("Store Type", [
    "Supermarket Type1", "Supermarket Type2", "Departmental Store", "Food Mart"
])

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

if st.button("Predict Sales"):
    pred = model.predict(input_data)[0]
    st.success(f"**Predicted Sales: ₹{pred:,.2f}**")
