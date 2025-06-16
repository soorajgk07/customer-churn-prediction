import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Churn Prediction", layout="wide")
st.title("🧠 Churn Prediction App")

# Load model and columns
try:
    model = joblib.load("churn_model.pkl")
    model_columns = joblib.load("model_columns.pkl")
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()

# Initialize session state
if "inputs" not in st.session_state:
    st.session_state.inputs = {col: 0.0 for col in model_columns}

# Sample 1 (Churn = 1)
sample_1 = {
    "Tenure": 4.0, "PreferredLoginDevice": 0.0, "WarehouseToHome": 6.0, "HourSpendOnApp": 3.0,
    "CityTier": 3.0, "Gender": 1.0, "NumberOfDeviceRegistered": 3.0, "SatisfactionScore": 2.0,
    "NumberOfAddress": 9.0, "Complain": 1.0, "OrderAmountHikeFromlastYear": 11.0, "CouponUsed": 1.0,
    "OrderCount": 1.0, "DaySinceLastOrder": 5.0, "CashbackAmount": 160.0,
    "PreferredPaymentMode_COD": 0.0, "PreferredPaymentMode_Credit Card": 0.0,
    "PreferredPaymentMode_Debit Card": 1.0, "PreferredPaymentMode_E wallet": 0.0,
    "PreferredPaymentMode_UPI": 0.0,
    "PreferedOrderCat_Fashion": 0.0, "PreferedOrderCat_Grocery": 0.0,
    "PreferedOrderCat_Laptop & Accessory": 1.0, "PreferedOrderCat_Mobile Phone": 0.0,
    "PreferedOrderCat_Others": 0.0,
    "MaritalStatus_Divorced": 0.0, "MaritalStatus_Married": 0.0, "MaritalStatus_Single": 1.0
}

# Sample 2 (Churn = 0)
sample_2 = {
    "Tenure": 8.0, "PreferredLoginDevice": 0.0, "WarehouseToHome": 6.0, "HourSpendOnApp": 3.0,
    "CityTier": 3.0, "Gender": 0.0, "NumberOfDeviceRegistered": 3.0, "SatisfactionScore": 4.0,
    "NumberOfAddress": 2.0, "Complain": 0.0, "OrderAmountHikeFromlastYear": 13.0, "CouponUsed": 1.0,
    "OrderCount": 1.0, "DaySinceLastOrder": 6.0, "CashbackAmount": 173.0,
    "PreferredPaymentMode_COD": 0.0, "PreferredPaymentMode_Credit Card": 0.0,
    "PreferredPaymentMode_Debit Card": 0.0, "PreferredPaymentMode_E wallet": 1.0,
    "PreferredPaymentMode_UPI": 0.0,
    "PreferedOrderCat_Fashion": 1.0, "PreferedOrderCat_Grocery": 0.0,
    "PreferedOrderCat_Laptop & Accessory": 0.0, "PreferedOrderCat_Mobile Phone": 0.0,
    "PreferedOrderCat_Others": 0.0,
    "MaritalStatus_Divorced": 1.0, "MaritalStatus_Married": 0.0, "MaritalStatus_Single": 0.0
}

# Auto-fill buttons
col1, col2, col3, col4 = st.columns(4)
if col1.button("Fill Sample 1"):
    st.session_state.inputs.update(sample_1)
if col2.button("Fill Sample 2"):
    st.session_state.inputs.update(sample_2)
if col3.button("Reset"):
    st.session_state.inputs = {col: 0.0 for col in model_columns}
if col4.button("Predict"):
    input_df = pd.DataFrame([st.session_state.inputs])
    prediction = model.predict(input_df)[0]
    label = "Churn" if prediction == 1 else "Not Churned"
    st.success(f"📢 Prediction: {label}")

st.subheader("Enter Feature Values Below:")
# Render input boxes
for col in model_columns:
    st.session_state.inputs[col] = st.number_input(
        label=col,
        value=float(st.session_state.inputs.get(col, 0.0)),
        key=col
    )
