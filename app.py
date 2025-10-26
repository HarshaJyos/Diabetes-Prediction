# app.py
import streamlit as st
import joblib
import numpy as np
import os

# -------------------------------
# 1. Check if .pkl files exist
# -------------------------------
required_files = [
    'diabetes_model.pkl',
    'scaler.pkl',
    'le_gender.pkl',
    'le_class.pkl',
    'column_means.pkl'
]

missing = [f for f in required_files if not os.path.exists(f)]
if missing:
    st.error(f"Missing files: {', '.join(missing)}. Please place them in the same folder as app.py")
    st.stop()

# -------------------------------
# 2. Load model & preprocessors
# -------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load('diabetes_model.pkl')
    scaler = joblib.load('scaler.pkl')
    le_gender = joblib.load('le_gender.pkl')
    le_class = joblib.load('le_class.pkl')
    means = joblib.load('column_means.pkl')
    return model, scaler, le_gender, le_class, means

model, scaler, le_gender, le_class, means = load_artifacts()

# -------------------------------
# 3. Streamlit UI
# -------------------------------
st.set_page_config(page_title="Diabetes Predictor", page_icon="Health")
st.title("Diabetes Prediction")
st.markdown("Enter patient details to predict diabetes risk.")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=30, step=1)

with col2:
    gender_options = [str(g).strip().upper() for g in le_gender.classes_]
    gender = st.selectbox("Gender", options=gender_options)

hba1c = st.number_input("HbA1c (Sugar Level)", min_value=0.0, max_value=20.0, value=5.0, step=0.1)

if st.button("Predict Diabetes", type="primary"):
    try:
        # Encode gender
        g_enc = le_gender.transform([gender])[0]

        # Build input row (same order as training)
        row = np.array([[
            g_enc, age,
            means['Urea'], means['Cr'],
            hba1c,
            means['Chol'], means['TG'],
            means['HDL'], means['LDL'],
            means['VLDL'], means['BMI']
        ]])

        # Scale
        row_scaled = scaler.transform(row)

        # Predict
        pred = model.predict(row_scaled)[0]
        label = le_class.inverse_transform([pred])[0].strip()

        # Map to Yes/No
        result = "Yes" if label in ('Y', 'Yes', 'P') else "No"  # 'P' sometimes means Prediabetes

        # Show result
        if result == "Yes":
            st.error(f"**Prediction: Diabetes - {result}**")
            st.warning("Consult a doctor immediately.")
        else:
            st.success(f"**Prediction: Diabetes - {result}**")
            st.info("Keep monitoring regularly.")

    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.info("Make sure all input values are valid and .pkl files are not corrupted.")

# -------------------------------
# 4. Footer
# -------------------------------
st.markdown("---")
st.caption("Model trained on clinical dataset | Built with Streamlit")