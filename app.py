# app.py
import streamlit as st
import joblib
import numpy as np
import os
import pandas as pd

# -------------------------------
# 1. Title & Description
# -------------------------------
st.set_page_config(page_title="Diabetes Predictor", page_icon="🩸")
st.title("Diabetes Risk Prediction (XGBoost)")
st.markdown("Enter **Age**, **Gender**, and **HbA1c** to predict diabetes risk using all clinical features.")

# -------------------------------
# 2. Load Model Files (with error handling)
# -------------------------------
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load('diabetes_model.pkl')
        scaler = joblib.load('scaler.pkl')
        le_gender = joblib.load('le_gender.pkl')
        means = joblib.load('column_means.pkl')
        return model, scaler, le_gender, means
    except FileNotFoundError as e:
        st.error(f"Model file not found: {e.filename}. Upload all .pkl files.")
        st.stop()
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        st.stop()

model, scaler, le_gender, means = load_artifacts()

# -------------------------------
# 3. Input Form
# -------------------------------
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=30, step=1)
    
    with col2:
        gender_options = [str(g).strip().upper() for g in le_gender.classes_]
        gender = st.selectbox("Gender", options=gender_options)
    
    hba1c = st.number_input("HbA1c (Sugar Level)", min_value=0.0, max_value=20.0, value=5.0, step=0.1)
    
    submitted = st.form_submit_button("Predict Diabetes")

# -------------------------------
# 4. Prediction with Spinner
# -------------------------------
if submitted:
    with st.spinner("Predicting..."):
        try:
            # Encode gender
            g_enc = le_gender.transform([gender])[0]
            
            # Build full feature vector (same order as training)
            row = np.array([[
                g_enc, age,
                means.get('Urea', 0), means.get('Cr', 0),
                hba1c,
                means.get('Chol', 0), means.get('TG', 0),
                means.get('HDL', 0), means.get('LDL', 0),
                means.get('VLDL', 0), means.get('BMI', 0)
            ]], dtype=float)
            
            # Scale
            row_scaled = scaler.transform(row)
            
            # Predict
            pred = model.predict(row_scaled)[0]
            result = "Yes" if pred == 1 else "No"
            
            # Display result
            if result == "Yes":
                st.error(f"**Prediction: Diabetes - {result}**")
                st.warning("High risk detected. Consult a doctor.")
            else:
                st.success(f"**Prediction: Diabetes - {result}**")
                st.info("Low risk. Continue monitoring.")
                
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
            st.info("Check inputs and ensure model files are uploaded.")

# -------------------------------
# 5. Footer
# -------------------------------
st.markdown("---")
st.caption("Model: XGBoost (uses all features) | Trained on clinical dataset | Built with Streamlit")