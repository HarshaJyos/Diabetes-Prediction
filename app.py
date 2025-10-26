# app.py
import streamlit as st
import joblib
import numpy as np
import os

# -------------------------------
# 1. Page Config
# -------------------------------
st.set_page_config(page_title="Diabetes Predictor", page_icon="Blood Drop", layout="centered")
st.title("Diabetes Risk Prediction")
st.markdown("""
**Enter your clinical values below.**  
The model uses **all 11 features** (6 from you, 5 imputed from dataset averages).
""")

# -------------------------------
# 2. Load Model & Artifacts
# -------------------------------
@st.cache_resource
def load_artifacts():
    required = ['diabetes_model.pkl', 'scaler.pkl', 'le_gender.pkl', 'column_means.pkl']
    missing = [f for f in required if not os.path.exists(f)]
    if missing:
        st.error(f"Missing files: {', '.join(missing)}\n\nUpload all `.pkl` files.")
        st.stop()
    
    try:
        model = joblib.load('diabetes_model.pkl')
        scaler = joblib.load('scaler.pkl')
        le_gender = joblib.load('le_gender.pkl')
        means = joblib.load('column_means.pkl')
        return model, scaler, le_gender, means
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

model, scaler, le_gender, means = load_artifacts()

# -------------------------------
# 3. Input Form (6 fields)
# -------------------------------
with st.form("input_form", clear_on_submit=False):
    st.subheader("Patient Details")
    
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=30, step=1)
        gender = st.selectbox("Gender", options=[str(g).strip().upper() for g in le_gender.classes_])
        bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
    
    with col2:
        hba1c = st.number_input("HbA1c (%)", min_value=0.0, max_value=20.0, value=5.0, step=0.1)
        chol = st.number_input("Cholesterol (mg/dL)", min_value=0.0, max_value=600.0, value=180.0, step=1.0)
        tg = st.number_input("Triglycerides (TG)", min_value=0.0, max_value=2000.0, value=150.0, step=1.0)
    
    submitted = st.form_submit_button("Predict Diabetes Risk", type="primary", use_container_width=True)

# -------------------------------
# 4. Prediction with Spinner
# -------------------------------
if submitted:
    with st.spinner("Analyzing clinical data..."):
        try:
            # Encode gender
            g_enc = le_gender.transform([gender])[0]
            
            # Build feature vector (same order as training)
            row = np.array([[
                g_enc,           # Gender
                age,             # Age
                means.get('Urea', 0),
                means.get('Cr', 0),
                hba1c,           # HbA1c
                chol,            # Chol (user input)
                tg,              # TG (user input)
                means.get('HDL', 0),
                means.get('LDL', 0),
                means.get('VLDL', 0),
                bmi              # BMI (user input)
            ]], dtype=float)
            
            # Scale
            row_scaled = scaler.transform(row)
            
            # Predict
            pred = model.predict(row_scaled)[0]
            prob = model.predict_proba(row_scaled)[0]
            confidence = max(prob) * 100
            result = "Yes" if pred == 1 else "No"
            
            # Display Result
            st.markdown("---")
            if result == "Yes":
                st.error(f"**High Risk: Diabetes = {result}**")
                st.warning(f"**Confidence: {confidence:.1f}%** – Urgent medical consultation recommended.")
            else:
                st.success(f"**Low Risk: Diabetes = {result}**")
                st.info(f"**Confidence: {confidence:.1f}%** – Continue healthy lifestyle.")
                
            # Show input summary
            with st.expander("View Input Summary"):
                st.json({
                    "Age": age,
                    "Gender": gender,
                    "HbA1c": hba1c,
                    "BMI": bmi,
                    "Cholesterol": chol,
                    "Triglycerides": tg
                })
                
        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.info("Please check your inputs.")

# -------------------------------
# 5. Footer
# -------------------------------
st.markdown("---")
st.caption("Model: XGBoost (99% accuracy) | Uses 11 clinical features | Built with Streamlit")