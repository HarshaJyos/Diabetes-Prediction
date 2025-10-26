# app.py
import streamlit as st
import joblib
import numpy as np
import os

# -------------------------------
# 1. Verify all .pkl files exist
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
    st.error(f"Missing files: {', '.join(missing)}\n\n"
             "Place all 5 `.pkl` files in the same folder as `app.py`.")
    st.stop()

# -------------------------------
# 2. Load model & artifacts
# -------------------------------
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load('diabetes_model.pkl')
        scaler = joblib.load('scaler.pkl')
        le_gender = joblib.load('le_gender.pkl')
        le_class = joblib.load('le_class.pkl')
        means = joblib.load('column_means.pkl')
        return model, scaler, le_gender, le_class, means
    except Exception as e:
        st.error(f"Failed to load model files: {e}")
        st.stop()

model, scaler, le_gender, le_class, means = load_artifacts()

# -------------------------------
# 3. UI
# -------------------------------
st.set_page_config(page_title="Diabetes Predictor", page_icon="Heartbeat")
st.title("Diabetes Risk Prediction")
st.markdown("Enter patient details below to predict diabetes.")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=30, step=1)

with col2:
    gender_options = [str(g).strip().upper() for g in le_gender.classes_]
    gender = st.selectbox("Gender", options=gender_options)

hba1c = st.number_input("HbA1c (Sugar Level)", min_value=0.0, max_value=20.0, value=5.0, step=0.1, format="%.1f")

if st.button("Predict Diabetes", type="primary"):
    with st.spinner("Predicting..."):
        try:
            # Encode gender
            g_enc = le_gender.transform([gender])[0]

            # Build full feature vector
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
            raw_label = le_class.inverse_transform([pred])[0].strip()
            result = "Yes" if raw_label in ('Y', 'Yes', 'P') else "No"

            # Display result
            if result == "Yes":
                st.error(f"**Prediction: Diabetes - {result}**")
                st.warning("High risk detected. Please consult a doctor.")
            else:
                st.success(f"**Prediction: Diabetes - {result}**")
                st.info("Low risk. Continue healthy habits.")

        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.info("Check inputs and ensure all model files are present.")

# -------------------------------
# 4. Footer
# -------------------------------
st.markdown("---")
st.caption("Trained on clinical dataset | Built with Streamlit | scikit-learn required")