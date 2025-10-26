import os
import json
from flask import Flask, request, jsonify, render_template_string
import joblib
import numpy as np

app = Flask(__name__)

# Load model artifacts (assume .pkl files are in parent dir)
model = joblib.load('../diabetes_model.pkl')
scaler = joblib.load('../scaler.pkl')
le_gender = joblib.load('../le_gender.pkl')
means = joblib.load('../column_means.pkl')

# Simple HTML template for the form
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Diabetes Predictor</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 600px; margin: 50px auto; padding: 20px; }
        input, select, button { width: 100%; padding: 10px; margin: 10px 0; }
        button { background: #007bff; color: white; border: none; cursor: pointer; }
        #result { margin-top: 20px; padding: 15px; border-radius: 5px; }
        .yes { background: #dc3545; color: white; }
        .no { background: #28a745; color: white; }
    </style>
</head>
<body>
    <h1>Diabetes Risk Prediction</h1>
    <form id="predictForm">
        <label>Age:</label>
        <input type="number" id="age" min="1" max="120" value="30" required>
        
        <label>Gender:</label>
        <select id="gender" required>
            {% for opt in gender_options %}
            <option value="{{ opt }}">{{ opt }}</option>
            {% endfor %}
        </select>
        
        <label>HbA1c (Sugar Level):</label>
        <input type="number" id="hba1c" min="0" max="20" value="5.0" step="0.1" required>
        
        <button type="submit">Predict Diabetes</button>
    </form>
    <div id="result"></div>

    <script>
        const form = document.getElementById('predictForm');
        const resultDiv = document.getElementById('result');
        
        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            const age = document.getElementById('age').value;
            const gender = document.getElementById('gender').value;
            const hba1c = document.getElementById('hba1c').value;
            
            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ age: parseInt(age), gender, hba1c: parseFloat(hba1c) })
            });
            
            const data = await response.json();
            resultDiv.innerHTML = `<div class="result ${data.result.toLowerCase()}">${data.message}</div>`;
        });
    </script>
</body>
</html>
"""

@app.route('/', methods=['GET'])
def home():
    gender_options = [str(g).strip().upper() for g in le_gender.classes_]
    return render_template_string(HTML_TEMPLATE, gender_options=gender_options)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        age = data['age']
        gender = data['gender']
        hba1c = data['hba1c']

        # Encode gender
        g_enc = le_gender.transform([gender])[0]

        # Build input row (same order as training: Gender, AGE, Urea, Cr, HbA1c, Chol, TG, HDL, LDL, VLDL, BMI)
        row = np.array([[
            g_enc, age,
            means['Urea'], means['Cr'],
            hba1c,
            means['Chol'], means['TG'],
            means['HDL'], means['LDL'],
            means['VLDL'], means['BMI']
        ]])

        # Scale and predict
        row_scaled = scaler.transform(row)
        pred = model.predict(row_scaled)[0]
        result = 'Yes' if pred == 1 else 'No'

        message = f"Prediction: Diabetes - {result}"
        if result == 'Yes':
            message += " | Consult a doctor immediately."

        return jsonify({'result': result, 'message': message})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Vercel serverless handler
def handler(request):
    # Simulate Flask's request for Vercel
    from flask import Request
    req = Request.from_flask(request)
    with app.test_request_context(request):
        return app.full_dispatch_request()

if __name__ == '__main__':
    app.run(debug=True)