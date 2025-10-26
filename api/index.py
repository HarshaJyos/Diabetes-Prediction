# api/predict.py
import os
import json
import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template_string

# --- Load model once at startup ---
try:
    model = joblib.load('../diabetes_model.pkl')
    scaler = joblib.load('../scaler.pkl')
    le_gender = joblib.load('../le_gender.pkl')
    means = joblib.load('../column_means.pkl')
except Exception as e:
    print(f"ERROR loading model: {e}")
    raise

app = Flask(__name__)

# --- HTML UI ---
HTML = """
<!DOCTYPE html>
<html>
<head>
  <title>Diabetes Predictor</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    body { font-family: system-ui; max-width: 500px; margin: 40px auto; padding: 20px; }
    input, select, button { width: 100%; padding: 12px; margin: 8px 0; font-size: 16px; }
    button { background: #0066cc; color: white; border: none; border-radius: 6px; cursor: pointer; }
    button:hover { background: #0055aa; }
    #result { margin-top: 20px; padding: 15px; border-radius: 6px; font-weight: bold; }
    .yes { background: #ffebee; color: #c62828; border: 1px solid #ffcdd2; }
    .no { background: #e8f5e8; color: #2e7d32; border: 1px solid #c8e6c9; }
  </style>
</head>
<body>
  <h1>Diabetes Risk Check</h1>
  <form id="form">
    <label>Age</label>
    <input type="number" id="age" min="1" max="120" value="30" required>
    
    <label>Gender</label>
    <select id="gender" required>
      {% for g in gender_options %}
      <option>{{ g }}</option>
      {% endfor %}
    </select>
    
    <label>HbA1c (%)</label>
    <input type="number" id="hba1c" min="0" max="20" step="0.1" value="5.0" required>
    
    <button type="submit">Predict</button>
  </form>
  <div id="result"></div>

  <script>
    document.getElementById('form').onsubmit = async (e) => {
      e.preventDefault();
      const data = {
        age: +document.getElementById('age').value,
        gender: document.getElementById('gender').value,
        hba1c: +document.getElementById('hba1c').value
      };
      const res = await fetch('/predict', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(data)
      });
      const json = await res.json();
      const div = document.getElementById('result');
      div.className = json.result === 'Yes' ? 'yes' : 'no';
      div.innerText = json.message;
    };
  </script>
</body>
</html>
"""

@app.route("/", methods=["GET"])
def home():
    gender_opts = [str(g).strip().upper() for g in le_gender.classes_]
    return render_template_string(HTML, gender_options=gender_opts)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        age = int(data["age"])
        gender = data["gender"]
        hba1c = float(data["hba1c"])

        # Encode gender
        g_enc = le_gender.transform([gender])[0]

        # Build feature vector
        row = np.array([[
            g_enc, age,
            means['Urea'], means['Cr'],
            hba1c,
            means['Chol'], means['TG'],
            means['HDL'], means['LDL'],
            means['VLDL'], means['BMI']
        ]])

        row_scaled = scaler.transform(row)
        pred = int(model.predict(row_scaled)[0])
        result = "Yes" if pred == 1 else "No"
        message = f"Diabetes Prediction: {result}"

        return jsonify({"result": result, "message": message})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Vercel handler
def handler(event, context=None):
    from werkzeug.wsgi import DispatcherMiddleware
    from flask import Flask
    app.wsgi_app = DispatcherMiddleware(Flask('dummy'), {'/': app})
    return app(event, context)

# Local testing
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)