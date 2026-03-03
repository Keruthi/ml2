from flask import Flask, request, jsonify
import pickle
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import os

app = Flask(__name__)

# -------------------------------
# Load Model and Scaler Safely
# -------------------------------
try:
    with open('logistic_regression_model.pkl', 'rb') as file:
        model = pickle.load(file)

    with open('standard_scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)

except Exception as e:
    print("Error loading model/scaler:", e)
    model = None
    scaler = None

# -------------------------------
# Feature Columns
# -------------------------------
feature_columns = [
    'rainfall_mm', 'storm_event_flag', 'wet_dry_index', 'inflow_m3_hr',
    'BOD_in', 'COD_in', 'TSS_in', 'TN_in', 'TP_in', 'NH4_in',
    'aeration_rate', 'sludge_recirculation', 'chemical_dose',
    'HRT', 'SRT', 'DO_setpoint', 'energy_kwh', 'total_operational_cost'
]

# -------------------------------
# Home Route
# -------------------------------
@app.route('/')
def home():
    return "ML Prediction API is Running 🚀"

# -------------------------------
# Health Route
# -------------------------------
@app.route('/health')
def health():
    return jsonify({"status": "Server is running"})

# -------------------------------
# Prediction Route (GET + POST)
# -------------------------------
@app.route('/predict', methods=['GET', 'POST'])
def predict():

    if model is None or scaler is None:
        return jsonify({"error": "Model or Scaler not loaded properly"}), 500

    try:
        # -------------------------------
        # Handle POST (JSON)
        # -------------------------------
        if request.method == "POST":
            if not request.is_json:
                return jsonify({"error": "Request must be JSON"}), 400
            data = request.get_json()

        # -------------------------------
        # Handle GET (Query Params)
        # -------------------------------
        elif request.method == "GET":
            data = request.args.to_dict()

        # Convert values to float
        input_data = {}
        for col in feature_columns:
            if col not in data:
                return jsonify({"error": f"Missing feature: {col}"}), 400
            input_data[col] = float(data[col])

        # Create DataFrame
        input_df = pd.DataFrame([input_data])

        # Scale
        scaled_input = scaler.transform(input_df)

        # Predict
        prediction = model.predict(scaled_input)
        prediction_proba = model.predict_proba(scaled_input)

        prob_0 = float(prediction_proba[0][0])
        prob_1 = float(prediction_proba[0][1])

        # -------------------------------
        # Create Graph
        # -------------------------------
        plt.figure()
        plt.bar(['Class 0', 'Class 1'], [prob_0, prob_1])
        plt.xlabel("Classes")
        plt.ylabel("Probability")
        plt.title("Prediction Probability")

        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plt.close()

        graph_url = base64.b64encode(img.getvalue()).decode()

        return jsonify({
            "prediction": int(prediction[0]),
            "probability_class_0": prob_0,
            "probability_class_1": prob_1,
            "graph_base64": graph_url
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -------------------------------
# Run App (Render Compatible)
# -------------------------------
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
