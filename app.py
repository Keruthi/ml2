from flask import Flask, request, jsonify
import pickle
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # ✅ IMPORTANT for server (no GUI)
import matplotlib.pyplot as plt
import io
import base64
import os

app = Flask(__name__)

# Load model
with open('logistic_regression_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load scaler
with open('standard_scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

feature_columns = [
    'rainfall_mm', 'storm_event_flag', 'wet_dry_index', 'inflow_m3_hr',
    'BOD_in', 'COD_in', 'TSS_in', 'TN_in', 'TP_in', 'NH4_in',
    'aeration_rate', 'sludge_recirculation', 'chemical_dose',
    'HRT', 'SRT', 'DO_setpoint', 'energy_kwh', 'total_operational_cost'
]

@app.route('/')
def home():
    return "ML Prediction API is Running 🚀"

@app.route('/predict', methods=['POST'])
def predict():

    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()

    try:
        input_df = pd.DataFrame([data], columns=feature_columns)
        scaled_input = scaler.transform(input_df)

        prediction = model.predict(scaled_input)
        prediction_proba = model.predict_proba(scaled_input)

        prob_0 = float(prediction_proba[0][0])
        prob_1 = float(prediction_proba[0][1])

        # Create Graph
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
            "prediction_text": f"Predicted Class: {int(prediction[0])}",
            "probability_class_0": prob_0,
            "probability_class_1": prob_1,
            "graph": graph_url
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ✅ IMPORTANT FIX FOR RENDER
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
