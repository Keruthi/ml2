from flask import Flask, request, jsonify
import pickle
import pandas as pd

# Initialize a Flask application instance
app = Flask(__name__)

# Load the logistic regression model
with open('logistic_regression_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the standard scaler
with open('standard_scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Define the feature columns - ensure this matches the order used during training
feature_columns = ['rainfall_mm', 'storm_event_flag', 'wet_dry_index', 'inflow_m3_hr',
                   'BOD_in', 'COD_in', 'TSS_in', 'TN_in', 'TP_in', 'NH4_in',
                   'aeration_rate', 'sludge_recirculation', 'chemical_dose',
                   'HRT', 'SRT', 'DO_setpoint', 'energy_kwh', 'total_operational_cost']

@app.route('/predict', methods=['POST'])
def predict():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json(force=True)
    
    # Ensure data is a dictionary
    if not isinstance(data, dict):
        return jsonify({"error": "Invalid JSON format. Expected a dictionary."}), 400

    try:
        # Convert incoming JSON data to a pandas DataFrame
        input_df = pd.DataFrame([data], columns=feature_columns)

        # Preprocess the data using the loaded scaler
        scaled_input = scaler.transform(input_df)

        # Make a prediction using the loaded model
        prediction = model.predict(scaled_input)
        prediction_proba = model.predict_proba(scaled_input)

        # Return the prediction as a JSON response
        return jsonify({
            "prediction": int(prediction[0]),
            "probability_class_0": float(prediction_proba[0][0]),
            "probability_class_1": float(prediction_proba[0][1])
        })
    except KeyError as e:
        return jsonify({"error": f"Missing expected feature in JSON data: {e}"}), 400
    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {e}"}), 500

# To run the app, you would typically use:
# if __name__ == '__main__':
#     app.run(debug=True)
