# api/routes/v1.py

from flask import Blueprint, request, jsonify
import pandas as pd
import joblib
import os

v1 = Blueprint("v1", __name__, url_prefix="/api/v1")

# Load model and column structure
model_path = os.path.join("model/random_forest", "random_forest_model.pkl")
columns_path = os.path.join("model/random_forest", "random_forest_model_columns.pkl")
model = joblib.load(model_path)
model_columns = joblib.load(columns_path)

# Logging function
def log_data_to_csv(data, filepath="data/new_sales.csv"):
    df = pd.DataFrame(data)
    file_exists = os.path.isfile(filepath)
    df.to_csv(filepath, mode='a', index=False, header=not file_exists)

# Input validator
def validate_record(record):
    required_fields = ["product", "customer", "quantity", "price"]
    missing = [field for field in required_fields if field not in record]
    if missing:
        return f"Missing fields: {', '.join(missing)}"

    if not isinstance(record['quantity'], (int, float)) or record['quantity'] <= 0:
        return "Quantity must be a positive number."

    if not isinstance(record['price'], (int, float)) or record['price'] <= 0:
        return "Price must be a positive number."

    return None

@v1.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Normalize input: ensure it's a list of records
        if isinstance(data, dict):
            data = [data]
        elif not isinstance(data, list):
            return jsonify({"error": "Input should be a JSON object or a list of objects."}), 400

        results = []
        valid_data_to_log = []

        for idx, record in enumerate(data):
            error = validate_record(record)
            if error:
                results.append({"index": idx, "error": error})
                continue

            input_df = pd.DataFrame([record])
            input_encoded = pd.get_dummies(input_df)

            # Add missing columns
            for col in model_columns:
                if col not in input_encoded:
                    input_encoded[col] = 0

            # Align column order
            input_encoded = input_encoded[model_columns]

            # Predict
            prediction = model.predict(input_encoded)[0]
            results.append({
                "index": idx,
                "predicted_total": round(prediction, 2)
            })

            valid_data_to_log.append(record)

        # Log only valid records
        if valid_data_to_log:
            log_data_to_csv(valid_data_to_log)

        return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e)}), 500
