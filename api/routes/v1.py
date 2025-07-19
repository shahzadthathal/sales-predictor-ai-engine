# api/routes/v1.py

from flask import Blueprint, request, jsonify
import pandas as pd
import joblib
import os

# Load model and columns once
model_path = os.path.join("model/random_forest", "random_forest_model.pkl")
columns_path = os.path.join("model/random_forest", "random_forest_model_columns.pkl")
model = joblib.load(model_path)
model_columns = joblib.load(columns_path)

v1 = Blueprint("v1", __name__, url_prefix="/api/v1")

@v1.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        required_fields = ["product", "customer", "quantity", "price"]
        missing = [field for field in required_fields if field not in data]
        if missing:
            return jsonify({"error": f"Missing fields: {', '.join(missing)}"}), 400

        if not isinstance(data['quantity'], (int, float)) or data['quantity'] <= 0:
            return jsonify({"error": "Quantity must be a positive number."}), 400

        if not isinstance(data['price'], (int, float)) or data['price'] <= 0:
            return jsonify({"error": "Price must be a positive number."}), 400

        input_df = pd.DataFrame([data])
        input_encoded = pd.get_dummies(input_df)

        for col in model_columns:
            if col not in input_encoded:
                input_encoded[col] = 0

        input_encoded = input_encoded[model_columns]
        prediction = model.predict(input_encoded)[0]

        return jsonify({"predicted_total": round(prediction, 2)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
