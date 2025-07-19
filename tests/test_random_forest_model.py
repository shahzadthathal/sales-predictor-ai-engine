import unittest
import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, r2_score
import os

class TestSalesPredictionModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load model and columns from model/ directory
        model_path = os.path.join("model/random_forest", "random_forest_model.pkl")
        columns_path = os.path.join("model/random_forest", "random_forest_model_columns.pkl")
        cls.model = joblib.load(model_path)
        cls.columns = joblib.load(columns_path)

        # Load and prepare test data
        df = pd.read_csv("data/sales_50.csv")
        df["total"] = df["quantity"] * df["price"]
        cls.X = pd.get_dummies(df[["product", "customer", "quantity", "price"]], drop_first=True)
        cls.y = df["total"]

        # Align test features with model columns
        for col in cls.columns:
            if col not in cls.X:
                cls.X[col] = 0
        cls.X = cls.X[cls.columns]

        # Predict once for all tests
        cls.y_pred = cls.model.predict(cls.X)

    def test_shape_match(self):
        """Ensure prediction count matches actual labels"""
        self.assertEqual(len(self.y_pred), len(self.y), "Mismatch in prediction and label counts")

    def test_mean_absolute_error(self):
        """MAE should be reasonably low (within ~AED 50)"""
        mae = mean_absolute_error(self.y, self.y_pred)
        print(f"🔍 Mean Absolute Error: {mae:.2f}")
        self.assertLess(mae, 50, "MAE should be below 50 for this dataset")

    def test_r2_score(self):
        """R² score should indicate strong prediction performance"""
        r2 = r2_score(self.y, self.y_pred)
        print(f"📊 R² Score: {r2:.2f}")
        self.assertGreater(r2, 0.95, "R² should be above 0.95 for a good model")

if __name__ == "__main__":
    unittest.main()
