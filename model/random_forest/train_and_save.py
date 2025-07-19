# 🔁 train_and_save_model.py

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import matplotlib.pyplot as plt

# Load dataset
#df = pd.read_csv("data/sales_50.csv")
df = pd.read_csv("data/sales_1000.csv")
df['total'] = df['quantity'] * df['price']

# Prepare features and target
X = pd.get_dummies(df[['product', 'customer', 'quantity', 'price']], drop_first=True)
y = df['total']

# Split for evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
print("\n✅ Model Evaluation:")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"R² Score: {r2:.2f}")

# Save model
joblib.dump(model, "model/random_forest/random_forest_model.pkl")
joblib.dump(X.columns.tolist(), "model/random_forest/random_forest_model_columns.pkl")
print("✅ Model and column structure saved.")

# Visualize
plt.figure(figsize=(10, 5))
plt.plot(y_test.values, label='Actual', marker='o')
plt.plot(predictions, label='Predicted', marker='x')
plt.title('Actual vs Predicted Sales Totals')
plt.xlabel('Sample Index')
plt.ylabel('Total (AED)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("charts/linear_regression_prediction_vs_actual.png")
plt.show()