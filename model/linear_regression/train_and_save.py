# 🔁 train_and_save_model.py

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib
import matplotlib.pyplot as plt

df = pd.read_csv("data/sales_1000.csv")
df['total'] = df['quantity'] * df['price']

X = pd.get_dummies(df[['product', 'customer', 'quantity', 'price']], drop_first=True)
y = df['total']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

joblib.dump(model, "model/linear_regression/linear_regression_model.pkl")
joblib.dump(X.columns.tolist(), "model/linear_regression/linear_regression_columns.pkl")
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