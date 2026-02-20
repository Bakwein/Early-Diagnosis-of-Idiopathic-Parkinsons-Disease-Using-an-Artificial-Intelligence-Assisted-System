import pandas as pd
import joblib
import numpy as np

# 1. Load the model and scaler
model = joblib.load("random_forest_pipelined_model.h5")
scaler = joblib.load("pipeline_scaler.pkl")

# 2. Load the data
df = pd.read_csv("features_pipelined.csv")

# 3. Separate features, labels, and 'name' column
X = df.drop(columns=["label", "name", "mfcc_6_mean", "mfcc_11_mean"]).values
y_true = df["label"].values
names = df["name"].values

# 4. Scaling
X_scaled = scaler.transform(X)

# 5. Prediction probabilities and class predictions
probabilities = model.predict_proba(X_scaled)
predictions = model.predict(X_scaled)

# 6. Print results
print("------ TEST RESULTS ------")
for name, true_label, pred_label, prob in zip(names, y_true, predictions, probabilities):
    label_str = "Parkinson" if pred_label == 1 else "Healthy"
    correctness = "✅ Correct" if pred_label == true_label else "❌ Incorrect"
    probability = prob[pred_label]
    print(f"{name}: Prediction = {label_str} (Probability: {probability:.4f}) → {correctness}")
