import joblib
import numpy as np

# Load model and scaler
model = joblib.load("best_trained_voice_model.h5")

scaler = joblib.load("voice_scaler.pkl")

def run_inference_from_analysis_for_voice(features_array: np.ndarray, patient_id="Unknown"):
    try:
        if not isinstance(features_array, np.ndarray):
            raise ValueError("Input features must be a NumPy array.")


        # Reshape for scaler/model (single sample = 1 row)
        X = features_array.reshape(1, -1)

        # Scale
        X_scaled = scaler.transform(X)

        # Predict
        probabilities = model.predict_proba(X_scaled)
        predictions = model.predict(X_scaled)

        predicted_label = predictions[0]
        #probability = float(probabilities[0][predicted_label])
        probability = float(probabilities[0][1])
        pred_label = "Parkinson" if predicted_label == 1 else "Healthy"

        return {
            "label": pred_label,
            "probability": round(probability, 4)
        }
    except Exception as e:
        print(f"❌ Inference failed: {e}")
        return {
            "error": str(e)
        }
