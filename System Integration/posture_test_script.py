# posture_test_script.py

import numpy as np
import joblib

# -------------------- Load Model --------------------
model = joblib.load("best_trained_posture_model.pkl")

# -------------------- Inference Function --------------------
def run_inference_from_analysis_for_posture(features_df, patient_id="Unknown"):
    try:
        if "Video" in features_df.columns:
            X_input = features_df.drop(columns=["Video"]).values
        else:
            X_input = features_df.values

        pred_proba = model.predict_proba(X_input)
        pred_class = model.predict(X_input)

        #probability = float(pred_proba[0][pred_class[0]])
        probability = float(pred_proba[0][1])
        pred_label = "Parkinson" if pred_class[0] == 1 else "Healthy"

        return {
            "label": pred_label,
            "probability": round(probability, 4)
        }
    except Exception as e:
        print(f"❌ Inference failed: {e}")
        return {
            "error": str(e)
        }
