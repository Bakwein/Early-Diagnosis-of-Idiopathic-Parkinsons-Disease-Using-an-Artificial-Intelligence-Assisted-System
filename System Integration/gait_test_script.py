# gait_test_script.py

import numpy as np
import pickle
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import interp1d

# -------------------- Load Model and Scaler --------------------
model = load_model("best_trained_walking_model.h5")

with open("walking_scaler.pkl", "rb") as f:
    scaler: StandardScaler = pickle.load(f)

# -------------------- Preprocessing Functions --------------------
def doc_to_matrix(doc):
    left_foot_lift = np.array(doc["left_foot_lift_heights"])
    right_foot_lift = np.array(doc["right_foot_lift_heights"])
    lfa_2d = np.array(doc["lfa_2d"])
    lfa_3d = np.array(doc["lfa_3d"])
    rfa_2d = np.array(doc["rfa_2d"])
    rfa_3d = np.array(doc["rfa_3d"])
    left_knee_angle = np.array(doc["left_knee_angle_list"])
    right_knee_angle = np.array(doc["right_knee_angle_list"])
    hip_angle_list_2d = np.array(doc["hip_angle_list_2d"])
    hip_angle_list_3d = np.array(doc["hip_angle_list_3d"])
    heel_angle_list_2d = np.array(doc["heel_angle_list_2d"])
    heel_angle_list_3d = np.array(doc["heel_angle_list_3d"])

    X_temp = np.column_stack([
        left_foot_lift,
        right_foot_lift,
        lfa_2d,
        lfa_3d,
        rfa_2d,
        rfa_3d,
        left_knee_angle,
        right_knee_angle,
        hip_angle_list_2d,
        hip_angle_list_3d,
        heel_angle_list_2d,
        heel_angle_list_3d
    ])
    return rescale_to_fixed_length(X_temp)

def rescale_to_fixed_length(data, target_length=1200):
    time_steps = np.arange(data.shape[0])
    new_time_steps = np.linspace(0, time_steps.max(), target_length)
    interpolator = interp1d(time_steps, data, axis=0, kind='linear', fill_value="extrapolate")
    return interpolator(new_time_steps)

# -------------------- Inference Function --------------------
def run_inference_from_analysis_for_walking(full_analysis, patient_id="Unknown"):
    try:
        matrix = doc_to_matrix(full_analysis)
        matrix_scaled = scaler.transform(matrix)
        matrix_scaled = np.expand_dims(matrix_scaled, axis=0)  # batch dimension

        pred_prob = model.predict(matrix_scaled)[0][0]
        pred_label = "Parkinson" if pred_prob > 0.5 else "Healthy"

        return {
            "label": pred_label,
            "probability": round(float(pred_prob), 4)
        }
    except Exception as e:
        print(f"❌ Inference failed: {e}")
        return {
            "error": str(e)
        }
