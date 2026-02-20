# tandem_walk_test_script.py

import numpy as np
import pymongo
import pickle
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import interp1d

# -------------------- Load Model and Scaler --------------------
model = load_model("best_trained_model.h5")

with open("scaler.pkl", "rb") as f:
    scaler: StandardScaler = pickle.load(f)

# -------------------- MongoDB Connection --------------------
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["Test_Module"]
test_collection = db["tandem_walk_test_module"]

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

def parse_patient_id(doc):
    return doc.get("Person", "Unknown")

# -------------------- Test Data Extraction and Prediction --------------------
X_test = []
patient_ids = []

for doc in test_collection.find():
    matrix = doc_to_matrix(doc)
    matrix_scaled = scaler.transform(matrix)
    X_test.append(matrix_scaled)
    patient_ids.append(parse_patient_id(doc))

X_test = np.array(X_test)

# -------------------- Make a Prediction --------------------
pred_probs = model.predict(X_test)
predictions = (pred_probs > 0.5).astype(int)

# -------------------- Print Results --------------------
print("------ TEST RESULTS ------")
for pid, pred, prob in zip(patient_ids, predictions, pred_probs):
    label = "Parkinson" if pred == 1 else "Healthy"
    print(f"{pid}: Prediction = {label} (Probability: {prob[0]:.4f})")
