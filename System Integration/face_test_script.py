# face_test_script.py
import json
from face_feature_extraction import extract_face_features
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

def run_inference_from_analysis_for_face(video_path: str):
    model_path = "best_trained_face_model.pkl"
    loaded_model = joblib.load(model_path)

    df_features = extract_face_features([video_path])
    if df_features.empty:
        raise ValueError("AU features could not be extracted, returned empty.")

    excluded_columns = ['total_frame', 'fps', 'time', 'videoPro_timedk']
    df_filtered = df_features.drop(columns=excluded_columns, errors='ignore')

    keep_columns = ['AU05', 'AU25', 'AU26', 'AU06', 'AU07', 'AU09', 'AU10', 'AU11', 'AU12', 'AU14', 'AU17',
                    'AU23', 'AU24', 'AU28', 'AU43', 'nf2', 'nf4', 'nf5', 'nf6', 'nf12', 'nf15',
                    'nf7', 'nf8', 'nf13', 'nf14', 'nf16', 'name']
    available_columns = [col for col in keep_columns if col in df_filtered.columns]

    name_column = df_filtered["name"].iloc[0] if "name" in df_filtered.columns else "Unknown"
    if "name" in available_columns:
        available_columns.remove("name")

    model_data = df_filtered[available_columns].copy()
    model_data = model_data.select_dtypes(include=['number'])

    o1_cols = ['AU05', 'AU25', 'AU26', 'nf14']
    available_o1 = [col for col in o1_cols if col in model_data.columns]
    if available_o1:
        model_data['o1'] = model_data[available_o1].apply(lambda x: (abs(x - x.mean())).mean(), axis=1)
        model_data.drop(columns=available_o1, inplace=True)

    o2_cols = ['AU06', 'AU09', 'AU10', 'AU43', 'AU12']
    available_o2 = [col for col in o2_cols if col in model_data.columns]
    if available_o2:
        model_data['o2'] = model_data[available_o2].apply(lambda x: (abs(x - x.mean())).mean(), axis=1)
        model_data.drop(columns=available_o2, inplace=True)

    Q1 = model_data.quantile(0.25)
    Q3 = model_data.quantile(0.75)
    IQR = Q3 - Q1
    is_outlier = (model_data < (Q1 - 1.5 * IQR)) | (model_data > (Q3 + 1.5 * IQR))
    for col in model_data.columns:
        mean_val = model_data[col].mean()
        model_data[col] = model_data[col].where(~is_outlier[col], mean_val)

    au_columns_norm = [col for col in model_data.columns if 'AU' in col]
    columns_to_scale = [col for col in model_data.columns if col not in au_columns_norm]
    if columns_to_scale:
        scaler = StandardScaler()
        model_data[columns_to_scale] = scaler.fit_transform(model_data[columns_to_scale])

    X_test = model_data.to_numpy()
    y_pred = loaded_model.predict(X_test)
    y_proba = loaded_model.predict_proba(X_test)

    prediction = "Parkinson" if y_pred[0] == 1 else "Healthy"
    healthy_prob = float(y_proba[0][0])
    parkinson_prob = float(y_proba[0][1])

    result = {
        "label": prediction,
        "probability": round(float(parkinson_prob), 4),  # Prediction confidence of the Parkinson model
    }

    return result


if __name__ == "__main__":
    try:
        video_path = "Cache/Video3_Face.mp4"
        result = run_inference_from_analysis_for_face(video_path)
        print(json.dumps(result))
    except Exception as e:
        print(json.dumps({"error": "No FACE video recording was found, so no result could be obtained."}))
