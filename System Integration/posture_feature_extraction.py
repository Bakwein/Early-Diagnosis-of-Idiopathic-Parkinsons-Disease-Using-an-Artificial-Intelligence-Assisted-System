# posture_feature_extraction.py

import os
import math
import cv2
import tsfel
import numpy as np
import pandas as pd
import mediapipe as mp

# -------------------- Config --------------------
selected_features = [
    "Shoulder_Line_Angle_3D_Histogram mode",
    "Shoulder_Line_Angle_2D_MFCC_6",
    "Shoulder_Line_Angle_3D_MFCC_2",
    "Head_Tilt_3D_LPCC_3",
    "Neck_Inclination_3D_LPCC_9",
    "Head_Tilt_3D_Absolute energy",
    "Head_Tilt_2D_LPCC_3",
    "Head_Tilt_2D_Mean",
    "Head_Tilt_2D_LPCC_4",
    "Head_Tilt_2D_Max",
    "Head_Tilt_3D_Area under the curve",
    "Neck_Inclination_3D_LPCC_3",
    "Torso_Inclination_2D_Interquartile range",
    "Shoulder_Line_Angle_3D_Positive turning points",
    "Head_Tilt_2D_Positive turning points",
    "Head_Tilt_3D_Negative turning points",
    "Shoulder_Line_Angle_2D_Spectrogram mean coefficient_41.94Hz",
    "Head_Tilt_2D_Root mean square",
    "Shoulder_Line_Angle_3D_Negative turning points",
    "Head_Tilt_3D_LPCC_9",
    "Shoulder_Line_Angle_3D_MFCC_5",
    "Shoulder_Line_Angle_3D_Wavelet entropy",
    "Head_Tilt_2D_Spectral entropy",
    "Shoulder_Line_Angle_2D_ECDF Percentile_0",
    "Shoulder_Line_Angle_2D_Spectral slope",
    "Neck_Inclination_2D_Positive turning points",
    "Head_Tilt_2D_LPCC_9",
    "Head_Tilt_2D_Human range energy",
    "Head_Tilt_2D_Interquartile range",
    "Shoulder_Line_Angle_3D_Mean absolute deviation"
    ]  # Likewise, it will be filled in from above

# -------------------- Helper Functions --------------------
def find_angle(x1, y1, x2, y2):
    theta = math.atan2(y2 - y1, x2 - x1)
    return abs(theta * 180 / math.pi - 90)

def find_angle_3d(x1, y1, z1, x2, y2, z2):
    v1 = np.array([x2 - x1, y2 - y1, z2 - z1])
    v2 = np.array([0, -1, 0])
    unit_v1 = v1 / np.linalg.norm(v1)
    unit_v2 = v2 / np.linalg.norm(v2)
    return np.degrees(np.arccos(np.clip(np.dot(unit_v1, unit_v2), -1.0, 1.0)))

# -------------------- Keypoint Extraction --------------------
def extract_posture_features(video_path):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    df = pd.DataFrame()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Video cannot be opened: {video_path}")

    video_name = os.path.basename(video_path).split('.')[0]

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        if image.shape[0] < image.shape[1]:
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        keypoints = pose.process(image_rgb)

        if keypoints.pose_landmarks:
            lm = keypoints.pose_landmarks
            lmPose = mp_pose.PoseLandmark

            try:
                l_shldr_x = int(lm.landmark[lmPose.LEFT_SHOULDER].x * image.shape[1])
                l_shldr_y = int(lm.landmark[lmPose.LEFT_SHOULDER].y * image.shape[0])
                l_ear_x = int(lm.landmark[lmPose.LEFT_EAR].x * image.shape[1])
                l_ear_y = int(lm.landmark[lmPose.LEFT_EAR].y * image.shape[0])
                l_hip_x = int(lm.landmark[lmPose.LEFT_HIP].x * image.shape[1])
                l_hip_y = int(lm.landmark[lmPose.LEFT_HIP].y * image.shape[0])

                l_shldr_3d = lm.landmark[lmPose.LEFT_SHOULDER]
                r_shldr_3d = lm.landmark[lmPose.RIGHT_SHOULDER]
                l_ear_3d = lm.landmark[lmPose.LEFT_EAR]
                l_hip_3d = lm.landmark[lmPose.LEFT_HIP]

                df = pd.concat([df, pd.DataFrame([{
                    'Video': video_name,
                    'Time': cap.get(cv2.CAP_PROP_POS_FRAMES),
                    'Neck_Inclination_2D': find_angle(l_shldr_x, l_shldr_y, l_ear_x, l_ear_y),
                    'Torso_Inclination_2D': find_angle(l_hip_x, l_hip_y, l_shldr_x, l_shldr_y),
                    'Neck_Inclination_3D': find_angle_3d(l_shldr_3d.x, l_shldr_3d.y, l_shldr_3d.z,
                                                         l_ear_3d.x, l_ear_3d.y, l_ear_3d.z),
                    'Torso_Inclination_3D': find_angle_3d(l_hip_3d.x, l_hip_3d.y, l_hip_3d.z,
                                                          l_shldr_3d.x, l_shldr_3d.y, l_shldr_3d.z),
                    'Shoulder_Height_Diff': abs(l_shldr_y - l_hip_y),
                    'Head_Tilt_2D': find_angle(l_shldr_x, l_shldr_y, l_ear_x, l_ear_y),
                    'Head_Tilt_3D': find_angle_3d(l_shldr_3d.x, l_shldr_3d.y, l_shldr_3d.z,
                                                  l_ear_3d.x, l_ear_3d.y, l_ear_3d.z),
                    'Shoulder_Line_Angle_2D': find_angle(l_shldr_3d.x, l_shldr_3d.y,
                                                         r_shldr_3d.x, r_shldr_3d.y),
                    'Shoulder_Line_Angle_3D': find_angle_3d(l_shldr_3d.x, l_shldr_3d.y, l_shldr_3d.z,
                                                            r_shldr_3d.x, r_shldr_3d.y, r_shldr_3d.z),
                }])], ignore_index=True)
            except:
                continue

    cap.release()

    if df.empty:
        raise ValueError("No valid keypoints extracted from posture video.")

    # -------------------- TSFEL Feature Extraction --------------------
    cfg = tsfel.get_features_by_domain()
    features = []

    for col in [
        "Neck_Inclination_2D", "Torso_Inclination_2D", "Neck_Inclination_3D", "Torso_Inclination_3D",
        "Shoulder_Height_Diff", "Head_Tilt_2D", "Head_Tilt_3D",
        "Shoulder_Line_Angle_2D", "Shoulder_Line_Angle_3D"
    ]:
        series = df[col].reset_index(drop=True).to_frame()
        features.append(tsfel.time_series_features_extractor(cfg, series, verbose=0))

    combined = pd.concat(features, axis=1)
    combined["Video"] = video_name

    return combined[selected_features + ["Video"]]
