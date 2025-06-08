import pandas as pd
import mediapipe as mp
import cv2
import math
import time
import os
import numpy as np
import tsfel
import joblib

def find_angle(x1, y1, x2, y2):
  theta = math.atan2(y2 - y1, x2 - x1)
  angle_in_degrees = theta * 180 / math.pi
  return abs(angle_in_degrees - 90) 

def find_angle_3d(x1, y1, z1, x2, y2, z2):
    v1 = np.array([x2 - x1, y2 - y1, z2 - z1])
    v2 = np.array([0, -1, 0])  # yukarı yön

    unit_v1 = v1 / np.linalg.norm(v1)
    unit_v2 = v2 / np.linalg.norm(v2)

    dot_product = np.dot(unit_v1, unit_v2)
    angle = np.arccos(np.clip(dot_product, -1.0, 1.0)) 
    angle_deg = np.degrees(angle)
    return angle_deg

def extract_features_from_video(video_path):
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Colors
    blue = (255, 127, 0)
    red = (50, 50, 255)
    green = (127, 255, 0)
    dark_blue = (127, 20, 0)
    light_green = (127, 233, 100)
    yellow = (0, 255, 255)
    pink = (255, 0, 255)
   
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    df = pd.DataFrame()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open video: {video_path}")
        return df
    video_name = os.path.basename(video_path).split('.')[0]  
    fps = cap.get(cv2.CAP_PROP_FPS)  
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) 
    

    frame_idx = 0
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("No frames read from video.")
            break

        if image.shape[0] < image.shape[1]:  # Dikey video ise
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            
        frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        # Convert the BGR image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        keypoints = pose.process(image_rgb)

        if keypoints.pose_landmarks:
            lm = keypoints.pose_landmarks
            lmPose = mp_pose.PoseLandmark

            
            l_shldr_x = int(lm.landmark[lmPose.LEFT_SHOULDER].x * image.shape[1])
            l_shldr_y = int(lm.landmark[lmPose.LEFT_SHOULDER].y * image.shape[0])
            l_ear_x = int(lm.landmark[lmPose.LEFT_EAR].x * image.shape[1])
            l_ear_y = int(lm.landmark[lmPose.LEFT_EAR].y * image.shape[0])
            l_hip_x = int(lm.landmark[lmPose.LEFT_HIP].x * image.shape[1])
            l_hip_y = int(lm.landmark[lmPose.LEFT_HIP].y * image.shape[0])

            
            neck_inclination = find_angle(l_shldr_x, l_shldr_y, l_ear_x, l_ear_y)
            torso_inclination = find_angle(l_hip_x, l_hip_y, l_shldr_x, l_shldr_y)

            
            l_shldr_3d = lm.landmark[lmPose.LEFT_SHOULDER]
            r_shldr_3d = lm.landmark[lmPose.RIGHT_SHOULDER]
            l_ear_3d = lm.landmark[lmPose.LEFT_EAR]
            l_hip_3d = lm.landmark[lmPose.LEFT_HIP]
            neck_3d = find_angle_3d(l_shldr_3d.x, l_shldr_3d.y, l_shldr_3d.z, l_ear_3d.x, l_ear_3d.y, l_ear_3d.z)
            torso_3d = find_angle_3d(l_hip_3d.x, l_hip_3d.y, l_hip_3d.z, l_shldr_3d.x, l_shldr_3d.y, l_shldr_3d.z)


            # NEW FEATURES

            
            shoulder_height_diff = abs(l_shldr_y - l_hip_y)

            
            head_tilt_2d = find_angle(l_shldr_x, l_shldr_y, l_ear_x, l_ear_y)
            head_tilt_3d = find_angle_3d(l_shldr_3d.x, l_shldr_3d.y, l_shldr_3d.z, l_ear_3d.x, l_ear_3d.y, l_ear_3d.z)

            
            shoulder_line_angle_2d = find_angle(l_shldr_3d.x, l_shldr_3d.y, r_shldr_3d.x, r_shldr_3d.y)
            shoulder_line_angle_3d = find_angle_3d(l_shldr_3d.x, l_shldr_3d.y, l_shldr_3d.z, r_shldr_3d.x, r_shldr_3d.y, r_shldr_3d.z)


            
            cv2.circle(image, (l_shldr_x, l_shldr_y), 7, yellow, -1)
            cv2.circle(image, (l_ear_x, l_ear_y), 7, yellow, -1)
            cv2.circle(image, (l_shldr_x, l_shldr_y - 100), 7, yellow, -1)
            cv2.circle(image, (l_hip_x, l_hip_y), 7, yellow, -1)
            cv2.circle(image, (l_hip_x, l_hip_y - 100), 7, yellow, -1)

            angle_text_string = f'Boyun : {int(neck_inclination)}  Govde : {int(torso_inclination)}'


            # add to dataframe
            df = pd.concat([df, pd.DataFrame([{
                'Video': video_name,
                'Time': frame_number,
                'Neck_Inclination_2D': neck_inclination,
                'Torso_Inclination_2D': torso_inclination,
                'Neck_Inclination_3D': neck_3d,
                'Torso_Inclination_3D': torso_3d,
                'Shoulder_Height_Diff': shoulder_height_diff,
                'Head_Tilt_2D': head_tilt_2d,
                'Head_Tilt_3D': head_tilt_3d,
                'Shoulder_Line_Angle_2D': shoulder_line_angle_2d,
                'Shoulder_Line_Angle_3D': shoulder_line_angle_3d,
            }])], ignore_index=True)

            cv2.putText(image, angle_text_string, (10, 30), font, 0.9, light_green, 2)
            cv2.putText(image, str(int(neck_inclination)), (l_shldr_x + 10, l_shldr_y), font, 0.9, light_green, 2)
            cv2.putText(image, str(int(torso_inclination)), (l_hip_x + 10, l_hip_y), font, 0.9, light_green, 2)

            cv2.line(image, (l_shldr_x, l_shldr_y), (l_ear_x, l_ear_y), green, 4)
            cv2.line(image, (l_shldr_x, l_shldr_y), (l_shldr_x, l_shldr_y - 100), green, 4)
            cv2.line(image, (l_hip_x, l_hip_y), (l_shldr_x, l_shldr_y), green, 4)
            cv2.line(image, (l_hip_x, l_hip_y), (l_hip_x, l_hip_y - 100), green, 4)                                  


        # Write frame to video output
        #video_output.write(image)
    
        cv2.imshow('Posture_Detection', image)

        if cv2.waitKey(1) & 0xFF == ord('x'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return df

selected_features = ['Shoulder_Line_Angle_3D_MFCC_2', 'Shoulder_Line_Angle_3D_Histogram mode', 'Shoulder_Line_Angle_3D_Median absolute deviation', 'Head_Tilt_3D_Absolute energy', 'Torso_Inclination_2D_Median absolute deviation', 'Torso_Inclination_2D_Interquartile range', 'Head_Tilt_3D_Mean diff', 'Shoulder_Line_Angle_3D_Spectral decrease', 'Shoulder_Line_Angle_3D_MFCC_5', 'Shoulder_Line_Angle_3D_Interquartile range', 'Head_Tilt_2D_Max', 'Shoulder_Line_Angle_3D_Positive turning points', 'Head_Tilt_2D_Skewness', 'Shoulder_Line_Angle_3D_Max power spectrum', 'Head_Tilt_2D_MFCC_3', 'Head_Tilt_2D_Average power', 'Neck_Inclination_2D_LPCC_9', 'Shoulder_Line_Angle_3D_Peak to peak distance', 'Head_Tilt_3D_Spectral variation', 'Shoulder_Line_Angle_3D_Spectrogram mean coefficient_19.35Hz', 'Neck_Inclination_2D_LPCC_6', 'Neck_Inclination_3D_Interquartile range', 'Shoulder_Line_Angle_2D_Spectrogram mean coefficient_6.45Hz', 'Head_Tilt_2D_Median absolute deviation', 'Head_Tilt_2D_Mean', 'Head_Tilt_3D_LPCC_9', 'Neck_Inclination_3D_Area under the curve', 'Head_Tilt_2D_LPCC_3', 'Shoulder_Line_Angle_3D_MFCC_9', 'Head_Tilt_2D_Interquartile range']


def get_model_features_with_tsfel(df, video_name):
    cfg = tsfel.get_features_by_domain()
    neck_sample_2d = df["Neck_Inclination_2D"].reset_index(drop=True)
    torso_sample_2d = df["Torso_Inclination_2D"].reset_index(drop=True)
    neck_sample_3d = df["Neck_Inclination_3D"].reset_index(drop=True)
    torso_sample_3d = df["Torso_Inclination_3D"].reset_index(drop=True)
    shoulder_height_diff = df["Shoulder_Height_Diff"].reset_index(drop=True)
    head_tilt_2d = df["Head_Tilt_2D"].reset_index(drop=True)
    head_tilt_3d = df["Head_Tilt_3D"].reset_index(drop=True)
    shoulder_line_angle_2d = df["Shoulder_Line_Angle_2D"].reset_index(drop=True)
    shoulder_line_angle_3d = df["Shoulder_Line_Angle_3D"].reset_index(drop=True)

    features_neck_2d = tsfel.time_series_features_extractor(cfg, neck_sample_2d.to_frame(), verbose=0)
    features_torso_2d = tsfel.time_series_features_extractor(cfg, torso_sample_2d.to_frame(), verbose=0)
    features_neck_3d = tsfel.time_series_features_extractor(cfg, neck_sample_3d.to_frame(), verbose=0)
    features_torso_3d = tsfel.time_series_features_extractor(cfg, torso_sample_3d.to_frame(), verbose=0)
    features_shoulder_height_diff = tsfel.time_series_features_extractor(cfg, shoulder_height_diff.to_frame(), verbose=0)
    features_head_tilt_2d = tsfel.time_series_features_extractor(cfg, head_tilt_2d.to_frame(), verbose=0)
    features_head_tilt_3d = tsfel.time_series_features_extractor(cfg, head_tilt_3d.to_frame(), verbose=0)
    features_shoulder_line_angle_2d = tsfel.time_series_features_extractor(cfg, shoulder_line_angle_2d.to_frame(), verbose=0)
    features_shoulder_line_angle_3d = tsfel.time_series_features_extractor(cfg, shoulder_line_angle_3d.to_frame(), verbose=0)

    combined_features = pd.concat([features_neck_2d, features_torso_2d, features_neck_3d, features_torso_3d, features_shoulder_height_diff, features_head_tilt_2d, features_head_tilt_3d, features_shoulder_line_angle_2d, features_shoulder_line_angle_3d], axis=1)
    combined_features['Video'] = video_name

    filtered_df = combined_features[selected_features + ['Video']]
    return filtered_df

# Feature extraction from video - Mediapipe
video_path = "video-path" # change this to your video file path
video_name = os.path.basename(video_path).split('.')[0]
print(f"Extracted features from video: {video_name}")
df_features = extract_features_from_video(video_path)
df_features_after_tsfel = get_model_features_with_tsfel(df_features, video_name)
model = joblib.load('posture_modality_no_scale.h5')
X_input = df_features_after_tsfel.drop(columns=['Video'])
y_pred = model.predict(X_input)
print(f"Predicted class: {y_pred[0]}")

y_proba = model.predict_proba(X_input)
print(f"Class 0 prob.: {y_proba[0][0]:.2f}")
print(f"Class 1 prob.: {y_proba[0][1]:.2f}")


