import pandas as pd
import mediapipe as mp
import cv2
import math
import time
import os
import numpy as np

def find_angle(x1, y1, x2, y2):
  theta = math.atan2(y2 - y1, x2 - x1)
  angle_in_degrees = theta * 180 / math.pi
  print(f'parkinson pizza:{abs(angle_in_degrees-90)}')
  return abs(angle_in_degrees - 90) 

def find_angle_3d(x1, y1, z1, x2, y2, z2):
    v1 = np.array([x2 - x1, y2 - y1, z2 - z1])
    v2 = np.array([0, -1, 0])

    unit_v1 = v1 / np.linalg.norm(v1)
    unit_v2 = v2 / np.linalg.norm(v2)

    dot_product = np.dot(unit_v1, unit_v2)
    angle = np.arccos(np.clip(dot_product, -1.0, 1.0)) 
    angle_deg = np.degrees(angle)
    return angle_deg

font = cv2.FONT_HERSHEY_SIMPLEX

# Colors
blue = (255, 127, 0)
red = (50, 50, 255)
green = (127, 255, 0)
dark_blue = (127, 20, 0)
light_green = (127, 233, 100)
yellow = (0, 255, 255)
pink = (255, 0, 255)

short_video_paths_parkinson = []
root_dir = "./Durus/healthy/"
for person_name in os.listdir(root_dir):
    person_path = os.path.join(root_dir, person_name)

    if os.path.isdir(person_path):
        for file in os.listdir(person_path):
            if file.endswith("TekAyak.mp4") or file.endswith("TekAyak.mov") or file.endswith("TekAyak.avi"):
                short_video_paths_parkinson.append(os.path.join(person_path, file))

print(short_video_paths_parkinson, len(short_video_paths_parkinson))

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

label = 0 # 1 parkinson 0 healthy

for file_name in short_video_paths_parkinson:
    cap = cv2.VideoCapture(file_name)
    if not cap.isOpened():
        print(f"Could not open video: {file_name}")
        continue
      
    video_name = os.path.basename(file_name).split('.')[0]  
    fps = cap.get(cv2.CAP_PROP_FPS)  
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)  

    df = pd.DataFrame()
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("No frames read from video.")
            break
            
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
                'Label': label  # Parkinson (1) veya Sağlıklı (0)
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

    csv_file = "new_posture_data_full_test_3105.csv"
    df.to_csv(csv_file, mode='a', index=False, header=not os.path.exists(csv_file))  # Append
