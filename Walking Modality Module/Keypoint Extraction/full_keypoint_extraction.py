import cv2
import mediapipe as mp
import numpy as np
import csv
import math
import matplotlib.pyplot as plt
import pandas as pd
from pymongo import MongoClient

# ----- Model
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# ----- Keypoints

LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28
LEFT_HEEL = 29
RIGHT_HEEL = 30
LEFT_FOOT_INDEX = 31  # Left toe tip
RIGHT_FOOT_INDEX = 32  # Right toe tip

# ----- Calculate angle between two points and the horizontal axis (ground)
def calculate_angle(x1, y1, x2, y2):
    return math.degrees(math.atan2(y2 - y1, x2 - x1))

# ----- Calculate 3D angle
def calculate_angle_3d(x1, y1, z1, x2, y2, z2):
    vector_1 = np.array([x1, y1, z1])
    vector_2 = np.array([x2, y2, z2])
    dot_product = np.dot(vector_1, vector_2)
    norm_1 = np.linalg.norm(vector_1)
    norm_2 = np.linalg.norm(vector_2)
    angle_rad = np.arccos(dot_product / (norm_1 * norm_2))
    angle_deg = np.degrees(angle_rad)
    return angle_deg

# ----- Calculate distance
def calculate_distance(z1, z2, y1, y2):
    height_diff = abs(y2 - y1)
    depth_diff = abs(z2 - z1)
    distance = math.sqrt(height_diff ** 2 + depth_diff ** 2)
    return distance

def calculate_3d_angle(a, b, c):
    """
    a, b, c: three-point (hip, knee, ankle) positions
    Calculates the angle for point b in the a -> b -> c sequence.
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle = np.arccos(cosine_angle)

    return np.degrees(angle)

def extend_with_sliding_window(data_list, target_length=1200, window_size=100):
    current_length = len(data_list)

    if current_length >= target_length:
        return data_list[:target_length]  # max length

    last_window = data_list[-window_size:]  # take last window
    missing_length = target_length - current_length  # calculate the minor length

    # Sliding window
    pattern = np.tile(last_window, missing_length // window_size + 1)[:missing_length]

    return data_list + pattern.tolist()  # append to the list


# ------ Information of the person
NAME_SURNAME = "Name - Surname"
SEX = "Female"
PARKINSON = "0"


# ----- Video path
video_path = "sample_video.mp4"
cap = cv2.VideoCapture(video_path)

frame_no = 0

full_analysis = {
    "left_foot_lift_heights": [], # for the foot lift height
    "right_foot_lift_heights": [], # for the foot lift height
    "lfa_2d": [], # for the 2d left foot placement angle
    "rfa_2d": [], # for the 2d right foot placement angle
    "lfa_3d": [], # for the 3d left foot placement angle
    "rfa_3d": [], # for the 3d right foot placement angle
    "left_knee_angle_list": [], # for the left knee angle
    "right_knee_angle_list": [], # for the right knee angle
    "hip_angle_list_2d": [], # for the 2d hip horizontal shift angle
    "hip_angle_list_3d": [], # for the 3d hip horizontal shift angle
    "heel_angle_list_2d": [], # for the 2d heel horizontal shift angle
    "heel_angle_list_3d": [], # for the 3d heel horizontal shift angle
}

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame_no += 1
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:

            # ---------------------------------- 2D Hip Horizontal Shift Angle ----------------------------------
            left_hip = results.pose_landmarks.landmark[LEFT_HIP]
            right_hip = results.pose_landmarks.landmark[RIGHT_HIP]

            x1, y1 = int(left_hip.x * frame.shape[1]), int(left_hip.y * frame.shape[0])
            x2, y2 = int(right_hip.x * frame.shape[1]), int(right_hip.y * frame.shape[0])

            angle = calculate_angle(x1, y1, x2, y2)
            full_analysis["hip_angle_list_2d"].append(angle)
            # ----------------------------------------------------------------------------------------------



            #---------------------------------- 3D Hip Horizontal Shift Angle ----------------------------------
            x1, y1, z1 = left_hip.x, left_hip.y, left_hip.z
            x2, y2, z2 = right_hip.x, right_hip.y, right_hip.z

            angle = calculate_angle_3d(x1, y1, z1, x2, y2, z2)
            full_analysis["hip_angle_list_3d"].append(angle)

            x1_pixel, y1_pixel = int(left_hip.x * frame.shape[1]), int(left_hip.y * frame.shape[0])
            x2_pixel, y2_pixel = int(right_hip.x * frame.shape[1]), int(right_hip.y * frame.shape[0])
            # ----------------------------------------------------------------------------------------------



            #---------------------------------- 2D Heel Horizontal Shift Angle ----------------------------------
            left_heel = results.pose_landmarks.landmark[LEFT_HEEL]
            right_heel = results.pose_landmarks.landmark[RIGHT_HEEL]

            x1, y1 = int(left_heel.x * frame.shape[1]), int(left_heel.y * frame.shape[0])
            x2, y2 = int(right_heel.x * frame.shape[1]), int(right_heel.y * frame.shape[0])

            angle = calculate_angle(x1, y1, x2, y2)
            full_analysis["heel_angle_list_2d"].append(angle)
            # ----------------------------------------------------------------------------------------------



            #---------------------------------- 3D Heel Horizontal Shift Angle ----------------------------------
            x1, y1, z1 = left_heel.x, left_heel.y, left_heel.z
            x2, y2, z2 = right_heel.x, right_heel.y, right_heel.z

            angle = calculate_angle_3d(x1, y1, z1, x2, y2, z2)
            full_analysis["heel_angle_list_3d"].append(angle)
            # ----------------------------------------------------------------------------------------------



            #---------------------------------- 2D Foot Placement Angle ----------------------------------
            left_heel = results.pose_landmarks.landmark[LEFT_HEEL]
            left_toe = results.pose_landmarks.landmark[LEFT_FOOT_INDEX]

            right_heel = results.pose_landmarks.landmark[RIGHT_HEEL]
            right_toe = results.pose_landmarks.landmark[RIGHT_FOOT_INDEX]

            x1_l, y1_l = int(left_heel.x * frame.shape[1]), int(left_heel.y * frame.shape[0])
            x2_l, y2_l = int(left_toe.x * frame.shape[1]), int(left_toe.y * frame.shape[0])
            left_foot_angle = calculate_angle(x1_l, y1_l, x2_l, y2_l)

            x1_r, y1_r = int(right_heel.x * frame.shape[1]), int(right_heel.y * frame.shape[0])
            x2_r, y2_r = int(right_toe.x * frame.shape[1]), int(right_toe.y * frame.shape[0])
            right_foot_angle = calculate_angle(x1_r, y1_r, x2_r, y2_r)

            full_analysis["lfa_2d"].append(angle)
            full_analysis["rfa_2d"].append(angle)
            #----------------------------------------------------------------------------------------------



            #---------------------------------- 3D Foot Placement Angle ----------------------------------
            left_heel = results.pose_landmarks.landmark[LEFT_HEEL]
            right_heel = results.pose_landmarks.landmark[RIGHT_HEEL]

            left_foot_index = results.pose_landmarks.landmark[LEFT_FOOT_INDEX]
            right_foot_index = results.pose_landmarks.landmark[RIGHT_FOOT_INDEX]

            left_heel_coords = (
            left_heel.x * frame.shape[1], left_heel.y * frame.shape[0], left_heel.z * frame.shape[1])
            right_heel_coords = (
            right_heel.x * frame.shape[1], right_heel.y * frame.shape[0], right_heel.z * frame.shape[1])
            left_toe_coords = (
            left_foot_index.x * frame.shape[1], left_foot_index.y * frame.shape[0], left_foot_index.z * frame.shape[1])
            right_toe_coords = (right_foot_index.x * frame.shape[1], right_foot_index.y * frame.shape[0],
                                right_foot_index.z * frame.shape[1])

            left_angle = calculate_angle_3d(left_heel_coords[0], left_heel_coords[1], left_heel_coords[2],
                                            left_toe_coords[0], left_toe_coords[1], left_toe_coords[2])
            right_angle = calculate_angle_3d(right_heel_coords[0], right_heel_coords[1], right_heel_coords[2],
                                             right_toe_coords[0], right_toe_coords[1], right_toe_coords[2])

            full_analysis["lfa_3d"].append(angle)
            full_analysis["rfa_3d"].append(angle)
            # ----------------------------------------------------------------------------------------------



            #---------------------------------- Knee Angle ----------------------------------

            landmarks = results.pose_landmarks.landmark

            # Left hip, knee and ankle points
            left_hip = [landmarks[LEFT_HIP].x * frame.shape[1], landmarks[LEFT_HIP].y * frame.shape[0],
                        landmarks[LEFT_HIP].z]
            left_knee = [landmarks[LEFT_KNEE].x * frame.shape[1], landmarks[LEFT_KNEE].y * frame.shape[0],
                         landmarks[LEFT_KNEE].z]
            left_ankle = [landmarks[LEFT_ANKLE].x * frame.shape[1], landmarks[LEFT_ANKLE].y * frame.shape[0],
                          landmarks[LEFT_ANKLE].z]

            # Right hip, knee and ankle points
            right_hip = [landmarks[RIGHT_HIP].x * frame.shape[1], landmarks[RIGHT_HIP].y * frame.shape[0],
                         landmarks[RIGHT_HIP].z]
            right_knee = [landmarks[RIGHT_KNEE].x * frame.shape[1], landmarks[RIGHT_KNEE].y * frame.shape[0],
                          landmarks[RIGHT_KNEE].z]
            right_ankle = [landmarks[RIGHT_ANKLE].x * frame.shape[1], landmarks[RIGHT_ANKLE].y * frame.shape[0],
                           landmarks[RIGHT_ANKLE].z]

            left_knee_angle = calculate_3d_angle(left_hip, left_knee, left_ankle)
            right_knee_angle = calculate_3d_angle(right_hip, right_knee, right_ankle)

            full_analysis["left_knee_angle_list"].append(angle)
            full_analysis["right_knee_angle_list"].append(angle)

            left_knee_pixel = (
            int(landmarks[LEFT_KNEE].x * frame.shape[1]), int(landmarks[LEFT_KNEE].y * frame.shape[0]))
            right_knee_pixel = (
            int(landmarks[RIGHT_KNEE].x * frame.shape[1]), int(landmarks[RIGHT_KNEE].y * frame.shape[0]))

            # ----------------------------------------------------------------------------------------------



            #---------------------------------- Foot Lift Height ----------------------------------
            left_ankle = results.pose_landmarks.landmark[LEFT_ANKLE]
            right_ankle = results.pose_landmarks.landmark[RIGHT_ANKLE]
            left_hip = results.pose_landmarks.landmark[LEFT_HIP]
            right_hip = results.pose_landmarks.landmark[RIGHT_HIP]

            x1, y1, z1 = left_ankle.x, left_ankle.y, left_ankle.z
            x2, y2, z2 = right_ankle.x, right_ankle.y, right_ankle.z
            y_hip_left = left_hip.y
            y_hip_right = right_hip.y

            left_height = calculate_distance(z1, 0, y1, y_hip_left)
            right_height = calculate_distance(z2, 0, y2, y_hip_right)

            full_analysis["left_foot_lift_heights"].append(angle)
            full_analysis["right_foot_lift_heights"].append(angle)

            x1_pixel, y1_pixel = int(left_ankle.x * frame.shape[1]), int(left_ankle.y * frame.shape[0])
            x2_pixel, y2_pixel = int(right_ankle.x * frame.shape[1]), int(right_ankle.y * frame.shape[0])
            # ----------------------------------------------------------------------------------------------


        cv2.imshow('Full Analysis', frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()


# ----- Apply sliding window for the angle lists
for key in full_analysis.keys():
    full_analysis[key] = extend_with_sliding_window(full_analysis[key])

# ----- Add person informations in the 'full_analysis' dict
full_analysis["Person"] = NAME_SURNAME
full_analysis["Sex"] = SEX
#full_analysis["Parkinson"] = PARKINSON


# ------------- MongoDB Connection -----------
client = MongoClient("mongodb://localhost:27017/")  # MongoDB connection addresses
db = client["Test_Module"]  # Database name
collection = db["gait_test_module"]  # Collection name
collection.insert_one(full_analysis)

print("Data saved to MongoDB successfully!")


