from feat import Detector
import pandas as pd
import cv2
import time
import numpy as np

# Configure the detector
detector = Detector(
    face_model="retinaface",
    landmark_model="mobilefacenet",
    au_model='xgb',
    emotion_model="resmasknet",
    facepose_model="img2pose",
)

# AU columns
au_columns = [
    'AU01', 'AU02', 'AU03', 'AU04', 'AU05', 'AU06', 'AU07', 'AU08', 'AU09', 'AU10', 'AU11', 'AU12',
    'AU13', 'AU14', 'AU15', 'AU16', 'AU17', 'AU18', 'AU19', 'AU20', 'AU21', 'AU22', 'AU23', 'AU24',
    'AU25', 'AU26', 'AU27', 'AU28', 'AU29', 'AU30', 'AU31', 'AU32', 'AU33', 'AU34', 'AU35', 'AU37',
    'AU38', 'AU39', 'AU40', 'AU41', 'AU42', 'AU43', 'AU44', 'AU45', 'AU46', 'AU51', 'AU52', 'AU53',
    'AU54', 'AU55', 'AU56', 'AU57', 'AU58', 'AU61', 'AU62', 'AU63', 'AU64', 'AU65', 'AU66'
]


def extract_name_from_path(video_path):
    """Extracts name from the file name"""
    if "dataset" in video_path:
        dataset_index = video_path.index("Data_Hicbirkirpmayok") + len("Data_Hicbirkirpmayok")
        relevant_path = video_path[dataset_index:]
        name_end_index = relevant_path.find("tekayak")
        if name_end_index != -1:
            name = relevant_path[:name_end_index].strip('/')
        else:
            name = relevant_path.strip('/')
    else:
        name = video_path.split('/')[-1].split('.')[0]
    return name


def process_video(video_path):
    """Predict AUs from the video"""
    predictions = detector.detect_video(video_path, skip_frames=100)
    au_data = predictions[au_columns] if all(col in predictions.columns for col in au_columns) else predictions

    # Filter out non-numeric data and convert to NaN
    for column in au_data.columns:
        au_data[column] = pd.to_numeric(au_data[column], errors='coerce')

    return au_data.mean().to_frame().T


def process_video_with_fallback(video_path, au_columns):
    """Process AUs from the video and fill missing AUs with zeros"""
    au_means = process_video(video_path)

    # Check missing AUs and fill with zeros
    for au in au_columns:
        if au not in au_means.columns:
            au_means[au] = 0.0

    return au_means[au_columns]


def get_video_info(video_path):
    """Get video information (total frames, fps, duration)"""
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError("No FACE video recording was found, so no result could be obtained.")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps

    cap.release()
    return total_frames, fps, duration


def feature_engineering(df, df2):
    """Feature engineering operations"""
    # Constant parameters
    params = (1, 1, 1)
    params2 = (1, 1, 2)
    params3 = (1, 1, 1)
    params4 = (1, 1, 1)
    params5 = (2, 7, 1)
    params6 = (1, 5, 2)
    params7 = (1, 1, 1)
    params8 = (1, 1, 2)
    params9 = (1, 1, 1)
    params10 = (1, 1, 1)
    params11 = (1, 1, 1)
    params12 = (1, 1, 3)
    params13 = (1, 6, 1)
    params14 = (1, 1, 2)
    params15 = (1, 1, 1)
    params152 = (1, 1, 1)
    params16 = (1, 1, 1)
    paramstg1 = (1, 1, 1)

    # Calculate new columns
    df['nf1'] = (df['AU01'] * params[0] + df['AU02'] * params[0] + df['AU04'] * params[0]) / params[2]

    df['nf2'] = (df['AU06'] * params2[0] + df['AU09'] * params2[0] + df['AU10'] * params2[0] + df['AU11'] * params2[0] +
                 df['AU12'] * params2[0] + df['AU17'] * params2[0]) / (params2[2] * params2[1])

    df['nf3'] = (((df['AU01'] + df['AU04']) * params3[0]) / (params3[2]))

    df['nf4'] = (params4[0] * (df['AU06'] + df['AU12']) - params4[1] * (df['AU25'] + df['AU26'])) / params4[2]

    df['nf5'] = ((df['AU07'] * params5[0] + df['AU10'] * params5[1] + df['AU11'] * params5[2]) + (
                df['AU24'] * params5[2]))

    df['nf6'] = (df['AU12'] * params6[0] + df['AU17'] * params6[1] + df['AU24'] * params6[2]) / 3

    df['nf7'] = 1 - ((df['AU06'] + df['AU07'] + df['AU43']) / params7[2])

    df['nf8'] = 1 - ((params8[2] * ((df['AU17'] + df['AU24']) * params8[0])) - params8[1] * df['AU28'])

    df['nf9'] = 1 - ((2 * params9[0] * (df['AU10'] + df['AU11'])) + (params9[0] * (df['AU17'] * df['AU24'])) + (
                2 * params9[0] * df['AU02'] * params9[1] * params9[2]))

    df['nf10'] = 1 - ((params10[1] * (-1 * df['AU01']) + df['AU02']) * params10[2] * df['AU04'])

    df['nf11'] = (1 / (1 + df2['time'])) * (df['AU06'] + df['AU12'] / params11[0]) * (
                -1 * (df['AU25'] + df['AU26'] / params11[0]))

    df['nf12'] = ((df['AU12']) + (params12[1] * params12[2] * df['AU17']) + (
                params12[1] * params12[2] * df['AU24'])) / 3

    df['nf13'] = 1 - ((0.3 * params13[1] * ((df['AU04'] + df['AU07']) / 2)) + (
                0.2 * params13[1] * ((df['AU10'] + df['AU17']) / 2)))

    df['nf14'] = 1 - ((params14[1] * params14[2] * df['AU43']))

    df['nf15'] = (df['AU06'] + df['AU07'] + df['AU09'] + df['AU10'] + df['AU11'] + df['AU12'] + df['AU14'] + df[
        'AU15'] + df['AU17'] + df['AU23'] + df['AU24'] + df['AU28'] + df['AU43']) / (13 * df2['time'])

    df['nf152'] = (df['AU06'] + df['AU07'] + df['AU09'] + df['AU10'] + df['AU11'] + df['AU12'] + df['AU14'] + df[
        'AU17'] + df['AU23'] + df['AU24'] + df['AU28'] + df['AU43'] + df['nf2'] + df['nf4'] + df['nf5'] + df['nf6'] +
                   df['nf12'] + df['nf15']) / (13 * df2['time'])

    df['nf16'] = (df['AU05'] + df['AU25'] + df['AU26'] + df['nf7'] + df['nf8'] + df['nf11'] + df['nf13'] + df[
        'nf14']) / (13 * df2['time'])

    df['tg1'] = (np.sin(df['AU05']) + np.sin(df['AU25']) + np.sin(df['AU26']) + np.sin(df['nf7']) + np.sin(
        df['nf8']) + np.sin(df['nf11']) + np.sin(df['nf13']) + np.sin(df['nf14'])) / np.sin((13 * df2['time']))

    return df


def extract_face_features(video_paths):
    """
    Main function: Extracts features from a list of videos
    video_paths: Video paths in list format
    Returns: DataFrame with feature engineering applied
    """
    all_results = pd.DataFrame()

    for video_path in video_paths:
        start_time = time.perf_counter()

        # Get video information
        total_frames, fps, duration = get_video_info(video_path)

        # Calculate AU averages
        au_means = process_video_with_fallback(video_path, au_columns)

        # Add video information
        au_means.loc[:, 'total_frame'] = total_frames
        au_means.loc[:, 'fps'] = fps
        au_means.loc[:, 'time'] = duration

        # Calculate processing time
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        elapsed_minutes = elapsed_time / 60

        au_means.loc[:, 'videoPro_timedk'] = round(elapsed_minutes, 3)
        au_means.loc[:, 'name'] = extract_name_from_path(video_path)

        # Create a copy for feature engineering
        df2 = au_means.copy()

        # Apply feature engineering
        au_means = feature_engineering(au_means, df2)

        # Combine the results
        all_results = pd.concat([all_results, au_means], ignore_index=True)

    return all_results