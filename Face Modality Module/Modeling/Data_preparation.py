'''
In this study, the py-feat model is used to generate Action Units (AU) based on face data from videos. 
features are extracted. The resulting AUs represent the movements of the facial muscles. 
It contains 66 different AU codes. Total number of frames for each video, video processing time, 
the total duration of the video (in seconds), the names of the individuals in the video and the relevant tag 
(label) information is determined.
'''

import os
from feat import Detector
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import time 



# Configure Detector
detector = Detector(
    face_model="retinaface",
    landmark_model="mobilefacenet",
    au_model='xgb',
    emotion_model="resmasknet",
    facepose_model="img2pose",
)



au_means = pd.DataFrame(columns=['name'])  
#all AU
au_columns = [
    'AU01', 'AU02', 'AU03', 'AU04', 'AU05', 'AU06', 'AU07', 'AU08', 'AU09', 'AU10', 'AU11', 'AU12',
    'AU13', 'AU14', 'AU15', 'AU16', 'AU17', 'AU18', 'AU19', 'AU20', 'AU21', 'AU22', 'AU23', 'AU24',
    'AU25', 'AU26', 'AU27', 'AU28', 'AU29', 'AU30', 'AU31', 'AU32', 'AU33', 'AU34', 'AU35', 'AU37',
    'AU38', 'AU39', 'AU40', 'AU41', 'AU42', 'AU43', 'AU44', 'AU45', 'AU46', 'AU51', 'AU52', 'AU53',
    'AU54', 'AU55', 'AU56', 'AU57', 'AU58', 'AU61', 'AU62', 'AU63', 'AU64', 'AU65', 'AU66'
]




########################################################################function1 
def extract_name_from_path(video_path):
    
    if "dataset" in video_path:
        
        dataset_index = video_path.index("Data_Hicbirkirpmayok") + len("Data_Hicbirkirpmayok")
        relevant_path = video_path[dataset_index:]

        # Take the part up to the word “unipod”, it depends on the file name you keep.
        name_end_index = relevant_path.find("tekayak")
        if name_end_index != -1:
            name = relevant_path[:name_end_index].strip('/')
        else:
            name = relevant_path.strip('/')
    else:
        
        name = video_path

    return name








########################################################################function2
def process_video(video_path):
    
    predictions = detector.detect_video(video_path, skip_frames=1)  # skip_frames=1 


    au_data = predictions[au_columns] if all(col in predictions.columns for col in au_columns) else predictions

     
    for column in au_data.columns:
        au_data[column] = pd.to_numeric(au_data[column], errors='coerce')


    
    return au_data.mean().to_frame().T 






########################################################################function3
def process_video_with_fallback(video_path, au_columns):
    
    au_means = process_video(video_path)

    
    for au in au_columns:
        if au not in au_means.columns:
            au_means[au] = 0.0  

    
    return au_means[au_columns]  





########################################################################function4
def get_video_info(video_path):
    
    cap = cv2.VideoCapture(video_path)

    
    if not cap.isOpened():
        print("Video açılamadı!")
        return None

   
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # FPS (Frames Per Second) 
    fps = cap.get(cv2.CAP_PROP_FPS)

    # time calculate
    duration = total_frames / fps  

    
    print(f"Toplam Kare Sayısı: {total_frames}")
    print(f"FPS (Kare/Saniye): {fps}")
    print(f"Süre (saniye): {duration:.2f}")

    
    cap.release()


    return total_frames, fps, duration






########################################################################function5
def process_videos_in_batches(video_paths, output_prefix,grup):
    # Start timer at the beginning of each video
    start_time = time.perf_counter()
   
    for i, video_path in enumerate(video_paths):
        df_results = pd.DataFrame()  

        
        total_frames, fps, duration = get_video_info(video_path)

        
        au_means = process_video_with_fallback(video_path, au_columns)



        # new columns add
        au_means.loc[:, 'total_frame'] = total_frames
        au_means.loc[:, 'fps'] = fps
        au_means.loc[:, 'time'] = duration


        # Timer finish
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time


        # I converted the time in seconds to minutes
        elapsed_minutes = elapsed_time / 60


        au_means.loc[:, 'videoPro_timedk'] = round(elapsed_minutes, 3)  
        au_means.loc[:, 'name'] = extract_name_from_path(video_path)



        if(output_prefix=="healthy"):
          au_means.loc[:, 'label'] = 0  # Healthy individual
        else:
          au_means.loc[:, 'label'] = 1  # sick individual

      
        df_results = pd.concat([df_results, au_means], ignore_index=True)

        
        if not df_results.empty:
           
            output_excel_path = f'/content/drive/MyDrive/asist_lab_bitirme/dataset_test/{output_prefix}_{grup}_{i}.xlsx'#THIS WILL BE CHANGED ACCORDING TO THE APPROPRIATE DATASET PATH
            df_results.to_excel(output_excel_path, index=False)
            print(f"{output_excel_path} kaydedildi.")






####################################################################################################
################################     finally video processing    ###################################

pathlist=["/content/drive/MyDrive/asist_lab_bitirme/test/parkinson/nihat-karasin-durus-tekayak-trim_YQ3k7X3f.mp4"]#Enter the video paths to be processed with py feat here
process_videos_in_batches(pathlist, 'parkinson',0)
















































