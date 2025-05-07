'''
Bu çalışmada, py-feat modeli kullanılarak videolardan yüz verisi üzerinden Action Unit (AU) 
özellikleri çıkarılmaktadır. Elde edilen AU'lar, yüz kaslarının hareketlerini temsil eden 
66 farklı AU kodunu içerir. Her video için toplam kare (frame) sayısı, video işleme süresi, 
videonun toplam süresi (saniye cinsinden), videodaki bireylerin isimleri ve ilgili etiket 
(label) bilgileri belirlenmektedir.
'''

import os
from feat import Detector
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import time 



# Detector'ı yapılandır
detector = Detector(
    face_model="retinaface",
    landmark_model="mobilefacenet",
    au_model='xgb',
    emotion_model="resmasknet",
    facepose_model="img2pose",
)


# AU ortalamalarını tutacak DataFrame
au_means = pd.DataFrame(columns=['name'])  # Başlangıçta sadece 'name' sütunu var
#tüm aular
au_columns = [
    'AU01', 'AU02', 'AU03', 'AU04', 'AU05', 'AU06', 'AU07', 'AU08', 'AU09', 'AU10', 'AU11', 'AU12',
    'AU13', 'AU14', 'AU15', 'AU16', 'AU17', 'AU18', 'AU19', 'AU20', 'AU21', 'AU22', 'AU23', 'AU24',
    'AU25', 'AU26', 'AU27', 'AU28', 'AU29', 'AU30', 'AU31', 'AU32', 'AU33', 'AU34', 'AU35', 'AU37',
    'AU38', 'AU39', 'AU40', 'AU41', 'AU42', 'AU43', 'AU44', 'AU45', 'AU46', 'AU51', 'AU52', 'AU53',
    'AU54', 'AU55', 'AU56', 'AU57', 'AU58', 'AU61', 'AU62', 'AU63', 'AU64', 'AU65', 'AU66'
]




########################################################################function1 
def extract_name_from_path(video_path):
    # "dataset" kelimesi sonrasındaki kısmı al
    if "dataset" in video_path:
        # "dataset" kelimesinin bulunduğu yeri bul ve sonrasındaki kısmı al
        dataset_index = video_path.index("Data_Hicbirkirpmayok") + len("Data_Hicbirkirpmayok")
        relevant_path = video_path[dataset_index:]

        # "tekayak" kelimesine kadar olan kısmı al
        name_end_index = relevant_path.find("tekayak")
        if name_end_index != -1:
            name = relevant_path[:name_end_index].strip('/')
        else:
            name = relevant_path.strip('/')
    else:
        # Eğer "dataset" yoksa, tüm yolu al (varsayılan durum)
        name = video_path

    return name








########################################################################function2
def process_video(video_path):
    # Video üzerinden AU'ları tahmin et
    predictions = detector.detect_video(video_path, skip_frames=1)  # skip_frames=1 yaptım

    # AU'ların ortalamalarını döndür
    # Önce mevcut AU'ların sütunlarını kontrol et
    au_data = predictions[au_columns] if all(col in predictions.columns for col in au_columns) else predictions

     # Verileri sayısal olmayanlardan ayıklayalım ve NaN'a dönüştürelim
    for column in au_data.columns:
        au_data[column] = pd.to_numeric(au_data[column], errors='coerce')


    # Geriye döndürülecek DataFrame'i oluştur
    return au_data.mean().to_frame().T  # AU'ların ortalama değerini döndür






########################################################################function3
def process_video_with_fallback(video_path, au_columns):
    # Videodan AU'ları işleyelim
    au_means = process_video(video_path)

    # Eksik AU'ları kontrol et ve sıfırla doldur
    for au in au_columns:
        if au not in au_means.columns:
            au_means[au] = 0.0  # Eksik AU için sıfır atama

    # Sıfır atanan AU'lar dahil, sadece mevcut AU'ları döndürelim
    return au_means[au_columns]  # Bu, belirlediğiniz tüm AU'lar için bir DataFrame döndürür.





########################################################################function4
def get_video_info(video_path):
    # VideoCapture ile videoyu aç
    cap = cv2.VideoCapture(video_path)

    # Video açıldı mı?
    if not cap.isOpened():
        print("Video açılamadı!")
        return None

    # Toplam kare sayısını al
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # FPS (Frames Per Second) değerini al
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Süreyi hesapla
    duration = total_frames / fps  # Toplam saniye

    # Video bilgilerini ekrana yazdır
    print(f"Toplam Kare Sayısı: {total_frames}")
    print(f"FPS (Kare/Saniye): {fps}")
    print(f"Süre (saniye): {duration:.2f}")

    # Video kaynağını serbest bırak
    cap.release()

    # Bilgileri döndür
    return total_frames, fps, duration






########################################################################function5
def process_videos_in_batches(video_paths, output_prefix,grup):
    # Timer başlat  her video başında
    start_time = time.perf_counter()
    # Her bir video için tek tek işle
    for i, video_path in enumerate(video_paths):
        df_results = pd.DataFrame()  # Her video için yeni bir DataFrame başlat

        # Video bilgilerini al
        total_frames, fps, duration = get_video_info(video_path)

        # AU ortalamalarını hesapla
        au_means = process_video_with_fallback(video_path, au_columns)



        # Yeni sütunları ekle
        au_means.loc[:, 'total_frame'] = total_frames
        au_means.loc[:, 'fps'] = fps
        au_means.loc[:, 'time'] = duration


        # Timer bitiş
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time


        # Saniye cinsinden süreyi dakika cinsine çevirdim
        elapsed_minutes = elapsed_time / 60


        au_means.loc[:, 'videoPro_timedk'] = round(elapsed_minutes, 3)  # 3 ondalık basamakla yuvarladım
        au_means.loc[:, 'name'] = extract_name_from_path(video_path)



        if(output_prefix=="healthy"):
          au_means.loc[:, 'label'] = 0  # Sağlıklı birey
        else:
          au_means.loc[:, 'label'] = 1  # hasta birey

        # Sonuçları DataFrame'e ekle
        df_results = pd.concat([df_results, au_means], ignore_index=True)

        # Eğer grup boş değilse kaydet
        if not df_results.empty:
            # Çıktı dosyasının adını, sadece output_prefix ve i kullanarak oluştur
            output_excel_path = f'/content/drive/MyDrive/asist_lab_bitirme/dataset_test/{output_prefix}_{grup}_{i}.xlsx'#BU DEĞİŞTİRİLECEKTİR.UYGUN DATASET YOLUNA GÖRE
            df_results.to_excel(output_excel_path, index=False)
            print(f"{output_excel_path} kaydedildi.")






####################################################################################################
################################     finally video processing    ###################################

pathlist=["/content/drive/MyDrive/asist_lab_bitirme/test/parkinson/nihat-karasin-durus-tekayak-trim_YQ3k7X3f.mp4"]#buraya py feat ile  işlenecek video yolları girilir
process_videos_in_batches(pathlist, 'parkinson',0)
















































