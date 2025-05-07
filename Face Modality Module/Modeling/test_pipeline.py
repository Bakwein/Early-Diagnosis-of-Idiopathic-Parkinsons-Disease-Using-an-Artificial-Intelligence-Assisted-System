import pandas as pd
import joblib
import numpy as np
from scipy.stats import zscore
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

#!     MODELDE TEST EDİLECEK CSV DOSYASININ YOLU
file_path = r"C:\Users\murat\OneDrive\Masaüstü\bitirme\Modeling\test.csv"

# DataFrame olarak oku
df = pd.read_csv(file_path)

########################    PİPELİNE   ######################
# DataFrame'i rastgele karıştır
df_shuffled = df.sample(frac=1, random_state=12).reset_index(drop=True)

# Sonuçları yazdır
print("Orijinal DataFrame:")
print(df)
print("\nKarıştırılmış DataFrame:")
print(df_shuffled)


###################################  VERİ SETİNDE NaN VE SIFIR DEĞER KONTROLÜ  ##############

def control_dataframe(df_shuffled):
   # NaN sayısını hesapla
    nan_counts = df_shuffled.isnull().sum()

    # Eğer herhangi bir NaN değeri varsa uyarı ver
    if nan_counts.sum() > 0:
        print("⚠️ Uyarı: DataFrame içinde NaN değerler var!")
        print("NaN sayıları:\n", nan_counts)
    else:
        print("✅ DataFrame içinde eksik (NaN) değer bulunmamaktadır.") 
    print("************************************")
        # Sıfır değerlerin sayısını hesapla
    zero_counts = (df_shuffled == 0).sum()

    # 1'e eşit veya 1'den büyük olanları filtrele
    zero_counts_filtered = zero_counts[zero_counts >= 1]

    # Sonucu yazdır
    if not zero_counts_filtered.empty:
        print("⚠️ 1 veya daha fazla sıfır içeren sütunlar:\n", zero_counts_filtered)
    else:
        print("✅ DataFrame içinde sıfır içeren sütun bulunmamaktadır.")   

    print("**********************************************")
    df_shuffled.info() 
    print("**************************************************")
    df_shuffled.describe()   
    print("**********************************************************")
    
control_dataframe(df_shuffled)


################   FEATURE ENGİNEERİNG  ####################
def feature_engineering():
    # Hariç tutmak istediğim sayısal sütunlar
    excluded_columns = ['total_frame', 'fps', 'time', 'videoPro_timedk']

    # İstenmeyen sütunları çıkararak yeni bir DataFrame oluştur
    df_filtered = df_shuffled.drop(columns=excluded_columns, errors='ignore')
    df_filtered_first_67 = df_filtered.iloc[:, :59]  # İlk 67 sütunu al
    # Korumak istediğiniz sütunlar
    keep_columns = ['AU05', 'AU25','AU26','AU06','AU07','AU09','AU10','AU11','AU12','AU14','AU17','AU23','AU24','AU28','AU43','nf2','nf4','nf5','nf6','nf12','nf15','nf7','nf8','nf13','nf14','nf16','label','name']

    # Sadece bu sütunları içeren yeni DataFrame oluştur
    model_data = df[keep_columns]


    # DataFrame'i rastgele karıştır
    model_data = model_data.sample(frac=1, random_state=12).reset_index(drop=True)

    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!          'name' sütununu ayrı bir değişkende sakladım
    name_column = model_data["name"] if "name" in model_data.columns else None

    # 📌 Sadece sayısal sütunları seç
    model_data = model_data.select_dtypes(include=['number'])

    print(model_data)
    print("**************************************************")
    # 📌 O1 özelliğini oluştur ('AU05', 'AU25', 'AU26', 'nf14' MAD)
    model_data.loc[:, 'o1'] = model_data[['AU05', 'AU25', 'AU26', 'nf14']].apply(lambda x: (abs(x - x.mean())).mean(), axis=1)

    # Kullanılan sütunları sil
    model_data.drop(columns=['AU05', 'AU25', 'AU26', 'nf14'], inplace=True)

    # 📌 O2 özelliğini oluştur ('AU06', 'AU09', 'AU10', 'AU43', 'AU12' MAD)
    model_data.loc[:, 'o2'] = model_data[['AU06', 'AU09', 'AU10', 'AU43', 'AU12']].apply(lambda x: (abs(x - x.mean())).mean(), axis=1)

    # Kullanılan sütunları sil
    model_data.drop(columns=['AU06', 'AU09', 'AU10', 'AU43', 'AU12'], inplace=True)

    # Sonuçları yazdır
    print(model_data)
    print("**************************************************")
    # Aykırı değerleri tanımlamak için IQR yöntemini kullan
    Q1 = model_data.quantile(0.25)
    Q3 = model_data.quantile(0.75)
    IQR = Q3 - Q1

    # Aykırı değerlerin bulunduğu yerleri True/False olarak belirle
    is_outlier = (model_data < (Q1 - 1.5 * IQR)) | (model_data > (Q3 + 1.5 * IQR))

    # Aykırı değerleri mean ile doldur
    model_data = model_data.copy()
    for column in model_data.columns:
        mean_value = model_data[column].mean()
        model_data[column] = model_data[column].where(~is_outlier[column], mean_value)

    return name_column,model_data    

def normalization(model_data):
    # "AU" içeren sütunları filtrele
    au_columns = [col for col in model_data.columns if 'AU' in col]

    # Sayısal değeri 1 geçen veya 0'dan az olan var mı kontrol et
    print(model_data[au_columns])
    print("Sayısal değeri 1 geçen veya sıfırdan az var mı kontrol etme")
    print(model_data[au_columns] > 1)

    # Bu sütunlar hariç normalizasyon yapılacaklar
    columns_to_scale = [col for col in model_data.columns if col not in au_columns + ["label"]]
    print(columns_to_scale)
    print("***************************************************")

    # StandardScaler uygula (sadece gerekli sütunlara)
    scaler = StandardScaler()
    model_data[columns_to_scale] = scaler.fit_transform(model_data[columns_to_scale])

    return model_data

name_column,model_data=feature_engineering()
model_data=normalization(model_data)


print("Sayısal değeri 1'den büyük veya 0'dan küçük olan değerlerin kontrolü:")
for col in model_data.columns:
    # Sütunda 1'den büyük veya 0'dan küçük değer var mı kontrol et
    if (model_data[col] > 1).any() or (model_data[col] < 0).any():
        # Aykırı değerlerin bulunduğu satırları yazdır
        outliers = model_data[(model_data[col] > 1) | (model_data[col] < 0)][col]
        print(f"{col} sütununda aykırı değerler (1'den büyük veya 0'dan küçük):")
        print(outliers)
        print("------")

##################   TEST ETME İŞLEMİ   ###############33


# Modeli yükle
model_path = r"C:\Users\murat\OneDrive\Masaüstü\bitirme\bitirme_train\best_model_k5_3.pkl"
loaded_model = joblib.load(model_path)

# X ve y'yi ayır
X_test = model_data.drop(columns=['label'])  # Modelin giriş özellikleri
y_test = model_data['label']  # Gerçek etiketler
# Model ile tahmin yap
y_pred = loaded_model.predict(X_test)


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Modelin doğruluğunu hesapla
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Doğruluğu: {accuracy:.4f}")

# 📊 Classification Report
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

# 🔷 Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=set(y_test), yticklabels=set(y_test))
plt.xlabel("Tahmin Edilen")
plt.ylabel("Gerçek Değer")
plt.title("Confusion Matrix")
plt.show()


################################  OLASILIKSAL OLARAK MODEL TAHMİNLERİ   #####################3

# Modelin tahmin olasılıklarını al
print("ilk örneğin 0 sınıfına ait olma olasılığı ilk sütun ve 1 sınıfına ait olma olasılığı ikinci sütundur ")
y_pred_proba = loaded_model.predict_proba(X_test)


for index in range(len(y_pred_proba)):
    print(f"{y_pred_proba[index]}  ve  {name_column[index]}")
    print("-------------------")











