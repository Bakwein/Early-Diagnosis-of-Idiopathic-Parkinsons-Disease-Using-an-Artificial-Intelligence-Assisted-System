import pandas as pd
import joblib
import numpy as np

# 1. Model ve scaler'ı yükle
model = joblib.load("random_forest_pipelined_model.pkl")
scaler = joblib.load("pipeline_scaler.pkl")

# 2. Veriyi yükle
df = pd.read_csv("features_pipeline.csv")

# 3. Özellikleri, etiketleri ve 'name' bilgisini ayır
X = df.drop(columns=["label", "name", "mfcc_6_mean", "mfcc_11_mean"]).values
y_true = df["label"].values
names = df["name"].values

# 4. Ölçeklendirme
X_scaled = scaler.transform(X)

# 5. Tahmin olasılıkları ve sınıflar
probabilities = model.predict_proba(X_scaled)
predictions = model.predict(X_scaled)

# 6. Sonuçları yazdır
print("------ TEST SONUÇLARI ------")
for name, true_label, pred_label, prob in zip(names, y_true, predictions, probabilities):
    etikete_cevir = "Parkinson" if pred_label == 1 else "Sağlıklı"
    dogruluk = "✅ Doğru" if pred_label == true_label else "❌ Yanlış"
    olasilik = prob[pred_label]
    print(f"{name}: Tahmin = {etikete_cevir} (Olasılık: {olasilik:.4f}) → {dogruluk}")
