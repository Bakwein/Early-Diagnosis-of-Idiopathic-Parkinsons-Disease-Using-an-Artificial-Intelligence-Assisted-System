import os
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib

print("1. Özellik çıkarımı başlıyor...")

def extract_features(file_name):
    y, sr = librosa.load(file_name, sr=44100)
    features = []
    features.append(np.mean(librosa.feature.zero_crossing_rate(y=y).T, axis=0))
    features.append(np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0))
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    features.extend([np.mean(mfcc.T, axis=0),
                     np.mean(librosa.feature.delta(mfcc).T, axis=0),
                     np.mean(librosa.feature.delta(mfcc, order=2).T, axis=0)])
    features.append(np.mean(librosa.feature.rms(y=y).T, axis=0))
    features.append(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr).T, axis=0))
    features.append(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr).T, axis=0))
    features.append(np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0))
    features.append(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr).T, axis=0))
    features.append(np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr).T, axis=0))
    return np.hstack(features)

base_dirs = ["healthy", "parkinson"]
labels = {"healthy": 0, "parkinson": 1}
data = []
for base_dir in base_dirs:
    for patient_folder in os.listdir(base_dir):
        patient_path = os.path.join(base_dir, patient_folder)
        if os.path.isdir(patient_path):
            for root, _, files in os.walk(patient_path):
                for file in files:
                    if file.endswith("n.mp3"):
                        file_path = os.path.join(root, file)
                        features = extract_features(file_path)
                        name = patient_folder
                        data.append(np.hstack((features, labels[base_dir], name)))

columns = ["zero_crossing_rate"] + \
          [f"chroma_stft_{i}" for i in range(1, 13)] + \
          [f"mfcc_{i}_mean" for i in range(1, 21)] + \
          [f"mfcc_{i}_delta" for i in range(1, 21)] + \
          [f"mfcc_{i}_delta2" for i in range(1, 21)] + \
          ["rmse", "spectral_centroid", "spectral_bandwidth"] + \
          [f"spectral_contrast_{i}" for i in range(1, 8)] + \
          ["spectral_rolloff"] + \
          [f"tonnetz_{i}" for i in range(1, 7)] + \
          ["label", "name"]

df = pd.DataFrame(data, columns=columns)
df.to_csv('features_pipeline.csv', index=False)
print("Öznitelik çıkarımı tamamlandı.\n")

print("2. Veri artırımı başlıyor...")

def add_gaussian_noise(data, noise_level=0.02):
    noise = np.random.normal(0, noise_level, data.shape)
    return data + noise

df = pd.read_csv("features_pipeline.csv")
augmented_data = []
labels_list = []
for i in range(len(df)):
    original = df.iloc[i, :-2].values.astype(float)
    label = df.iloc[i, -2]
    noisy_sample = add_gaussian_noise(original)
    augmented_data.extend([original, noisy_sample])
    labels_list.extend([label, label])

X_augmented = np.array(augmented_data)
y_augmented = np.array(labels_list)
smote = SMOTE(sampling_strategy="auto", random_state=42)
X_final, y_final = smote.fit_resample(X_augmented, y_augmented)
columns = df.columns[:-2]
df_final = pd.DataFrame(X_final, columns=columns)
df_final["label"] = y_final

df_final.to_csv("features_pipeline_augmented.csv", index=False)
print("Veri artırımı tamamlandı.\n")

print("3. Model eğitimi başlıyor...")
df_model = pd.read_csv("features_pipeline_augmented.csv")
X = df_model.drop(columns=["label", "mfcc_6_mean", "mfcc_11_mean"]).values
y = df_model["label"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=cv, scoring="accuracy", n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
print(f"\nEn iyi parametreler: {grid_search.best_params_}")
cv_scores = cross_val_score(best_model, X_train, y_train, cv=cv, scoring="accuracy")
print(f"\nCV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

plt.figure(figsize=(8,4))
sns.boxplot(x=cv_scores)
plt.title('Cross-validation Accuracy Dağılımı')
plt.xlabel('Accuracy')
plt.show()

best_model.fit(X_train, y_train)
y_train_pred = best_model.predict(X_train)
y_test_pred = best_model.predict(X_test)
print(f"\nEğitim Accuracy: {accuracy_score(y_train, y_train_pred):.4f}")
print("\nEğitim Classification Report:\n", classification_report(y_train, y_train_pred, digits=4))
print(f"\nTest Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
print("\nTest Classification Report:\n", classification_report(y_test, y_test_pred, digits=4))
print(f"\nTrain Log Loss: {log_loss(y_train, best_model.predict_proba(X_train)):.4f}")
print(f"Test Log Loss: {log_loss(y_test, best_model.predict_proba(X_test)):.4f}")

cm = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Sağlıklı", "Parkinson"], yticklabels=["Sağlıklı", "Parkinson"])
plt.xlabel("Tahmin Edilen")
plt.ylabel("Gerçek Değer")
plt.title("Test Seti Confusion Matrix")
plt.show()

joblib.dump(best_model, 'random_forest_pipelined_model.pkl')
joblib.dump(scaler, 'pipeline_scaler.pkl')
print("Model ve scaler başarıyla kaydedildi.\n")

print("4. Model test işlemi başlıyor...")
model = joblib.load("random_forest_pipelined_model.pkl")
scaler = joblib.load("pipeline_scaler.pkl")
df_test = pd.read_csv("features_pipeline.csv")
X = df_test.drop(columns=["label", "name", "mfcc_6_mean", "mfcc_11_mean"]).values
y_true = df_test["label"].values
names = df_test["name"].values
X_scaled = scaler.transform(X)
probabilities = model.predict_proba(X_scaled)
predictions = model.predict(X_scaled)
print("\n------ TEST SONUÇLARI ------")
for name, true_label, pred_label, prob in zip(names, y_true, predictions, probabilities):
    etikete_cevir = "Parkinson" if pred_label == 1 else "Sağlıklı"
    dogruluk = "✅ Doğru" if pred_label == true_label else "❌ Yanlış"
    olasilik = prob[int(pred_label)]
    print(f"{name}: Tahmin = {etikete_cevir} (Olasılık: {olasilik:.4f}) → {dogruluk}")