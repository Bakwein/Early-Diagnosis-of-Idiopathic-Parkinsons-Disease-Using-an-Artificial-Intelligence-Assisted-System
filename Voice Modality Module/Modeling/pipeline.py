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

print("1. Feature extraction starting...")

def extract_features(file_name):
    y, sr = librosa.load(file_name, sr=44100) #44.1 khz
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
label_dict = {"healthy": 0, "parkinson": 1}
all_data = []
for base_dir in base_dirs:
    for person_folder in os.listdir(base_dir):
        person_path = os.path.join(base_dir, person_folder)
        if os.path.isdir(person_path):
            for root, _, files in os.walk(person_path):
                for file in files:
                    if file.endswith("n.mp3"):
                        file_path = os.path.join(root, file)
                        features = extract_features(file_path)
                        name = person_folder
                        all_data.append(np.hstack((features, label_dict[base_dir], name)))

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

df = pd.DataFrame(all_data, columns=columns)
df.to_csv('features_pipelined.csv', index=False)
print("Feature extraction completed.\n")

print("2. Data augmentation starting...")

def add_gaussian_noise(data, noise_level=0.02):
    noise = np.random.normal(0, noise_level, data.shape)
    return data + noise

df = pd.read_csv("features_pipelined.csv")
augmented_samples = []
label_list = []
for i in range(len(df)):
    original = df.iloc[i, :-2].values.astype(float)
    label = df.iloc[i, -2]
    noisy_sample = add_gaussian_noise(original)
    augmented_samples.extend([original, noisy_sample])
    label_list.extend([label, label])

X_augmented = np.array(augmented_samples)
y_augmented = np.array(label_list)
smote = SMOTE(sampling_strategy="auto", random_state=42)
X_final, y_final = smote.fit_resample(X_augmented, y_augmented)
columns = df.columns[:-2]
df_final = pd.DataFrame(X_final, columns=columns)
df_final["label"] = y_final

df_final.to_csv("features_pipelined_augmented.csv", index=False)
print("Data augmentation completed.\n")

print("3. Model training starting...")
df_model = pd.read_csv("features_pipelined_augmented.csv")
X = df_model.drop(columns=["label", "mfcc_6_mean", "mfcc_11_mean"]).values # "mfcc_6_mean" and "mfcc_11_mean" were removed based on feature importance.
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
print(f"\nBest parameters: {grid_search.best_params_}")
cv_scores = cross_val_score(best_model, X_train, y_train, cv=cv, scoring="accuracy")
print(f"\nCV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

best_model.fit(X_train, y_train)
y_train_pred = best_model.predict(X_train)
y_test_pred = best_model.predict(X_test)
print(f"\nTraining Accuracy: {accuracy_score(y_train, y_train_pred):.4f}")
print("\nTraining Classification Report:\n", classification_report(y_train, y_train_pred, digits=4))
print(f"\nTest Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
print("\nTest Classification Report:\n", classification_report(y_test, y_test_pred, digits=4))
print(f"\nTrain Log Loss: {log_loss(y_train, best_model.predict_proba(X_train)):.4f}")
print(f"Test Log Loss: {log_loss(y_test, best_model.predict_proba(X_test)):.4f}")

cm = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Healthy", "Parkinson"], yticklabels=["Healthy", "Parkinson"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Test Set Confusion Matrix")
plt.show()

joblib.dump(best_model, 'random_forest_pipelined_model.h5')
joblib.dump(scaler, 'pipeline_scaler.pkl')
print("Model and scaler saved successfully.\n")

print("4. Model testing starting...")
model = joblib.load("random_forest_pipelined_model.h5")
scaler = joblib.load("pipeline_scaler.pkl")
df_test = pd.read_csv("features_pipelined.csv")
X = df_test.drop(columns=["label", "name", "mfcc_6_mean", "mfcc_11_mean"]).values
y_true = df_test["label"].values
names = df_test["name"].values
X_scaled = scaler.transform(X)
probabilities = model.predict_proba(X_scaled)
predictions = model.predict(X_scaled)
print("\n------ TEST RESULTS ------")
for name, true_label, pred_label, prob in zip(names, y_true, predictions, probabilities):
    label_str = "Parkinson" if pred_label == 1 else "Healthy"
    correctness = "✅ Correct" if pred_label == true_label else "❌ Incorrect"
    probability = prob[int(pred_label)]
    print(f"{name}: Prediction = {label_str} (Probability: {probability:.4f}) → {correctness}")
