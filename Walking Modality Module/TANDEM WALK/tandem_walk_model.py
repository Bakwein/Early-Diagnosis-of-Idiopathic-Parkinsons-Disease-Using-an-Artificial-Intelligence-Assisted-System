import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pymongo
import pickle
import json

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, Bidirectional, BatchNormalization
from tensorflow.keras.optimizers import RMSprop, Adam
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from tensorflow.python.keras.callbacks import EarlyStopping

from scipy.interpolate import interp1d

import random
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

#######################################################################################################################
#                                           DATA PREPARATION PART
#######################################################################################################################

# -------------------- MongoDB Connection and Data Retrieval --------------------
client = pymongo.MongoClient("mongodb://localhost:27017/") # MongoDB database connection.
db = client["Gait_Module"] # Name of the MongoDB database.
tandem_walk_collection = db["tandem_walk_analysis"] # Name of the collection in the MongoDB database that contains gait angle data.
# -------------------------------------------------------------------------------


# ----- Data lists
X_list = []  # Will store the time series matrix for each record.
y_list = []  # Parkinson label (0/1).
patient_id_list = []  # Patient ID (can be used for patient-specific separation).


def doc_to_matrix(doc):
    """
    --> All time series lists contained in a single MongoDB record are combined into a single 2D NumPy array.

    --> In each frame (row), all features (columns) received by the function are included together. The format required
        for input to the GRU model is created:        (    TimeSteps   *   NumFeatures    )

    --> For example, let's say we have 3 separate time series (A, B, C). If each has 5 time steps (frames), then:
        A = [A1, A2, A3, A4, A5]
        B = [B1, B2, B3, B4, B5]
        C = [C1, C2, C3, C4, C5]

        When column_stack is applied, it transforms into the following format:
        ⌈ A0 B0 C0 ⌉
        | A1 B1 C1 |
        | A2 B2 C2 |
        | A3 B3 C3 |
        ⌊ A4 B4 C4 ⌋
    """

    left_foot_lift = np.array(doc["left_foot_lift_heights"])
    right_foot_lift = np.array(doc["right_foot_lift_heights"])
    lfa_2d = np.array(doc["lfa_2d"])
    lfa_3d = np.array(doc["lfa_3d"])
    rfa_2d = np.array(doc["rfa_2d"])
    rfa_3d = np.array(doc["rfa_3d"])
    left_knee_angle = np.array(doc["left_knee_angle_list"])
    right_knee_angle = np.array(doc["right_knee_angle_list"])
    hip_angle_list_2d = np.array(doc["hip_angle_list_2d"])
    hip_angle_list_3d = np.array(doc["hip_angle_list_3d"])
    heel_angle_list_2d = np.array(doc["heel_angle_list_2d"])
    heel_angle_list_3d = np.array(doc["heel_angle_list_3d"])

    # All these arrays are combined along the same axis.
    X_temp = np.column_stack([
        left_foot_lift,
        right_foot_lift,
        lfa_2d,
        lfa_3d,
        rfa_2d,
        rfa_3d,
        left_knee_angle,
        right_knee_angle,
        hip_angle_list_2d,
        hip_angle_list_3d,
        heel_angle_list_2d,
        heel_angle_list_3d
    ])
    # X_temp.shape => (1200, number_of_features)
    return X_temp

def parse_parkinson_label(doc):
    """
    The 'Parkinson' field is stored as a string with values "0" or "1".
    This function converts it to an integer value.
    """
    return int(doc["Parkinson"])

def parse_patient_id(doc):
    """
    The patient's identity (e.g., 'Person' or 'PatientName') is retrieved and returned.
    It will be used for patient-specific separation.
    """
    return doc["Person"]



# We retrieve the collection (tandem_walk_collection) containing tandem walk analysis data, which we fetched from MongoDB.
for doc in tandem_walk_collection.find():
    X_temp = doc_to_matrix(doc)
    y_temp = parse_parkinson_label(doc)
    patient_id = parse_patient_id(doc)

    X_list.append(X_temp)
    y_list.append(y_temp)
    patient_id_list.append(patient_id)


# Conversion to NumPy arrays.
X = np.array(X_list)  # X.shape => (total_number_of_records, 1200, number_of_features)
y = np.array(y_list)  # y.shape => (total_number_of_records,)
patient_ids = np.array(patient_id_list)

print("X shape:", X.shape)
print("y shape:", y.shape)


#######################################################################################################################
#                                       SCALING & DATA AUGMENTATION PART
#######################################################################################################################


# -------------------- Function to Convert to Fixed Time Steps --------------------
def rescale_to_fixed_length(data, target_length=1200):
    """
    This function is used to normalize the data back to 1200 time steps
    if the data length changes during the time scaling step (time_scale function).

    It fills in missing or extra time steps using linear interpolation (interp1d).

    :param data: Original Time Series Data
    :param target_length: Target Length (desired scaled size)
    :return: Transformed (alternative version of) the data
    """

    time_steps = np.arange(data.shape[0])
    new_time_steps = np.linspace(0, time_steps.max(), target_length)
    interpolator = interp1d(time_steps, data, axis=0, kind='linear', fill_value="extrapolate")
    return interpolator(new_time_steps)
# -------------------------------------------------------------------------------


# -------------------- Data Augmentation Functions --------------------
def time_shift(data, shift_size=2):
    """
    This function is used to create a different dataset by shifting the time series data by a certain number of steps.

    :param data: Original Time Series Data
    :param shift_size: Shift Step
    :return: Transformed (alternative version of) the data
    """
    return np.roll(data, shift_size, axis=0)


def add_jitter(data, noise_level=0.02):
    """
    This function is used to increase variability by adding noise to the time series data.

    :param data: Original Time Series Data
    :param noise_level: Noise Level
    :return: Transformed (alternative version of) the data
    """

    noise = np.random.normal(0, noise_level, data.shape)
    return data + noise


def time_scale(data, scale_factor=1.2):
    """
    This function is used to generate new data by changing the walking speed
    (either speeding up or slowing down).

    :param data: Original Time Series Data
    :param scale_factor: Factor that changes the scale of the data
    :return: Transformed (alternative version of) the data
    """

    time_steps = np.arange(data.shape[0])
    scaled_steps = np.linspace(0, time_steps.max(), int(time_steps.size * scale_factor))
    interpolator = interp1d(time_steps, data, axis=0, kind='linear', fill_value="extrapolate")
    return rescale_to_fixed_length(interpolator(scaled_steps))


def reverse_time_series(data):
    """
    This function is used to generate new data by reversing the time series.

    :param data: Original Time Series Data
    :return: Transformed (alternative version of) the data
    """

    return np.flip(data, axis=0)
# -------------------------------------------------------------------------------


# Data augmentation application step.
X_augmented = np.concatenate([
    X,
    np.array([time_shift(sample) for sample in X]),
    np.array([add_jitter(sample) for sample in X]),
    np.array([time_scale(sample) for sample in X]),
    np.array([reverse_time_series(sample) for sample in X])
])

y_augmented = np.tile(y, 5)  # Extend the labels according to the number of data samples.

# Data Standardization
scaler = StandardScaler()
X_scaled = np.array([scaler.fit_transform(sample) for sample in X_augmented])

#######################################################################################################################
#                                            MODELING PART
#######################################################################################################################


# Hyperparameter combinations
gru_units_1_options = [64, 128, 256]
gru_units_2_options = [32, 64, 128]
gru_units_3_options = [16, 32, 64]
dense_units_1_options = [32, 64, 128]
dense_units_2_options = [16, 32, 64]
dropout_rate_1_options = [0.3, 0.4, 0.5]
dropout_rate_2_options = [0.3, 0.4, 0.5]
dropout_rate_3_options = [0.2, 0.3, 0.4]
dropout_rate_4_options = [0.1, 0.2, 0.3]
dropout_rate_5_options = [0.1, 0.2, 0.3]
learning_rate_options = [0.0001, 0.0005, 0.001]
optimizer_options = ['rmsprop', 'adam']
batch_size_options = [8, 16, 32]
epochs_options = [50, 75, 100]

# Class Weights Calculation
class_weights = compute_class_weight('balanced', classes=np.unique(y_augmented), y=y_augmented)
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)


# Function for random combination selection
def random_hyperparameter_search(n_iter=20):
    import random
    combinations = []
    for _ in range(n_iter):
        params = {
            'gru_units_1': random.choice(gru_units_1_options),
            'gru_units_2': random.choice(gru_units_2_options),
            'gru_units_3': random.choice(gru_units_3_options),
            'dense_units_1': random.choice(dense_units_1_options),
            'dense_units_2': random.choice(dense_units_2_options),
            'dropout_rate_1': random.choice(dropout_rate_1_options),
            'dropout_rate_2': random.choice(dropout_rate_2_options),
            'dropout_rate_3': random.choice(dropout_rate_3_options),
            'dropout_rate_4': random.choice(dropout_rate_4_options),
            'dropout_rate_5': random.choice(dropout_rate_5_options),
            'learning_rate': random.choice(learning_rate_options),
            'optimizer': random.choice(optimizer_options),
            'batch_size': random.choice(batch_size_options),
            'epochs': random.choice(epochs_options)
        }
        combinations.append(params)
    return combinations


# Model building function
def build_model(input_shape, gru_units_1=128, gru_units_2=64, gru_units_3=32,
                dense_units_1=64, dense_units_2=32, dropout_rate_1=0.4,
                dropout_rate_2=0.4, dropout_rate_3=0.3, dropout_rate_4=0.2,
                dropout_rate_5=0.2, learning_rate=0.0005, optimizer='rmsprop'):
    model = Sequential([
        Bidirectional(GRU(gru_units_1, return_sequences=True), input_shape=input_shape),
        BatchNormalization(),
        Dropout(dropout_rate_1),
        Bidirectional(GRU(gru_units_2, return_sequences=True)),
        BatchNormalization(),
        Dropout(dropout_rate_2),
        GRU(gru_units_3, return_sequences=False),
        BatchNormalization(),
        Dropout(dropout_rate_3),
        Dense(dense_units_1, activation='relu'),
        Dropout(dropout_rate_4),
        Dense(dense_units_2, activation='relu'),
        Dropout(dropout_rate_5),
        Dense(1, activation='sigmoid')
    ])

    # Optimizer selection
    if optimizer == 'rmsprop':
        opt = RMSprop(learning_rate=learning_rate)
    else:
        opt = Adam(learning_rate=learning_rate)

    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model


print("Hyperparameter optimization is starting...")
print("X shape:", X_scaled.shape)
print("y shape:", y_augmented.shape)

# Generate random hyperparameter combinations
param_combinations = random_hyperparameter_search(n_iter=20)
print(f"A total of {len(param_combinations)} parameter combinations will be tested.")

# Variables required for hyperparameter optimization
best_params = None
best_score = 0
all_results = []

# Stratified K-Fold Cross Validation
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=5)

# Evaluate each parameter combination
for i, params in enumerate(param_combinations):
    print(f"\nTesting parameter combination {i + 1}/{len(param_combinations)}:")
    print(params)

    # Metric list for cross-validation
    fold_accuracy_scores = []

    # K-fold cross validation
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_scaled, y_augmented)):
        print(f"Fold {fold + 1}/5")
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y_augmented[train_idx], y_augmented[val_idx]

        # Build the model
        model = build_model(
            input_shape=(X_train.shape[1], X_train.shape[2]),
            gru_units_1=params['gru_units_1'],
            gru_units_2=params['gru_units_2'],
            gru_units_3=params['gru_units_3'],
            dense_units_1=params['dense_units_1'],
            dense_units_2=params['dense_units_2'],
            dropout_rate_1=params['dropout_rate_1'],
            dropout_rate_2=params['dropout_rate_2'],
            dropout_rate_3=params['dropout_rate_3'],
            dropout_rate_4=params['dropout_rate_4'],
            dropout_rate_5=params['dropout_rate_5'],
            learning_rate=params['learning_rate'],
            optimizer=params['optimizer']
        )

        # Train the model
        model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=params['epochs'],
            batch_size=params['batch_size'],
            callbacks=[early_stopping],
            verbose=0,
            class_weight=class_weight_dict
        )

        # Evaluation
        y_pred_prob = model.predict(X_val, verbose=0)
        y_pred = (y_pred_prob > 0.5).astype(int)
        acc = accuracy_score(y_val, y_pred)
        fold_accuracy_scores.append(acc)

        # Clear memory
        tf.keras.backend.clear_session()

    # Average accuracy for this parameter combination
    avg_accuracy = np.mean(fold_accuracy_scores)
    print(f"Average accuracy: {avg_accuracy:.4f}")

    # Save the results
    result = {
        'params': params,
        'accuracy': avg_accuracy
    }
    all_results.append(result)

    # Update the best parameter set
    if avg_accuracy > best_score:
        best_score = avg_accuracy
        best_params = params
        print(f"New best score: {best_score:.4f}")

# Display the best parameters
print("\nBest Parameters:")
print(best_params)
print("Best Score:", best_score)

# ---------- Save all results.

with open("hyperparameter_search_results.json", "w") as f:
    json.dump(all_results, f, indent=4)

# Save the best parameters
with open("best_params.json", "w") as f:
    json.dump(best_params, f, indent=4)

# Lists to store the metrics
accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []
roc_auc_scores = []
conf_matrices = []

best_model = None
best_accuracy = 0.0

print("\nCross-validation is starting with the best parameters...")

# Perform Cross-Validation with the best parameters
for train_idx, val_idx in kf.split(X_scaled, y_augmented):
    X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
    y_train, y_val = y_augmented[train_idx], y_augmented[val_idx]

    # Build the model with the best parameters
    model = build_model(
        input_shape=(X_train.shape[1], X_train.shape[2]),
        gru_units_1=best_params['gru_units_1'],
        gru_units_2=best_params['gru_units_2'],
        gru_units_3=best_params['gru_units_3'],
        dense_units_1=best_params['dense_units_1'],
        dense_units_2=best_params['dense_units_2'],
        dropout_rate_1=best_params['dropout_rate_1'],
        dropout_rate_2=best_params['dropout_rate_2'],
        dropout_rate_3=best_params['dropout_rate_3'],
        dropout_rate_4=best_params['dropout_rate_4'],
        dropout_rate_5=best_params['dropout_rate_5'],
        learning_rate=best_params['learning_rate'],
        optimizer=best_params['optimizer']
    )

    early_stopping = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)

    # Train the model
    history = model.fit(
        X_train, y_train, validation_data=(X_val, y_val),
        epochs=best_params['epochs'],
        batch_size=best_params['batch_size'],
        callbacks=[early_stopping],
        verbose=1,
        class_weight=class_weight_dict
    )

    # Evaluation
    y_pred_prob = model.predict(X_val)
    y_pred = (y_pred_prob > 0.5).astype(int)

    acc = accuracy_score(y_val, y_pred)
    accuracy_scores.append(accuracy_score(y_val, y_pred))
    precision_scores.append(precision_score(y_val, y_pred))
    recall_scores.append(recall_score(y_val, y_pred))
    f1_scores.append(f1_score(y_val, y_pred))
    roc_auc_scores.append(roc_auc_score(y_val, y_pred_prob))
    conf_matrices.append(confusion_matrix(y_val, y_pred))

    # Save the best model
    if acc > best_accuracy:
        best_accuracy = acc
        best_model = model

# Saving the best model
if best_model:
    best_model.save("best_trained_model.h5")
    best_model.save("best_trained_model.keras")
    with open("best_trained_model.pkl", "wb") as f:
        pickle.dump(best_model, f)
    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

#######################################################################################################################
#                                            RESULTS PART
#######################################################################################################################

# Saving all confusion matrices
for i, conf_matrix in enumerate(conf_matrices):
    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Healthy", "Parkinson"],
                yticklabels=["Healthy", "Parkinson"])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix - Fold {i+1}")
    plt.savefig(f"confusion_matrix_fold_{i+1}.png")
    plt.close()

# You can also visualize the average of all confusion matrices if you like.
avg_conf_matrix = np.mean(conf_matrices, axis=0).astype(int)
plt.figure(figsize=(6, 5))
sns.heatmap(avg_conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Healthy", "Parkinson"],
            yticklabels=["Healthy", "Parkinson"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Average Confusion Matrix")
plt.savefig("confusion_matrix_average.png")
plt.show()

# Average Metric Values
print("Avg Accuracy:", np.mean(accuracy_scores))
print("Avg Precision:", np.mean(precision_scores))
print("Avg Recall:", np.mean(recall_scores))
print("Avg F1-Score:", np.mean(f1_scores))
print("Avg ROC-AUC:", np.mean(roc_auc_scores))

with open("metrics_results.txt", "w") as f:
    f.write(f"Avg Accuracy: {np.mean(accuracy_scores):.4f}\n")
    f.write(f"Avg Precision: {np.mean(precision_scores):.4f}\n")
    f.write(f"Avg Recall: {np.mean(recall_scores):.4f}\n")
    f.write(f"Avg F1-Score: {np.mean(f1_scores):.4f}\n")
    f.write(f"Avg ROC-AUC: {np.mean(roc_auc_scores):.4f}\n")
    f.write(f"\nBest Parameters:\n")
    for param, value in best_params.items():
        f.write(f"{param}: {value}\n")
