# Early Diagnosis of Idiopathic Parkinson's Disease Using an AI-Assisted System

---

## Citation

> This repository accompanies a peer-reviewed publication. If you use this code, data, or methodology in your research, please cite the following work:

```bibtex
@article{tacsyurek2026novel,
  title={A novel multimodal AI framework for early diagnosis of idiopathic Parkinson’s disease},
  author={Ta{\c{s}}y{\"u}rek, Efe Y{\i}lmaz and Altun, {\c{S}}aban Murat and Uncu, Ata Emir and Tunca, Sefa and Omurca, Sevin{\c{c}} {\.I}lhan and Pehlivano{\u{g}}lu, Meltem Kurt and Alag{\"o}z, Aybala Neslihan and Kalkan, O{\u{g}}ulcan},
  journal={Medical \& Biological Engineering \& Computing},
  pages={1--23},
  year={2026},
  publisher={Springer}
}
```

> **Note to readers:** As this article has been officially published, the authors kindly request that any academic work, derivative project, or software using this system cites the reference above. Proper attribution supports continued open research in AI-assisted Parkinson's diagnostics.

---

## Overview

This project implements a **multi-modal, AI-assisted clinical decision support system** for the early diagnosis of idiopathic Parkinson's disease. Instead of relying on a single biomarker, the system integrates five independent modality pipelines — gait analysis, facial action unit analysis, voice analysis, postural analysis, and tandem walk analysis — into a unified probability score delivered through a web-based interface.

Each modality captures a distinct set of Parkinson's-related motor or neurological symptoms:

| Modality | Symptom Targeted | Model Type |
|---|---|---|
| Normal Walking (Gait) | Gait abnormalities, step asymmetry | Bidirectional GRU |
| Tandem Walking | Balance and coordination deficits | Bidirectional GRU |
| Face | Facial hypomimia (reduced expression) | XGBoost |
| Voice | Dysphonia, tremor in speech | Random Forest |
| Posture | Stooped / flexed posture | Sklearn classifier |

The five modality scores are fused into a single composite score, and the result is presented to the clinician via a **Gradio** web UI.

---

## Repository Structure

```
├── Walking Modality Module/
│   ├── Keypoint Extraction/
│   │   └── full_keypoint_extraction.py
│   ├── GAIT/
│   │   ├── gait_model.py
│   │   ├── gait_test_script.py
│   │   ├── best_params.json
│   │   ├── hyperparameter_search_results.json
│   │   └── metrics_results.txt
│   └── TANDEM WALK/
│       ├── tandem_walk_model.py
│       ├── tandem_walk_test_script.py
│       ├── best_params.json
│       ├── hyperparameter_search_results.json
│       └── metrics_results.txt
│
├── Face Modality Module/
│   ├── requirements.txt
│   └── Modeling/
│       ├── Data_preparation.py
│       ├── Data_generation_and_scaling_pipeline.py
│       └── test_pipeline.py
│
├── Voice Modality Module/
│   ├── test_pipeline_result.txt
│   └── Modeling/
│       ├── pipeline.py
│       ├── test.py
│       ├── features_pipelined.csv
│       └── features_pipelined_augmented.csv
│
├── Posture Modality Module/
│   ├── posture_module_mediapipe_feature_extraction.py
│   ├── pipeline_posture_py.py
│   ├── pipeline_posture_noscale.py
│   ├── tsfel.ipynb
│   ├── posture_modelity_model_training.ipynb
│   └── pipeline_posture.ipynb
│
└── System Integration/
    ├── app.py
    ├── full_keypoint_extraction.py
    ├── voice_feature_extraction.py
    ├── posture_feature_extraction.py
    ├── face_feature_extraction.py
    ├── gait_test_script.py
    ├── tandem_walk_test_script.py
    ├── voice_test_script.py
    ├── posture_test_script.py
    └── face_test_script.py
```

---

## Walking Modality Module

### Overview

The walking modality consists of two sub-modules: **Normal Gait** analysis and **Tandem Walk** analysis. Both sub-modules share the same feature extraction pipeline but differ in the walking task performed and the resulting model architecture and performance.

The patient is recorded with a camera (optionally an Intel RealSense depth camera for true 3D joint angles) while performing the walking task. Skeletal keypoints are extracted per frame, then biomechanical angles are computed and fed into a deep sequential model.

### Keypoint Extraction

**File:** `Keypoint Extraction/full_keypoint_extraction.py`

- Uses **MediaPipe Pose** to detect and track lower-body skeletal landmarks from video.
- Tracked landmark indices: hips (23, 24), knees (25, 26), ankles (27, 28), heels (29, 30), foot tips (31, 32).
- Computes the following biomechanical angles per frame:
  - Left/right **foot angle** (2D and 3D)
  - Left/right **knee angle**
  - Left/right **hip angle**
  - Left/right **heel angle**
  - Left/right **foot lift height**
- Extracted time series (12 features × T timesteps) are stored in **MongoDB** under:
  - Normal gait → `Gait_Module.gait_analysis`
  - Tandem walk → `Gait_Module.tandem_walk_analysis`

> **Hardware note:** The system optionally supports an **Intel RealSense** depth camera (`pyrealsense2`) to obtain true 3D skeletal coordinates, improving the accuracy of 3D angle calculations over depth-estimated angles from a standard RGB camera.

### Normal Gait Model

**File:** `GAIT/gait_model.py` | **Test script:** `GAIT/gait_test_script.py`

**Input:** Variable-length time series from MongoDB, resampled to a fixed sequence length via linear interpolation, then standardized per feature with `StandardScaler`.

**Architecture:** Bidirectional GRU (3 stacked layers) with fully connected head:

```
Input → BiGRU(256) → BatchNorm → Dropout
      → BiGRU(64)  → BatchNorm → Dropout
      → BiGRU(16)  → BatchNorm → Dropout
      → Dense(32, ReLU) → Dropout
      → Dense(64, ReLU) → Dropout
      → Dense(1, Sigmoid)
```

---

**Saved model files:**
- `best_trained_walking_model.h5` — Keras HDF5 format
- `best_trained_walking_model.keras` — Keras native format
- `best_trained_walking_model.pkl` — Pickle format (used by integration layer)

### Tandem Walk Model

**File:** `TANDEM WALK/tandem_walk_model.py` | **Test script:** `TANDEM WALK/tandem_walk_test_script.py`

The tandem walk task requires the patient to walk heel-to-toe along a straight line, stressing balance and coordination — symptoms that manifest distinctively in Parkinson's patients.

**Architecture:** Bidirectional GRU (3 stacked layers, smaller capacity than normal gait):

```
Input → BiGRU(64) → BatchNorm → Dropout
      → BiGRU(32) → BatchNorm → Dropout
      → BiGRU(16) → BatchNorm → Dropout
      → Dense(1, Sigmoid)
```

---

## Face Modality Module

### Overview

The face modality targets **facial hypomimia** — the reduced spontaneous facial expressivity that is a hallmark of Parkinson's disease. The patient is recorded in a standardized frontal-facing video. Facial Action Units (AUs) are extracted per frame using a deep learning-based facial analysis framework, then combined into engineered composite features that are fed into a trained XGBoost classifier.

The face module runs in a **dedicated Python virtual environment** (`face_env`) to isolate the heavy `py-feat` dependency stack from the rest of the system.

### Feature Extraction

**File:** `Modeling/Data_preparation.py` | Integration: `face_feature_extraction.py`

Uses **py-feat** (`Detector` class) with the following sub-models:

| Sub-task | Model |
|---|---|
| Face detection | `retinaface` |
| Facial landmarks | `mobilefacenet` |
| Action Units | `xgb` |
| Emotion recognition | `resmasknet` |
| Head pose estimation | `img2pose` |

- Extracts **66 Facial Action Units (AU01 – AU66)** per frame.
- Frames are sampled with `skip_frames=100` during inference to reduce compute cost.
- AU values are averaged across all valid frames to produce a single feature vector per video.

### Feature Engineering

**File:** `Modeling/Data_generation_and_scaling_pipeline.py`

Raw AU values are combined into **18+ engineered composite features** (`nf1` – `nf18`, `tg1`) via weighted linear combinations of anatomically related AU groups. Example groupings:

- **Brow raise** (inner/outer): combines AU01, AU02
- **Cheek & lid raise** / **smile**: combines AU06, AU12, AU14
- **Lip movements**: combines AU20, AU23, AU24, AU25, AU26

This compression step encodes clinically meaningful facial movement patterns and reduces feature dimensionality before classification.

### Model

**File:** `Modeling/test_pipeline.py` | Integration: `face_test_script.py`

- Loaded from `best_trained_face_model.pkl` (XGBoost-based, serialized via `joblib`).
- Inference pipeline:
  1. Extract raw AUs from video frames.
  2. Compute compound features (`o1`, `o2` groups).
  3. Apply IQR-based outlier clipping.
  4. Predict with the XGBoost classifier; return `predict_proba` score.

---

## Voice Modality Module

### Overview

The voice modality captures **dysphonia** — the vocal impairments (tremor, breathiness, reduced volume, irregular pitch) associated with the laryngeal and respiratory muscle dysfunction in Parkinson's disease. A short voice recording is processed by a librosa-based feature extraction pipeline and classified by a trained Random Forest model.

### Feature Extraction

**File:** `Modeling/pipeline.py` | Integration: `voice_feature_extraction.py`

Audio files (`.mp3`) are loaded at **44.1 kHz**. The following feature groups are extracted:

| Feature Group | Count |
|---|---|
| Zero Crossing Rate | 1 |
| Chroma STFT | 12 |
| MFCC (mean) | 20 |
| MFCC delta (mean) | 20 |
| MFCC delta² (mean) | 20 |
| RMS Energy | 1 |
| Spectral Centroid | 1 |
| Spectral Bandwidth | 1 |
| Spectral Contrast | 7 |
| Spectral Rolloff | 1 |
| Tonnetz | 6 |
| **Total** | **~90–100** |

Extracted features are exported to `features_pipelined.csv` and an augmented version `features_pipelined_augmented.csv`.

### Model

**File:** `Modeling/test.py` | Integration: `voice_test_script.py`

- Model: **Random Forest**, serialized as `best_trained_voice_model.h5` via `joblib`.
- Two MFCC features (`mfcc_6_mean`, `mfcc_11_mean`) are excluded during inference.
- Input is scaled using the saved `voice_scaler.pkl` before prediction.

---

## Posture Modality Module

### Overview

The posture modality targets the **stooped, forward-flexed posture** characteristic of Parkinson's disease — a result of rigidity and loss of postural reflexes. The patient is recorded standing still or moving gently, and several postural angles relative to the vertical axis are extracted over time using MediaPipe Pose. TSFEL is then used to extract a rich set of time-series statistical and spectral features from each angle signal.

### Feature Extraction

**File:** `posture_module_mediapipe_feature_extraction.py` | Integration: `posture_feature_extraction.py`

**Postural angles measured (2D and 3D):**

| Signal | Description |
|---|---|
| `Head_Tilt` | Lateral head inclination from vertical |
| `Neck_Inclination` | Neck forward bend from vertical |
| `Shoulder_Line_Angle` | Tilt of the shoulder girdle |
| `Torso_Inclination` | Forward lean of the torso |

For each signal, **TSFEL** (Time Series Feature Extraction Library) computes statistical, spectral, and temporal features, including:
- MFCC and LPCC coefficients
- Spectrogram coefficients
- Spectral entropy and energy
- IQR, turning points, median absolute deviation
- Autocorrelation-based features

### Model

**File:** Integration `posture_test_script.py` | Training: `posture_modelity_model_training.ipynb`

- 30 features are selected from the full TSFEL output, dominated by head tilt, shoulder line angle, and neck inclination signals in both 2D and 3D.
- Model loaded from `best_trained_posture_model.pkl` (sklearn-compatible, returns `predict_proba`).
- Training workflow (data preparation → feature extraction → model selection → evaluation) is documented in the Jupyter notebooks:
  - `tsfel.ipynb` — feature engineering exploration
  - `posture_modelity_model_training.ipynb` — model training and evaluation
  - `pipeline_posture.ipynb` — full pipeline assembly

---

## System Integration

### Overview

The system integration layer combines all five modality pipelines into a single clinical decision support application served via a **Gradio** web UI. Each modality runs independently and in parallel; their probability scores are fused into a composite diagnosis score.

**Entry point:** `System Integration/app.py`

### Architecture

```
Clinician Input (Gradio UI)
        │
        ├─ Video1_Walking.mp4  ──► Gait Pipeline       ─► P(Parkinson) ─┐
        ├─ Video2_Tandem.mp4   ──► Tandem Walk Pipeline ─► P(Parkinson) ─┤
        ├─ Video3_Face.mp4     ──► Face Pipeline        ─► P(Parkinson) ─┤─► Mean Score ──► Diagnosis
        ├─ Video4_Posture.mp4  ──► Posture Pipeline     ─► P(Parkinson) ─┤
        └─ Voice.mp3           ──► Voice Pipeline       ─► P(Parkinson) ─┘
                                                                         │
                                                              MongoDB (feedback store)
```

### Workflow

1. The clinician enters **patient name** and **patient ID**, then uploads up to five files (any subset is valid; unavailable modalities are skipped).
2. Uploaded files are cached in a local `Cache/` directory.
3. On clicking **"Analyze"**, all available modalities are launched **in parallel** via `ThreadPoolExecutor` (max 5 workers).
4. The **Face modality** runs in a **separate subprocess** using the `face_env` virtual environment to isolate `py-feat` dependencies.
5. Each pipeline returns `{"label": "Parkinson" | "Healthy", "probability": float}`.
6. The final composite score is computed as the **equal-weight average** of all available modality probabilities × 100.
7. The result, along with the patient details and individual modality scores, is stored in **MongoDB** (`Parkinson_Diagnosis_Feedback.Feed_Back`) for clinical feedback and audit.
8. An **"Exit"** button clears the cache and terminates the process cleanly.

### Parallel Execution

| Modality | Execution Context | Model Format |
|---|---|---|
| Gait (Normal Walking) | ThreadPoolExecutor worker | Keras `.pkl` |
| Tandem Walk | ThreadPoolExecutor worker | Keras `.h5` |
| Voice | ThreadPoolExecutor worker | joblib Random Forest |
| Posture | ThreadPoolExecutor worker | joblib sklearn |
| Face | Subprocess (`face_env`) | joblib XGBoost |

### Score Fusion

The final composite score is calculated as:

```
Final Score (%) = mean([P_gait, P_tandem, P_face, P_voice, P_posture]) × 100
```

Only modalities for which a video/audio file was provided are included in the mean. This allows partial assessments when not all recordings are available.

---

## Installation

### Prerequisites

- Python 3.9+
- MongoDB running locally (default port 27017)
- *(Optional)* Intel RealSense SDK for 3D depth-based keypoint extraction

### Setup

```bash
# Clone the repository
git clone https://github.com/FILL_IN/FILL_IN.git
cd Early-Diagnosis-of-Idiopathic-Parkinsons-Disease

# Install dependencies
pip install -r "Face Modality Module/requirements.txt"
```

For the face modality, create a dedicated virtual environment:

```bash
python -m venv face_env
face_env\Scripts\activate       # Windows
pip install py-feat==0.6.2
```

### Running the Application

```bash
cd "System Integration"
python app.py
```

The Gradio web UI will open automatically in your default browser.

---

## Model Files

The trained model files are **not included** in this repository due to size constraints. They must be trained locally using the provided training scripts and notebooks, or obtained from the authors upon request.

| Modality | Expected File | Location |
|---|---|---|
| Gait | `best_trained_walking_model.pkl` | `Walking Modality Module/GAIT/` |
| Tandem Walk | `best_trained_model.h5`, `scaler.pkl` | `Walking Modality Module/TANDEM WALK/` |
| Voice | `best_trained_voice_model.h5` (joblib), `voice_scaler.pkl` | `Voice Modality Module/Modeling/` |
| Posture | `best_trained_posture_model.pkl` | `Posture Modality Module/` |
| Face | `best_trained_face_model.pkl` | `Face Modality Module/Modeling/` |

---

## Dataset

The dataset used in this project was collected from real patients under ethical approval and with informed consent.  
In compliance with privacy regulations and ethical considerations, the dataset cannot be publicly shared.

---

## Key Dependencies

| Library | Version | Purpose |
|---|---|---|
| `tensorflow` | 2.17.1 | Bidirectional GRU models (Walking) |
| `scikit-learn` | 1.6.1 | Scalers, Random Forest, cross-validation |
| `xgboost` | 2.1.4 | Face AU classification |
| `mediapipe` | — | Pose keypoint extraction |
| `librosa` | — | Voice feature extraction |
| `py-feat` | 0.6.2 | Facial Action Unit detection |
| `tsfel` | — | Time-series feature extraction (Posture) |
| `pymongo` | — | MongoDB data storage and feedback |
| `gradio` | — | Web UI |
| `imbalanced-learn` | — | SMOTE oversampling |
| `joblib` | — | Model serialization |
| `opencv-python` | — | Video frame processing |
| `pyrealsense2` | 2.55.1.6486 | Intel RealSense depth camera support |

---

## License

FILL IN LICENSE INFORMATION

---
