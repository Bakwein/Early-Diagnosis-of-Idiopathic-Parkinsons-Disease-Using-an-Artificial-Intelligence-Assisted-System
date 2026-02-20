import os
import numpy as np
import librosa

def extract_voice_features_single(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=44100)
        features = []

        # Feature Extraction
        features.append(np.mean(librosa.feature.zero_crossing_rate(y=y).T, axis=0))
        features.append(np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0))
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        features.extend([
            np.mean(mfcc.T, axis=0),
            np.mean(librosa.feature.delta(mfcc).T, axis=0),
            np.mean(librosa.feature.delta(mfcc, order=2).T, axis=0)
        ])
        features.append(np.mean(librosa.feature.rms(y=y).T, axis=0))
        features.append(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr).T, axis=0))
        features.append(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr).T, axis=0))
        features.append(np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0))
        features.append(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr).T, axis=0))
        features.append(np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr).T, axis=0))

        flat_features = np.hstack(features)
        flat_features = np.delete(flat_features, [17, 22])

        return flat_features  # Type: np.ndarray
    except Exception as e:
        raise RuntimeError(f"Voice feature extraction failed: {str(e)}")
