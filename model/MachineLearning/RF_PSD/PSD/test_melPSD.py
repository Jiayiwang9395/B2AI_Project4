import os
import numpy as np
import librosa
from scipy import signal
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from imblearn.over_sampling import SMOTE

# Constants
TARGET_SAMPLE_RATE = 16000  # Target sample rate for resampling
CHUNK_LENGTH_S = 4  # Chunk length in seconds
CHUNK_LENGTH = TARGET_SAMPLE_RATE * CHUNK_LENGTH_S  # Convert seconds to samples

def psd_with_bands(cfg, audio_data, band_width=5):
    freqs, psd = signal.welch(audio_data, cfg.FEATS.FS, nperseg=cfg.FEATS.FS * cfg.FEATS.WINDOW,
                              noverlap=None, scaling='density')
    band_edges = np.arange(0, freqs[-1] + band_width, band_width)
    band_powers = []

    for i in range(len(band_edges) - 1):
        low = band_edges[i]
        high = band_edges[i + 1]
        idx_band = np.logical_and(freqs >= low, freqs < high)
        psd_band = psd[idx_band]
        avg_power = np.mean(psd_band)
        rms_value = np.sqrt(np.mean(np.square(psd_band)))
        max_value = np.max(psd_band)
        band_powers.append([avg_power, rms_value, max_value])

    return np.array(band_powers).flatten()

def mel_spectrogram(cfg, audio_data):
    mel_spec = librosa.feature.melspectrogram(y=audio_data, sr=cfg.FEATS.FS, n_mels=128, fmax=8000)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db.flatten()

def get_audio(audio_path):
    audio_data, sr = librosa.load(audio_path, sr=TARGET_SAMPLE_RATE)  # Resample to target sample rate
    if audio_data.ndim > 1:
        audio_data = audio_data.mean(axis=1)

    # Ensure the audio data is of the fixed length
    if len(audio_data) < CHUNK_LENGTH:
        audio_data = np.pad(audio_data, (0, CHUNK_LENGTH - len(audio_data)), 'constant')
    else:
        audio_data = audio_data[:CHUNK_LENGTH]

    class Config:
        class FEATS:
            FS = sr
            WINDOW = 1  # Window size in seconds

    return audio_data, Config

def load_data(annotation_file):
    audio_paths = []
    labels = []
    with open(annotation_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            audio_paths.append(parts[0])
            labels.append(int(parts[1]))
    return audio_paths, labels

def extract_features(audio_paths, labels):
    features = []
    for path in audio_paths:
        audio_data, cfg = get_audio(path)
        psd_features = psd_with_bands(cfg, audio_data)
        mel_features = mel_spectrogram(cfg, audio_data)
        combined_features = np.concatenate((psd_features, mel_features))
        features.append(combined_features)
    return np.array(features), np.array(labels)

def plot_confusion_matrix(cm, classes, save_path, title='Confusion Matrix'):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(save_path)
    plt.close()

def main():
    main_data_dir = '/data/jiayiwang/summerschool/models/Data_preprocessing/'
    data_type = 'Reg'
    ways = '_MelPSD_Oversampling'

    annotations_dir = os.path.join(main_data_dir, data_type)
    test_annotations_file = os.path.join(annotations_dir, 'test.txt')
    val_annotations_file = os.path.join(annotations_dir, 'val.txt')
    train_annotations_file = os.path.join(annotations_dir, 'train.txt')

    main_save_dir = '/data/jiayiwang/summerschool/models/saved/PSD'
    model_type = 'RandomForest'
    save_path = os.path.join(main_save_dir, data_type, model_type)
    os.makedirs(save_path, exist_ok=True)

    # Load data
    train_audio_paths, train_labels = load_data(train_annotations_file)
    val_audio_paths, val_labels = load_data(val_annotations_file)
    test_audio_paths, test_labels = load_data(test_annotations_file)

    # Extract features
    train_features, train_labels = extract_features(train_audio_paths, train_labels)
    val_features, val_labels = extract_features(val_audio_paths, val_labels)
    test_features, test_labels = extract_features(test_audio_paths, test_labels)

    # Combine train and validation data for training
    X_train = np.concatenate((train_features, val_features))
    y_train = np.concatenate((train_labels, val_labels))

    # # Oversample the minority class using SMOTE
    # smote = SMOTE(random_state=42)
    # X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Train Random Forest model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    # rf_model.fit(X_train_resampled, y_train_resampled)
    rf_model.fit(X_train, y_train)

    # Evaluate on test set
    y_pred = rf_model.predict(test_features)
    accuracy = accuracy_score(test_labels, y_pred)
    report = classification_report(test_labels, y_pred)
    cm = confusion_matrix(test_labels, y_pred)

    print(f'Test Accuracy: {accuracy}')
    print(report)

    # Save the model
    joblib.dump(rf_model, os.path.join(save_path, f'rf_model{ways}.pkl'))

    # Save the classification report
    with open(os.path.join(save_path, f'classification_report{ways}.txt'), 'w') as f:
        f.write(f'Test Accuracy: {accuracy}\n\n')
        f.write(report)

    # Save the confusion matrix plot
    plot_confusion_matrix(cm, classes=[0, 1], save_path=os.path.join(save_path, f'confusion_matrix{ways}.png'))

    # Save test results
    with open(os.path.join(save_path, f'test_results{ways}.txt'), 'w') as f:
        for path, true_label, pred_label in zip(test_audio_paths, test_labels, y_pred):
            f.write(f"{path}\t{true_label}\t{pred_label}\n")

if __name__ == "__main__":
    main()
