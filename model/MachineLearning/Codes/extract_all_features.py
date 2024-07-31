import os
import numpy as np
import pandas as pd
import librosa
import config as config  # Import the new config module
import pywt
from scipy import signal
from scipy.stats import skew, kurtosis

# Feature extraction functions
def extract_mfcc(audio_data, sr, n_mfcc=13):
    return librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=n_mfcc)

def extract_mel_spectrogram(audio_data, sr, n_mels=128):
    mel_spectrogram = librosa.feature.melspectrogram(y=audio_data, sr=sr, n_mels=n_mels)
    return librosa.power_to_db(mel_spectrogram, ref=np.max)

def extract_spectrogram(audio_data, sr, n_fft=2048, hop_length=512):
    spectrogram = np.abs(librosa.stft(audio_data, n_fft=n_fft, hop_length=hop_length))
    return librosa.amplitude_to_db(spectrogram, ref=np.max)

def psd_with_bands(audio_data, sr, band_width=5):
    freqs, psd = signal.welch(audio_data, sr, nperseg=sr * 1, noverlap=None, scaling='density')
    # band_edges = np.arange(0, freqs[-1] + band_width, band_width)
    # band_powers = []

    # for i in range(len(band_edges) - 1):
    #     low = band_edges[i]
    #     high = band_edges[i + 1]
    #     idx_band = np.logical_and(freqs >= low, freqs < high)
    #     psd_band = psd[idx_band]
    #     avg_power = np.mean(psd_band)
    #     rms_value = np.sqrt(np.mean(np.square(psd_band)))
    #     max_value = np.max(psd_band)
    #     band_powers.append([avg_power, rms_value, max_value])

    # return np.array(band_powers).flatten()
    return psd

def extract_wavelet(audio_data):
    level = 5  # Set the desired level here
    coeffs = pywt.wavedec(audio_data, 'db4', level=level)
    return np.concatenate([coeff for coeff in coeffs])

def extract_spectral_centroid(audio_data, sr):
    return librosa.feature.spectral_centroid(y=audio_data, sr=sr)

def extract_spectral_contrast(audio_data, sr):
    return librosa.feature.spectral_contrast(y=audio_data, sr=sr)

def extract_spectral_bandwidth(audio_data, sr):
    return librosa.feature.spectral_bandwidth(y=audio_data, sr=sr)

def extract_spectral_flatness(audio_data):
    return librosa.feature.spectral_flatness(y=audio_data)

def extract_spectral_rolloff(audio_data, sr):
    return librosa.feature.spectral_rolloff(y=audio_data, sr=sr)

def extract_chroma_stft(audio_data, sr):
    return librosa.feature.spectral_rolloff(y=audio_data, sr=sr)


def extract_zero_crossing_rate(audio_data):
    return librosa.feature.zero_crossing_rate(y=audio_data)

def calculate_statistics(features):
    if features.ndim == 2:
        features_mean = np.mean(features, axis=1)
        features_std = np.std(features, axis=1)
        features_var = np.var(features, axis=1)
        features_median = np.median(features, axis=1)
        features_skewness = skew(features, axis=1)
    else:
        features_mean = np.mean(features)
        features_std = np.std(features)
        features_var = np.var(features)
        features_median = np.median(features)
        features_skewness = skew(features)
    
    combined_features = np.hstack((features_mean, features_std, features_var, features_median, features_skewness))
    return combined_features

def get_audio(audio_path):
    audio_data, sr = librosa.load(audio_path, sr=config.TARGET_SAMPLE_RATE)
    if audio_data.ndim > 1:
        audio_data = audio_data.mean(axis=1)
    if len(audio_data) < config.CHUNK_LENGTH:
        audio_data = np.pad(audio_data, (0, config.CHUNK_LENGTH - len(audio_data)), 'constant')
    else:
        audio_data = audio_data[:config.CHUNK_LENGTH]
    return audio_data, sr

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
    feature_names = []
    for path in audio_paths:
        audio_data, sr = get_audio(path)
        feature_list = []
        feature_name_list = []

        if 'mfcc' in config.FEATURE:
            mfcc_features = extract_mfcc(audio_data, sr)
            mfcc_statistics = calculate_statistics(mfcc_features)
            feature_list.extend(mfcc_statistics)
            feature_name_list.extend([f'mfcc_mean_{i}' for i in range(13)] +
                                     [f'mfcc_std_{i}' for i in range(13)] +
                                     [f'mfcc_var_{i}' for i in range(13)] +
                                     [f'mfcc_median_{i}' for i in range(13)] +
                                     [f'mfcc_skew_{i}' for i in range(13)])
            print(f"MFCC shape: {mfcc_features.shape}")
            print(f"MFCC statistics shape: {mfcc_statistics.shape}")

        if 'mel' in config.FEATURE:
            mel_features = extract_mel_spectrogram(audio_data, sr)
            mel_statistics = calculate_statistics(mel_features)
            feature_list.extend(mel_statistics)
            feature_name_list.extend([f'mel_mean_{i}' for i in range(640)] +
                                     [f'mel_std_{i}' for i in range(640)] +
                                     [f'mel_var_{i}' for i in range(640)] +
                                     [f'mel_median_{i}' for i in range(640)] +
                                     [f'mel_skew_{i}' for i in range(640)])
            print(f"Mel Spectrogram shape: {mel_features.shape}")
            print(f"Mel Spectrogram statistics shape: {mel_statistics.shape}")

        if 'spectrogram' in config.FEATURE:
            spectrogram_features = extract_spectrogram(audio_data, sr)
            spectrogram_statistics = calculate_statistics(spectrogram_features)
            feature_list.extend(spectrogram_statistics)
            feature_name_list.extend([f'spectrogram_mean_{i}' for i in range(5125)] +
                                     [f'spectrogram_std_{i}' for i in range(5125)] +
                                     [f'spectrogram_var_{i}' for i in range(5125)] +
                                     [f'spectrogram_median_{i}' for i in range(5125)] +
                                     [f'spectrogram_skew_{i}' for i in range(5125)])
            print(f"Spectrogram shape: {spectrogram_features.shape}")
            print(f"Spectrogram statistics shape: {spectrogram_statistics.shape}")

        if 'psd' in config.FEATURE:
            psd_features = psd_with_bands(audio_data, sr, band_width=10)
            psd_statistics = calculate_statistics(psd_features)
            feature_list.extend(psd_statistics)
            feature_name_list.extend([f'psd_mean', f'psd_std', f'psd_var', f'psd_median', f'psd_skew'])
            print(f"PSD statistics shape: {psd_statistics.shape}")

        if 'wavelet' in config.FEATURE:
            wavelet_features = extract_wavelet(audio_data)  # Removed level parameter
            wavelet_statistics = calculate_statistics(wavelet_features)
            feature_list.extend(wavelet_statistics)
            feature_name_list.extend([f'wavelet_mean', f'wavelet_std', f'wavelet_var', f'wavelet_median', f'wavelet_skew'])
            print(f"Wavelet shape: {wavelet_features.shape}")
            print(f"Wavelet statistics shape: {wavelet_statistics.shape}")

        if 'zcr' in config.FEATURE:
            zcr_features = extract_zero_crossing_rate(audio_data)
            zcr_statistics = calculate_statistics(zcr_features)
            feature_list.extend(zcr_statistics)
            feature_name_list.extend([f'zcr_mean', f'zcr_std', f'zcr_var', f'zcr_median', f'zcr_skew'])
            print(f"Zero-Crossing Rate shape: {zcr_features.shape}")
            print(f"Zero-Crossing Rate statistics shape: {zcr_statistics.shape}")

        if 'spectral_centroid' in config.FEATURE:
            spectral_centroid_features = extract_spectral_centroid(audio_data, sr)
            spectral_centroid_statistics = calculate_statistics(spectral_centroid_features)
            feature_list.extend(spectral_centroid_statistics)
            feature_name_list.extend([f'spectral_centroid_mean', f'spectral_centroid_std', f'spectral_centroid_var', f'spectral_centroid_median', f'spectral_centroid_skew'])
            print(f"Spectral Centroid shape: {spectral_centroid_features.shape}")
            print(f"Spectral Centroid statistics shape: {spectral_centroid_statistics.shape}")

        if 'spectral_bandwidth' in config.FEATURE:
            spectral_bandwidth_features = extract_spectral_bandwidth(audio_data, sr)
            spectral_bandwidth_statistics = calculate_statistics(spectral_bandwidth_features)
            feature_list.extend(spectral_bandwidth_statistics)
            feature_name_list.extend([f'spectral_bandwidth_mean', f'spectral_bandwidth_std', f'spectral_bandwidth_var', f'spectral_bandwidth_median', f'spectral_bandwidth_skew'])
            print(f"Spectral Bandwidth shape: {spectral_bandwidth_features.shape}")
            print(f"Spectral Bandwidth statistics shape: {spectral_bandwidth_statistics.shape}")

        if 'spectral_contrast' in config.FEATURE:
            spectral_contrast_features = extract_spectral_contrast(audio_data, sr)
            spectral_contrast_statistics = calculate_statistics(spectral_contrast_features)
            feature_list.extend(spectral_contrast_statistics)
            feature_name_list.extend([f'spectral_contrast_mean_{i}' for i in range(7)] +
                                     [f'spectral_contrast_std_{i}' for i in range(7)] +
                                     [f'spectral_contrast_var_{i}' for i in range(7)] +
                                     [f'spectral_contrast_median_{i}' for i in range(7)] +
                                     [f'spectral_contrast_skew_{i}' for i in range(7)])
            print(f"Spectral Contrast shape: {spectral_contrast_features.shape}")
            print(f"Spectral Contrast statistics shape: {spectral_contrast_statistics.shape}")
        
        
        if 'spectral_flatness' in config.FEATURE:
            spectral_flatness_features = extract_spectral_flatness(audio_data)
            spectral_flatness_statistics = calculate_statistics(spectral_flatness_features)
            feature_list.extend(spectral_flatness_statistics)
            feature_name_list.extend([f'spectral_flatness_mean', f'spectral_flatness_std', f'spectral_flatness_var', f'spectral_flatness_median', f'spectral_flatness_skew'])
            print(f"Spectral Flatness shape: {spectral_flatness_features.shape}")
            print(f"Spectral Flatness statistics shape: {spectral_flatness_statistics.shape}")
        
        if 'spectral_rolloff' in config.FEATURE:
            spectral_rolloff_features = extract_spectral_rolloff(audio_data, sr)
            spectral_rolloff_statistics = calculate_statistics(spectral_rolloff_features)
            feature_list.extend(spectral_rolloff_statistics)
            feature_name_list.extend([f'spectral_rolloff_mean', f'spectral_rolloff_std', f'spectral_rolloff_var', f'spectral_rolloff_median', f'spectral_rolloff_skew'])
            print(f"Spectral Rolloff shape: {spectral_rolloff_features.shape}")
            print(f"Spectral Rolloff statistics shape: {spectral_rolloff_statistics.shape}")

        features.append([path] + feature_list)
        if not feature_names:
            feature_names.extend(['path'] + feature_name_list)

    features_array = np.array(features)
    print(f"Features array shape: {features_array.shape}")
    print(f"Number of feature names: {len(feature_names)}")
    return features_array, np.array(labels), feature_names


def save_features_to_csv(features, labels, feature_names, output_file):
    print(f"Saving features with shape: {features.shape} and {len(feature_names)} feature names")
    df = pd.DataFrame(features, columns=feature_names[:features.shape[1]])
    df['label'] = labels
    df.to_csv(output_file, index=False)

def main():
    annotations_dir = os.path.join(config.MAIN_DATA_DIR, config.DATA_TYPE,f'fold{config.fold}')
    train_annotations_file = os.path.join(annotations_dir, 'train.txt')
    #val_annotations_file = os.path.join(annotations_dir, 'val.txt')
    test_annotations_file = os.path.join(annotations_dir, 'test.txt')

    output_train_file = os.path.join(config.MAIN_SAVE_DIR, config.DATA_TYPE, f'{config.DATA_TYPE}_train_features{config.WAYS}.csv')
    output_test_file = os.path.join(config.MAIN_SAVE_DIR, config.DATA_TYPE, f'{config.DATA_TYPE}_test_features{config.WAYS}.csv')
    print(output_train_file)

    os.makedirs(os.path.dirname(output_train_file), exist_ok=True)

    # Load data
    train_audio_paths, train_labels = load_data(train_annotations_file)
    #val_audio_paths, val_labels = load_data(val_annotations_file)
    test_audio_paths, test_labels = load_data(test_annotations_file)

    # Combine train and validation data for training
    # all_train_audio_paths = train_audio_paths + val_audio_paths
    # all_train_labels = train_labels + val_labels

    # Extract features and print shapes
    train_features, train_labels, feature_names = extract_features(train_audio_paths, train_labels)
    test_features, test_labels, _ = extract_features(test_audio_paths, test_labels)

    # Save features to CSV for inspection
    save_features_to_csv(train_features, train_labels, feature_names, output_train_file)
    save_features_to_csv(test_features, test_labels, feature_names, output_test_file)

if __name__ == "__main__":
    main()