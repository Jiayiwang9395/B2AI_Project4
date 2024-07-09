import librosa
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import freeze_support

def extract_features(file_path, label):
    try:
        # Extract details from filename
        filename = os.path.basename(file_path)
        parts = filename.split('_')
        
        # Example filename: Control-10C_ses-1_FIMO_Avid_NS_chunk1
        record_type = parts[0]  # Control-10C
        session = parts[1]       # ses-1
        data_type = parts[2]     # FIMO
        device_type = parts[3]   # Avid
        file_name = '_'.join(parts[:-1])  # Control-10C_ses-1_FIMO_Avid_NS_chunk1
        
        # Load audio file and extract features
        y, sr = librosa.load(file_path)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
        
        features = {
            'mfcc_mean': np.mean(mfccs, axis=1),
            'mfcc_var': np.var(mfccs, axis=1),
            'spectral_centroid_mean': np.mean(spectral_centroid),
            'spectral_bandwidth_mean': np.mean(spectral_bandwidth),
            'spectral_contrast_mean': np.mean(spectral_contrast),
            'spectral_rolloff_mean': np.mean(spectral_rolloff),
            'zero_crossing_rate_mean': np.mean(zero_crossing_rate),
            'record_number': record_type,
            'session_number': session,
            'data_type': data_type,
            'device_type': device_type,
            'file_name': file_name,
            'label': label
        }
        
        return features
    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")
        return None

def process_batch(batch):
    return [extract_features(row['path'], row['label']) for _, row in batch.iterrows()]

def extract_features_for_dataset(input_file_path, output_file_path):
    # Read your file with paths and labels
    data = pd.read_csv(input_file_path, sep=' ', header=None, names=['path', 'label'])

    # Set batch size
    batch_size = 1000

    # Process files in batches
    features_list = []
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = []
        for i in range(0, len(data), batch_size):
            batch = data.iloc[i:i+batch_size]
            futures.append(executor.submit(process_batch, batch))
        
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing {input_file_path}"):
            batch_results = future.result()
            features_list.extend([f for f in batch_results if f is not None])

    # Create a DataFrame with all features
    df_features = pd.DataFrame(features_list)

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    # Save features to an Excel file
    df_features.to_excel(output_file_path, index=False)

    print(f"Features extracted and saved to {output_file_path}")

def main():
    main_path = r'C:\Users\seera\OneDrive\Desktop\B2AI\data\proceessing'
    data_types = ['FIMO', 'RP', 'Deep', 'Reg']  # Specify the data types

    for data_type in data_types:
        for dataset_type in ['train', 'test', 'val']:
            input_file_path = os.path.join(main_path, 'tables', data_type, f'{dataset_type}.txt')
            output_dir = os.path.join(main_path, 'audio_features', data_type)
            output_file_path = os.path.join(output_dir, f'{dataset_type}.xlsx')

            extract_features_for_dataset(input_file_path, output_file_path)

if __name__ == '__main__':
    freeze_support()
    main()
