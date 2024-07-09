import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchaudio
import numpy as np

class AudioDataset(Dataset):
    def __init__(self, annotations_file, target_sample_rate=44100, transform=None, feature_type='mel_spectrogram', mean=None, std=None):
        self.annotations = pd.read_csv(annotations_file, sep=' ', header=None, names=['path', 'label'])
        self.transform = transform
        self.target_sample_rate = target_sample_rate
        self.feature_type = feature_type
        self.mean = mean
        self.std = std
        if self.feature_type == 'mfcc' and (self.mean is None or self.std is None):
            self.calculate_mfcc_stats()

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        try:
            audio_path = self.annotations.iloc[idx, 0]
            label = int(self.annotations.iloc[idx, 1])
            waveform, sample_rate = torchaudio.load(audio_path)

            if sample_rate != self.target_sample_rate:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.target_sample_rate)
                waveform = resampler(waveform)

            if self.feature_type == 'mel_spectrogram':
                transform = torchaudio.transforms.MelSpectrogram(sample_rate=self.target_sample_rate, n_mels=128)
                features = transform(waveform)
                features = torchaudio.transforms.AmplitudeToDB()(features)
                features = torch.nn.functional.interpolate(features.unsqueeze(0), size=(96, 64)).squeeze(0)  # Reshape to [96, 64]
                # Ensure the shape is [batch_size, channels, height, width] where channels=1
            elif self.feature_type == 'mfcc':
                transform = torchaudio.transforms.MFCC(sample_rate=self.target_sample_rate)
                features = transform(waveform)
                features = (features - self.mean) / self.std 
            else:
                features = waveform

            if self.transform:
                features = self.transform(features)

            return audio_path, features, self.target_sample_rate, label
        except Exception as e:
            print(f"Error processing index {idx}: {e}")
            raise

    def calculate_mfcc_stats(self):
        all_mfccs = []
        for idx in range(len(self.annotations)):
            audio_path = self.annotations.iloc[idx, 0]
            waveform, sample_rate = torchaudio.load(audio_path)
            if sample_rate != self.target_sample_rate:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.target_sample_rate)
                waveform = resampler(waveform)
            transform = torchaudio.transforms.MFCC(sample_rate=self.target_sample_rate)
            mfcc = transform(waveform)
            all_mfccs.append(mfcc)
        all_mfccs = torch.cat(all_mfccs, dim=0)
        self.mean = all_mfccs.mean(dim=0, keepdim=True)
        self.std = all_mfccs.std(dim=0, keepdim=True)
