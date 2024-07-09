import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import os


def load_audio(file_path):
    audio, sr = librosa.load(file_path, sr=None)
    print(sr)
    return audio, sr

def chunk_audio(audio, sr, chunk_length_s=4, overlap=0.5):
    chunk_length = int(sr * chunk_length_s)  # Convert seconds to samples
    step = int(chunk_length * (1 - overlap))
    chunks = []
    num_samples = len(audio)
    
    # Pad audio with zeros if necessary
    if num_samples % chunk_length != 0:
        padding = chunk_length - (num_samples % chunk_length)
        audio = np.pad(audio, (0, padding), 'constant')
    
    print(f'Padded audio length: {len(audio)/sr} seconds')
    
    for start in range(0, len(audio) - chunk_length + 1, step):
        chunks.append(audio[start:start + chunk_length])
        print(f'{start/sr}:{(start + chunk_length)/sr}')
    return np.array(chunks)

def save_chunks(chunks, sr, output_dir, base_filename):
    for i, chunk in enumerate(chunks):
        output_path = f"{output_dir}/{base_filename}_chunk{i+1}.wav"
        sf.write(output_path, chunk, sr)
        print(f"Saved chunk {i+1} to {output_path}")

def plot_waveform(audio, sr, title='Audio Waveform'):
    plt.figure(figsize=(14, 5))
    times = np.arange(len(audio)) / sr
    plt.plot(times, audio)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title(title)
    plt.show()

def main():
    # Example usage for both breathing and reading recordings
    main_path = '/Users/oliviawang/Desktop/summerschool/data/'
    data_type = 'Stridor'
    audio_path = os.path.join(main_path, f'Filtered_{data_type}DB')
    output_dir_base = os.path.join(main_path, 'chunks_DB',f'chunk_{data_type}')

    for subdir, dirs, files in os.walk(audio_path):
        for file in files:
            if file.endswith('.wav'):
                audio_path = os.path.join(subdir, file)
                #Find the Patient number
                patient_folder = os.path.basename(subdir)
                #Create the patient folder in the output dir 
                output_dir = os.path.join(output_dir_base, patient_folder)
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                    print(f"Created directory: {output_dir}")
                base_filename = file.split('.')[0]
                #print(base_filename)
                audio,sr = load_audio(audio_path)
                chunks = chunk_audio(audio, sr, chunk_length_s=4, overlap=0.5)
                save_chunks(chunks, sr, output_dir, base_filename)

    # # Load the audio
    # audio, sr = load_audio(audio_path)

    # # Chunk the audio
    # chunks = chunk_audio(audio, sr, chunk_length_s=4, overlap=0.5)

    # # Save the chunks to separate WAV files
    # save_chunks(chunks, sr, output_dir, base_filename)
    # plot_waveform(audio, sr, title='Full Audio Waveform')


if __name__ == "__main__":
    main()