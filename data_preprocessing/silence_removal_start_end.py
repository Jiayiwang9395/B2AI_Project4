import os
import wave
import numpy as np
import warnings
import shutil

def read_wav(filename):
    with wave.open(filename, 'rb') as wav_file:
        n_channels, sampwidth, framerate, n_frames, _, _ = wav_file.getparams()
        frames = wav_file.readframes(n_frames)
        audio_data = np.frombuffer(frames, dtype=np.int16)
        if n_channels == 2:
            audio_data = audio_data[::2]  # If stereo, just take one channel
    return audio_data, framerate

def write_wav(filename, audio_data, framerate):
    with wave.open(filename, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(framerate)
        wav_file.writeframes(audio_data.tobytes())

def trim_silence(audio_data, framerate, silence_threshold=100, min_silence_len=0.8):
    chunk_size = int(0.01 * framerate)  # 10ms chunks
    min_silence_samples = int(min_silence_len * framerate)
    
    # Calculate RMS for each chunk
    num_chunks = len(audio_data) // chunk_size
    audio_data = audio_data[:num_chunks * chunk_size]
    chunks = audio_data.reshape((num_chunks, chunk_size))
    
    # Use np.maximum to ensure we don't take the square root of negative values
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rms = np.sqrt(np.maximum(np.mean(np.square(chunks.astype(np.float64)), axis=1), 0))
    
    # Find where audio starts and ends
    is_sound = rms > silence_threshold
    sound_starts = np.where(is_sound)[0]
    
    if len(sound_starts) == 0:
        return audio_data  # Return original if no sound found
    
    start_trim = sound_starts[0]
    end_trim = sound_starts[-1] + 1
    
    # Ensure we're not cutting off less than min_silence_len
    if start_trim * chunk_size > min_silence_samples:
        start_trim = start_trim * chunk_size
    else:
        start_trim = 0
    
    if (num_chunks - end_trim) * chunk_size > min_silence_samples:
        end_trim = end_trim * chunk_size
    else:
        end_trim = len(audio_data)
    
    return audio_data[start_trim:end_trim]

def process_directory(input_dir, output_dir):
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.wav'):
                input_path = os.path.join(root, file)
                relative_path = os.path.relpath(input_path, input_dir)
                # print(relative_path)
                output_path = os.path.join(output_dir, relative_path)
                
                # Create the output directory if it doesn't exist
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                try:
                    # Read audio file
                    audio_data, framerate = read_wav(input_path)
                    
                    # Trim silence from start and end
                    trimmed_audio = trim_silence(audio_data, framerate, silence_threshold=100, min_silence_len=0.8)
                    
                    # Save trimmed audio
                    write_wav(output_path, trimmed_audio, framerate)
                    
                    # print(f"Processed: {relative_path}")
                except wave.Error as e:
                    print(f"Error processing file: {input_path}")
                    print(f"Error details: {str(e)}")
                except Exception as e:
                    print(f"Unexpected error processing file: {input_path}")
                    print(f"Error details: {str(e)}")


# Define input and output directories
input_dir = r"C:\Users\seera\OneDrive\Desktop\B2AI\data\filtered_data"
output_dir = r"C:\Users\seera\OneDrive\Desktop\B2AI\data\Silence_removal_b2aiDB\Filtered_StridorDB"
# input_dir = r"C:\Users\seera\OneDrive\Desktop\B2AI\data\Filtered_b2aiDB\Filtered_b2aiDB"
# output_dir = r"C:\Users\seera\OneDrive\Desktop\B2AI\data\Silence_removal_b2aiDB\Filtered_b2aiDB"


# Clean the output directory if it exists, or create it
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir)

# Process all files
process_directory(input_dir, output_dir)

print(f"All files processed. Results saved in {output_dir}")