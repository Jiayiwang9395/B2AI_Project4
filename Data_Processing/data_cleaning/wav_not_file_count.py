import os

def find_wav_files_without_words(directory, search_words):
    results = []
    search_words = [word.lower() for word in search_words]
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.wav'):
                if not any(word in file.lower() for word in search_words):
                    parent_dir = os.path.basename(root)
                    results.append((parent_dir, file))
    return results

# Replace this with the path to your folder
folder_path = r"C:\Users\seera\OneDrive\Desktop\B2AI\data\pre_filtered_data"

# The words you're looking for in the filenames
search_words = ["fimo", "reg", "rp", "deep", "rmo"]  # Add all words you want to search for

files_without_words = find_wav_files_without_words(folder_path, search_words)

print(f"Files that don't contain any of the search words {search_words}:")
for parent_dir, file in files_without_words:
    print(f"Parent Directory: {parent_dir}, File: {file}")

print(f"\nTotal number of .wav files without search words: {len(files_without_words)}")