import os
import shutil

# Source and destination directories
source_dir = r"C:\Users\seera\OneDrive\Desktop\B2AI\data\pre_filtered_data"
dest_dir = r"C:\Users\seera\OneDrive\Desktop\B2AI\data\filtered_data"

# List of names to include (converted to lowercase for case-insensitive matching)
NAMES_TO_INCLUDE = [
    "fimo", "rp", "deep", "rmo", "ravid", "reg",
    "2m", "2a",  "4m",  "4a",  "7m",  "7a"
]

def should_copy_file(filename):
    if not filename.lower().endswith('.wav'):
        return False
    
    if "thyroid" in filename.lower() or "cricoid" in filename.lower():
        return False
    
    file_name_lower = os.path.splitext(filename)[0].lower()
    return any(substring in file_name_lower for substring in NAMES_TO_INCLUDE)
    
    # return False

def has_wav_files(root, files):
    for file in files:
        if should_copy_file(file):
            return True
    return False

def clean_destination_folder(dst):
    if os.path.exists(dst):
        for root, dirs, files in os.walk(dst, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                shutil.rmtree(os.path.join(root, name))

def copy_filtered_files(src, dst):
    for root, dirs, files in os.walk(src):
        rel_path = os.path.relpath(root, src)
        path_parts = rel_path.split(os.sep)
        
        # Skip folders that don't contain any .wav files matching the criteria
        if not has_wav_files(root, files):
            continue
        
        # Determine the destination folder based on the path
        dst_folder = None
        if "OLDMETHOD" in path_parts:
            if "CONTROL" in path_parts:
                if len(path_parts) > 2:  # Expecting CONTROLS/Control 4C, etc.
                    # print(path_parts, path_parts[3])
                    dst_folder = os.path.join(dst, path_parts[3])
            else:
                if len(path_parts) > 2:  # Expecting INTIAL/Patient1, REVISED/Patient1, etc.
                    # print("else-->", path_parts, path_parts[2])
                    dst_folder = os.path.join(dst, path_parts[2])
        elif "UPDATEDMETHOD" in path_parts:
            if "CONTROLS" in path_parts:
                if len(path_parts) > 2:  # Expecting CONTROLS/Control 4C, etc.
                    # print(path_parts)
                    dst_folder = os.path.join(dst, path_parts[2])
            else:
                if len(path_parts) > 1:  # Expecting Patient1, Patient2, etc.
                    # print(path_parts)
                    dst_folder = os.path.join(dst, path_parts[1])
        
        if dst_folder is None:
            continue  # Skip if we couldn't determine the destination folder
        
        for file in files:
            if should_copy_file(file):
                src_file = os.path.join(root, file)
                
                # Create a unique filename if a file with the same name already exists
                base_name, extension = os.path.splitext(file)
                counter = 1
                dst_file = os.path.join(dst_folder, file)
                while os.path.exists(dst_file):
                    dst_file = os.path.join(dst_folder, f"{base_name}_{counter}{extension}")
                    counter += 1
                
                # Create the destination directory if it doesn't exist
                os.makedirs(dst_folder, exist_ok=True)
                
                shutil.copy2(src_file, dst_file)
                # print(f"Copied: {src_file} -> {dst_file}")

# Clean the destination folder before copying files
clean_destination_folder(dest_dir)

# Run the function to copy files
copy_filtered_files(source_dir, dest_dir)
