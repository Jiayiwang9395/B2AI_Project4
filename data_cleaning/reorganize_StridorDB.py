import os
import shutil
import re

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

def has_wav_files(root, files):
    return any(should_copy_file(file) for file in files)

def clean_destination_folder(dst):
    if os.path.exists(dst):
        for root, dirs, files in os.walk(dst, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                shutil.rmtree(os.path.join(root, name))

def remove_spaces(name):
    return re.sub(r'\s+', '-', name)

def copy_filtered_files(src, dst):
    for root, dirs, files in os.walk(src):
        rel_path = os.path.relpath(root, src)
        path_parts = rel_path.split(os.sep)
        
        if not has_wav_files(root, files):
            continue
        
        dst_folder = None
        if "OLDMETHOD" in path_parts:
            if "CONTROL" in path_parts:
                if len(path_parts) > 2:
                    dst_folder = os.path.join(dst, remove_spaces(path_parts[3]))
            else:
                if len(path_parts) > 2:
                    dst_folder = os.path.join(dst, remove_spaces(path_parts[2]))
        elif "UPDATEDMETHOD" in path_parts:
            if "CONTROLS" in path_parts:
                if len(path_parts) > 2:
                    dst_folder = os.path.join(dst, remove_spaces(path_parts[2]))
            else:
                if len(path_parts) > 1:
                    dst_folder = os.path.join(dst, remove_spaces(path_parts[1]))
        
        if dst_folder is None:
            continue

        for file in files:
            if should_copy_file(file):
                src_file = os.path.join(root, file)
                
                base_name, extension = os.path.splitext(file)
                counter = 1
                dst_file = os.path.join(dst_folder, remove_spaces(file))
                while os.path.exists(dst_file):
                    dst_file = os.path.join(dst_folder, f"{remove_spaces(base_name)}_{counter}{extension}")
                    counter += 1
                
                os.makedirs(dst_folder, exist_ok=True)
                
                shutil.copy2(src_file, dst_file)
                # print(f"Copied: {src_file} -> {dst_file}")

# Clean the destination folder before copying files
clean_destination_folder(dest_dir)

# Run the function to copy files
copy_filtered_files(source_dir, dest_dir)