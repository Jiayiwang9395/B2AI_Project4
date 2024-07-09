import os
import re

def rename_s_to_ns(main_directory, folders_to_process):
    pattern = re.compile(r'(.+_)S\.wav$')

    for folder in folders_to_process:
        directory = os.path.join(main_directory, folder)
        if not os.path.exists(directory):
            print(f"Folder not found: {directory}")
            continue

        print(f"Processing folder: {folder}")
        for root, dirs, files in os.walk(directory):
            for filename in files:
                match = pattern.match(filename)
                if match:
                    old_path = os.path.join(root, filename)
                    base = match.group(1)
                    new_filename = f"{base}NS.wav"
                    new_path = os.path.join(root, new_filename)

                    try:
                        os.rename(old_path, new_path)
                        print(f"Renamed: {filename} -> {new_filename}")
                    except OSError as e:
                        print(f"Error renaming {filename}: {e}")

    print("Processing complete.")


# Specify the main directory path
main_directory = r"C:\Users\seera\OneDrive\Desktop\B2AI\data\filtered_data"

# List of folders to process
folders_to_process = [
    "Patient-14",
    "Patient-15",
    "Patient-19",
    "Patient-20",
    "Patient-26",
    "Patient-38"
]

# Call the function
rename_s_to_ns(main_directory, folders_to_process)