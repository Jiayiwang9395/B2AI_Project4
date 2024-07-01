import os
import re

# Source directory
source_dir = r"C:\Users\seera\OneDrive\Desktop\B2AI\data\filtered_data"

def get_breath_type(filename):
    filename_lower = filename.lower()
    if any(term in filename_lower for term in ['regular breath', 'rmo', 'reg', '2m', '2a']):
        return 'Reg'
    elif any(term in filename_lower for term in ['fimo', '4m', '4a' ]):
        return 'FIMO'
    elif any(term in filename_lower for term in ['deep breath', 'deep']):
        return 'Deep'
    elif any(term in filename_lower for term in ['rainbow passage', 'rp', 'ravid', '7m', '7a']):
        return 'RP'
    return ''

def get_device_type(filename):
    filename_lower = filename.lower()
    if any(term in filename_lower for term in ['avid', 'avid-flat', 'avid1in', 'avid2in', 'avidn', '2a', '4a', '7a']):
        return 'Avid'
    elif any(term in filename_lower for term in ['12 inch', '12 inch-flat', '12in&3in', '12', '12inch', '12inch-flat', '2m', '4m', '7m']):
        return '12inch'
    elif any(term in filename_lower for term in ['ipad', '6 inch']):
        return 'iPad'
    elif any(term in filename_lower for term in ['1inch', '1in']):
        return '1inch'
    return ''

def should_rename_file(filename):
    return filename.lower().endswith('.wav')

def rename_files(directory):
    for root, dirs, files in os.walk(directory):
        folder_name = os.path.basename(root).replace(" ", "")
        is_patient = 'patient' in folder_name.lower()
        
        # Keep track of used filenames in this folder
        used_filenames = set()
        
        for file in files:
            if should_rename_file(file):
                old_path = os.path.join(root, file)
                
                breath_type = get_breath_type(file)
                device_type = get_device_type(file)
                subject_type = 'S' if is_patient else 'NS'
                
                # Start with no counter
                counter = 0
                while True:
                    new_parts = [folder_name]
                    if counter > 0:
                        new_parts.append(str(counter))
                    if breath_type:
                        new_parts.append(breath_type)
                    if device_type:
                        new_parts.append(device_type)
                    new_parts.append(subject_type)
                    
                    new_filename = '_'.join(new_parts) + '.wav'
                    new_path = os.path.join(root, new_filename)
                    
                    if new_filename not in used_filenames and not os.path.exists(new_path):
                        # We found a unique filename
                        used_filenames.add(new_filename)
                        break
                    
                    # Increment counter and try again
                    counter += 1
                
                # Rename the file
                os.rename(old_path, new_path)
                # print(f"Renamed: {old_path} -> {new_path}")

# Run the function
rename_files(source_dir)