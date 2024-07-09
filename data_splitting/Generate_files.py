import os

# Define the base directory and output files
base_dir = r'C:\Users\seera\OneDrive\Desktop\B2AI\data\chunk_data\chunk'
data_type = 'Stridor'
base_dir = os.path.join(base_dir, f'chunk_{data_type}')
print(f'base_dir is {base_dir}')

output_dir = 'C:/Users/seera/OneDrive/Desktop/B2AI/data/proceessing/tables'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)  # Create the output directory if it doesn't exist

output_files = {
    'FIMO': f'FIMO_chunk_{data_type}.txt',
    'Reg': f'Reg_chunk_{data_type}.txt',
    'Deep': f'Deep_chunk_{data_type}.txt',
    'RP': f'RP_chunk_{data_type}.txt'
}

# Print base directory and output file paths for debugging
for key, output_file in output_files.items():
    output_path = os.path.join(output_dir, output_file)
    # print(f"Output file for {key}: {output_path}")

# Open output files in write mode
output_handlers = {key: open(os.path.join(output_dir, output_file), 'w') for key, output_file in output_files.items()}

#Iterate through each sub-folder and file
for subdir, dirs, files in os.walk(base_dir):
    for file in files:
        
        if file.endswith('.wav'):
            file_path = os.path.join(subdir, file)
            #print(file_path)
            print(file)
            #need to check on yout path
            stridor = file.split('_')[4]
            print(stridor)
            if stridor =='S':
                label = 1
            elif stridor == 'NS':
                label = 0
            else:
                print('Label is Wrong')
                print(stridor)
            for key in output_files.keys():
                if key in file:
                    output_handlers[key].write(f"{file_path} {label}\n")
                    #print(f"Wrote to {key} file: {file_path} {label}")  # Debug print

# Close all output files
for handler in output_handlers.values():
    handler.close()

print("Done processing the folders and files.")
