import os
import pandas as pd
import numpy as np

def stratified_split_by_recordings(data, train_size, val_size, test_size, random_state=42):
    assert train_size + val_size + test_size == 1, "The sum of train_size, val_size, and test_size must be 1."
    
    patients = data['patient'].unique()
    np.random.seed(random_state)
    np.random.shuffle(patients)

    train_patients, val_patients, test_patients = [], [], []
    train_count, val_count, test_count = 0, 0, 0
    total_count = len(data)

    for patient in patients:
        patient_data = data[data['patient'] == patient]
        patient_recordings = len(patient_data)

        if train_count / total_count < train_size:
            train_patients.append(patient)
            train_count += patient_recordings
        elif val_count / total_count < val_size:
            val_patients.append(patient)
            val_count += patient_recordings
        else:
            test_patients.append(patient)
            test_count += patient_recordings

    train_data = data[data['patient'].isin(train_patients)]
    val_data = data[data['patient'].isin(val_patients)]
    test_data = data[data['patient'].isin(test_patients)]
    
    return train_data, val_data, test_data, train_patients, val_patients, test_patients

def check_labels(data):
    labels = data['label'].unique()
    return set(labels) == {0, 1}

def main():
    main_path = '/data/jiayiwang/summerschool/models/Data_preprocessing'
    data_type = 'Reg'
    main_file_path = os.path.join(main_path, f'{data_type}_chunk_Stridor.txt')
    output_dir = os.path.join(main_path, data_type)
    
    os.makedirs(output_dir, exist_ok=True)

    data = pd.read_csv(main_file_path, sep=' ', header=None, names=['path', 'label'])
    data['patient'] = data['path'].apply(lambda x: x.split('/')[-2])

    train_data, val_data, test_data, train_patients, val_patients, test_patients = stratified_split_by_recordings(
        data, train_size=0.70, val_size=0.10, test_size=0.20)

    if check_labels(train_data) and check_labels(val_data) and check_labels(test_data):
        print("Each dataset contains both labels.")
    else:
        print("One or more datasets are missing a label.")

    train_data[['path', 'label']].to_csv(os.path.join(output_dir, 'train.txt'), sep=' ', header=False, index=False)
    val_data[['path', 'label']].to_csv(os.path.join(output_dir, 'val.txt'), sep=' ', header=False, index=False)
    test_data[['path', 'label']].to_csv(os.path.join(output_dir, 'test.txt'), sep=' ', header=False, index=False)

    print("Data split completed and saved to txt files.")

    print("Label distribution for training set:")
    print(train_data['label'].value_counts().sort_index())
    print("Label distribution for validation set:")
    print(val_data['label'].value_counts().sort_index())
    print("Label distribution for test set:")
    print(test_data['label'].value_counts().sort_index())

    print("\nNumber of patients in training set:", len(train_patients))
    print("Patients in training set:", train_patients)
    print("\nNumber of patients in validation set:", len(val_patients))
    print("Patients in validation set:", val_patients)
    print("\nNumber of patients in test set:", len(test_patients))
    print("Patients in test set:", test_patients)

if __name__ == "__main__":
    main()
